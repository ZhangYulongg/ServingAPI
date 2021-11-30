import os
import subprocess
import numpy as np
import copy
import cv2
import requests
import json
import sys

from paddle_serving_server.pipeline import PipelineClient
from paddle_serving_app.reader import CenterCrop, RGB2BGR, Transpose, Div, Normalize, RCNNPostprocess
from paddle_serving_app.reader import Sequential, File2Image, Resize, Transpose, BGR2RGB, SegPostprocess
import paddle.inference as paddle_infer

sys.path.append("../")
from util import *


class TestImagenetPipeline(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="imagenet", example_path="Pipeline/PaddleClas/imagenet", model_dir="ResNet50_vd_model",
                                   client_dir="ResNet50_vd_client_config")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util

    def teardown_method(self):
        print_log(["stderr.log", "stdout.log",
                   "log/serving.ERROR", "PipelineServingLogs/pipeline.log"], iden="after predict")
        kill_process(9999)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        seq = Sequential([
            File2Image(), Resize(256), CenterCrop(224), RGB2BGR(), Transpose((2, 0, 1)),
            Div(255), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True)
        ])
        filename = "daisy.jpg"
        im = seq(filename)[np.newaxis, :]
        input_dict = {}
        input_dict["image"] = im.astype("float32")

        pd_config = paddle_infer.Config("ResNet50_vd_model")
        pd_config.disable_gpu()
        pd_config.switch_ir_optim(False)

        predictor = paddle_infer.create_predictor(pd_config)

        input_names = predictor.get_input_names()
        for i, input_name in enumerate(input_names):
            input_handle = predictor.get_input_handle(input_name)
            input_handle.copy_from_cpu(input_dict[input_name])

        predictor.run()

        output_data_dict = {}
        output_names = predictor.get_output_names()
        for _, output_data_name in enumerate(output_names):
            output_handle = predictor.get_output_handle(output_data_name)
            output_data = output_handle.copy_to_cpu()
            output_data_dict[output_data_name] = output_data
        # 对齐serving output
        output_data_dict["score"] = output_data_dict["save_infer_model/scale_0"]
        del output_data_dict["save_infer_model/scale_0"]
        self.truth_val = output_data_dict
        # 真实概率值
        self.truth_prob = np.max(self.truth_val["score"])
        self.truth_prob = {"prob": np.array([self.truth_prob])}
        print(self.truth_prob)

    def predict_pipeline_rpc(self, batch_size=1):
        # 1.prepare feed_data
        with open("daisy.jpg", "rb") as f:
            image_data = f.read()
        image = cv2_to_base64(image_data)
        # TODO 原示例不支持batch
        feed_dict = {"image": image}

        # 2.init client
        fetch = ["label", "prob"]
        client = PipelineClient()
        client.connect(['127.0.0.1:9999'])

        # 3.predict for fetch_map
        ret = client.predict(feed_dict=feed_dict, fetch=fetch)
        print(ret)
        # 转换为dict
        result = {"prob": np.array(eval(ret.value[1]))}
        print(result)
        return result

    def predict_pipeline_http(self, batch_size=1):
        # 1.prepare feed_data
        with open("daisy.jpg", "rb") as f:
            image_data = f.read()
        image = cv2_to_base64(image_data)
        feed_dict = {"key": [], "value": []}
        # TODO 原示例不支持batch
        feed_dict["key"].append("image")
        feed_dict["value"].append(image)

        # 2.predict for fetch_map
        url = "http://127.0.0.1:18082/imagenet/prediction"
        r = requests.post(url=url, data=json.dumps(feed_dict))
        print(r.json())
        # 转换为dict of numpy array
        result = {"prob": np.array(eval(r.json()["value"][1]))}
        return result

    def test_gpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} resnet50_web_service.py",
            sleep=8,
        )

        # 2.resource check
        assert count_process_num_on_port(9999) == 1  # gRPC Server
        assert count_process_num_on_port(18082) == 1  # gRPC gateway 代理、转发
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is False

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU", filename="stderr.log")

        # 4.predict by rpc
        # batch_size=1
        result = self.predict_pipeline_rpc(batch_size=1)
        self.serving_util.check_result(result_data=result, truth_data=self.truth_prob, batch_size=1)
        # predict by http
        result = self.predict_pipeline_http(batch_size=1)  # batch_size=1
        self.serving_util.check_result(result_data=result, truth_data=self.truth_prob, batch_size=1)

        # 5.release
        stop_server(sleep_time=5)
        assert count_process_num_on_port(9999) == 0  # gRPC Server
        assert count_process_num_on_port(18082) == 0

