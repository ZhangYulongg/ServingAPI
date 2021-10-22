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

from util import *


class TestImagenetPipeline(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="fit_a_line", example_path="pipeline/simple_web_service", model_dir="uci_housing_model",
                                   client_dir="uci_housing_client")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util

    def teardown_method(self):
        print_log(["stderr.log", "stdout.log",
                   "log/serving.ERROR", "PipelineServingLogs/pipeline.log"], iden="after predict")
        kill_process(9998)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        data = np.array(
            [0.0137, -0.1136, 0.2553, -0.0692, 0.0582, -0.0727, -0.1583, -0.0584, 0.6283, 0.4919, 0.1856, 0.0795,
             -0.0332]).astype("float32")[np.newaxis, :]
        input_dict = {"x": data}

        pd_config = paddle_infer.Config("uci_housing_model/")
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
        # 对齐Serving output
        output_data_dict["prob"] = output_data_dict["fc_0.tmp_1"]
        del output_data_dict["fc_0.tmp_1"]
        self.truth_val = output_data_dict
        print(self.truth_val, self.truth_val["prob"].shape)

    def predict_pipeline_rpc(self, batch_size=1):
        # 1.prepare feed_data
        feed_dict = {'x': '0.0137, -0.1136, 0.2553, -0.0692, 0.0582, -0.0727, -0.1583, -0.0584, 0.6283, 0.4919, 0.1856, 0.0795, -0.0332'}
        # TODO 原示例不支持batch

        # 2.init client
        # fetch = ["label", "prob"]
        client = PipelineClient()
        client.connect(['127.0.0.1:9998'])

        # 3.predict for fetch_map
        ret = client.predict(feed_dict=feed_dict)
        print(ret)
        # 转换为dict
        result = {"prob": np.array(eval(ret.value[0]))}
        print(result)
        return result

    def predict_pipeline_http(self, batch_size=1):
        # 1.prepare feed_data
        data = '0.0137, -0.1136, 0.2553, -0.0692, 0.0582, -0.0727, -0.1583, -0.0584, 0.6283, 0.4919, 0.1856, 0.0795, ' \
               '-0.0332'
        feed_dict = {"key": [], "value": []}
        # TODO 原示例不支持batch
        feed_dict["key"].append("x")
        feed_dict["value"].append(data)

        # 2.predict for fetch_map
        url = "http://127.0.0.1:18082/uci/prediction"
        r = requests.post(url=url, data=json.dumps(feed_dict))
        print(r.json())
        # 转换为dict of numpy array
        result = {"prob": np.array(eval(r.json()["value"][0]))}
        return result

    def test_cpu_mkldnn(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} web_service.py",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9998) == 1  # gRPC Server
        assert count_process_num_on_port(18082) == 1  # gRPC gateway 代理、转发
        assert check_gpu_memory(0) is False

        # 3.keywords check
        check_keywords_in_server_log("MKLDNN is enabled", filename="stderr.log")

        # 4.predict by rpc
        # batch_size=1
        result = self.predict_pipeline_rpc(batch_size=1)
        self.serving_util.check_result(result_data=result, truth_data=self.truth_val, batch_size=1)
        # # predict by http
        result = self.predict_pipeline_http(batch_size=1)  # batch_size=1
        self.serving_util.check_result(result_data=result, truth_data=self.truth_val, batch_size=1)

        # 5.release
        kill_process(9998)
        kill_process(18082)


