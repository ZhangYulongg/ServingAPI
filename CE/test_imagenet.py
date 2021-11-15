import os
import subprocess
import numpy as np
import copy
import cv2

from paddle_serving_client import Client, HttpClient
from paddle_serving_app.reader import Sequential, URL2Image, Resize, File2Image
from paddle_serving_app.reader import CenterCrop, RGB2BGR, Transpose, Div, Normalize
import paddle.inference as paddle_infer

from util import *


class TestImagenet(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="imagenet", example_path="C++/PaddleClas/imagenet", model_dir="ResNet50_vd_model",
                                   client_dir="ResNet50_vd_client_config")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util

    def teardown_method(self):
        print_log(["stderr.log", "stdout.log",
                   "log/serving.ERROR", "PipelineServingLogs/pipeline.log"], iden="after predict")
        kill_process(9696)
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
        print(self.truth_val, self.truth_val["score"].shape)

    def predict_brpc(self, batch_size=1):
        seq = Sequential([
            File2Image(), Resize(256), CenterCrop(224), RGB2BGR(), Transpose((2, 0, 1)),
            Div(255), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True)
        ])
        filename = "daisy.jpg"
        im = seq(filename)

        fetch = ["score"]
        endpoint_list = ['127.0.0.1:9696']

        client = Client()
        client.load_client_config(self.serving_util.client_config)
        client.connect(endpoint_list)

        if batch_size == 1:
            fetch_map = client.predict(feed={"image": im}, fetch=fetch, batch=False)
        else:
            im = im[np.newaxis, :]
            im = np.repeat(im, repeats=batch_size, axis=0)
            fetch_map = client.predict(feed={"image": im}, fetch=fetch, batch=True)
        print(fetch_map)
        return fetch_map

    def predict_http(self, mode="proto", compress=False, batch_size=1):
        seq = Sequential([
            File2Image(), Resize(256), CenterCrop(224), RGB2BGR(), Transpose((2, 0, 1)),
            Div(255), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True)
        ])
        filename = "daisy.jpg"
        data = seq(filename)

        fetch = ["score"]

        client = HttpClient()
        client.load_client_config(self.serving_util.client_config)
        if mode == "proto":
            client.set_http_proto(True)
        elif mode == "json":
            client.set_http_proto(False)
        elif mode == "grpc":
            client.set_use_grpc_client(True)
        else:
            exit(-1)
        if compress:
            client.set_response_compress(True)
            client.set_request_compress(True)
        client.connect(["127.0.0.1:9696"])

        if batch_size == 1:
            fetch_map = client.predict(feed={"image": data}, fetch=fetch, batch=False)
        else:
            data = data[np.newaxis, :]
            data = np.repeat(data, repeats=batch_size, axis=0)
            fetch_map = client.predict(feed={"image": data}, fetch=fetch, batch=True)
        # 转换为dict包numpy array
        result_dict = self.serving_util.parse_http_result(fetch_map)
        print(result_dict)
        return result_dict

    def test_cpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model ResNet50_vd_model --port 9696",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9696) == 1
        assert check_gpu_memory(0) is False

        # 3.keywords check

        # 4.predict by brpc
        # batch_size 1
        result_data = self.predict_brpc(batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        # batch_size 2
        result_data = self.predict_brpc(batch_size=2)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=2)
        # predict by http
        # batch_size 1
        result_data = self.predict_http(batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        # batch_size 2
        result_data = self.predict_http(batch_size=2)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=2)

        # 5.release
        kill_process(9696)

    def test_gpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model ResNet50_vd_model --port 9696 --gpu_ids 0",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9696) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is False

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU", filename="stderr.log")

        # 4.predict by brpc
        # batch_size 1
        result_data = self.predict_brpc(batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        # batch_size 2
        result_data = self.predict_brpc(batch_size=2)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=2)
        # predict by http
        # batch_size 1
        result_data = self.predict_http(batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        # batch_size 2
        result_data = self.predict_http(batch_size=2)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=2)

        # 5.release
        kill_process(9696, 2)
