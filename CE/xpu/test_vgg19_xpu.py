import os
import subprocess
import numpy as np
import copy
import cv2
import time
import sys

from paddle_serving_client import Client, HttpClient
from paddle_serving_client.utils import MultiThreadRunner
from paddle_serving_client.utils import benchmark_args, show_latency
from paddle_serving_app.reader import Sequential, URL2Image, Resize, File2Image
from paddle_serving_app.reader import CenterCrop, RGB2BGR, Transpose, Div, Normalize
import paddle.inference as paddle_infer

sys.path.append("../")
from util import *


class TestVGG19XPU(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="vgg19", example_path="C++/xpu/vgg19", model_dir="serving_server",
                                   client_dir="serving_client")
        serving_util.check_model_data_exist()
        self.serving_util = serving_util
        self.get_truth_val_by_inference(self)

    def teardown_method(self):
        print_log(["stderr.log", "stdout.log",
                   "log/serving.ERROR", "PipelineServingLogs/pipeline.log"], iden="after predict")
        kill_process(9393)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        # TODO XPU下使用预测库拿到基准值
        pass

    def predict_brpc(self, batch_size=1):
        # 1.prepare feed_data
        seq = Sequential([
            File2Image(), Resize(256), CenterCrop(224), RGB2BGR(), Transpose((2, 0, 1)),
            Div(255), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True)
        ])
        image_file = "daisy.jpg"
        img = seq(image_file)

        # 2.init client
        fetch = ["save_infer_model/scale_0"]
        endpoint_list = ["127.0.0.1:9393"]
        client = Client()
        client.load_client_config(self.serving_util.client_config)
        client.connect(endpoint_list)

        # 3.predict for fetch_map
        if batch_size == 1:
            fetch_map = client.predict(feed={"image": img}, fetch=fetch, batch=False)
        else:
            img = img[np.newaxis, :]
            img = np.repeat(img, repeats=batch_size, axis=0)
            fetch_map = client.predict(feed={"image": img}, fetch=fetch, batch=True)
        print(fetch_map)
        return fetch_map

    def predict_http(self, mode="proto", compress=False, batch_size=1):
        # 1.prepare feed_data
        seq = Sequential([
            File2Image(), Resize(256), CenterCrop(224), RGB2BGR(), Transpose((2, 0, 1)),
            Div(255), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True)
        ])
        image_file = "daisy.jpg"
        img = seq(image_file)

        # 2.init client
        fetch = ["save_infer_model/scale_0"]
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
        client.connect(["127.0.0.1:9393"])

        # 3.predict for fetch_map
        if batch_size == 1:
            fetch_map = client.predict(feed={"image": img}, fetch=fetch, batch=False)
        else:
            img = img[np.newaxis, :]
            img = np.repeat(img, repeats=batch_size, axis=0)
            fetch_map = client.predict(feed={"image": img}, fetch=fetch, batch=True)
        # print(fetch_map)
        # 转换为dict包numpy array
        result_dict = self.serving_util.parse_http_result(fetch_map)
        # print(result_dict)
        return result_dict

    def test_xpu_lite(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model serving_server --port 9393 --use_lite --use_xpu --ir_optim",
            sleep=15,
        )

        # 2.resource check
        assert count_process_num_on_port(9393) == 1

        # 3.keywords check
        check_keywords_in_server_log("Running pass: __xpu__", filename="stderr.log")

        # 4.predict by brpc
        # batch_size 1
        result_data = self.predict_brpc(batch_size=1)
        print("shape:", result_data["save_infer_model/scale_0"].shape)
        # batch_size 2
        result_data = self.predict_brpc(batch_size=2)
        print("shape:", result_data["save_infer_model/scale_0"].shape)
        # predict by http
        # TODO 必须开启压缩，否则超过默认buffer space报错
        # batch_size 1
        # result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        # print("shape:", result_data["save_infer_model/scale_0"].shape)
        # os.system("sleep 2")
        # compress
        # result_data = self.predict_http(mode="json", compress=True, batch_size=1)
        # print("shape:", result_data["save_infer_model/scale_0"].shape)

        # result_data = self.predict_http(mode="grpc", compress=True, batch_size=1)
        # print("shape:", result_data["save_infer_model/scale_0"].shape)
        # 5.release
        kill_process(9393, 2)

    def test_cpu_lite(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model serving_server --port 9393 --use_lite --ir_optim",
            sleep=13,
        )

        # 2.resource check
        assert count_process_num_on_port(9393) == 1

        # 3.keywords check

        # 4.predict by brpc
        # batch_size 1
        result_data = self.predict_brpc(batch_size=1)
        print("shape:", result_data["save_infer_model/scale_0"].shape)
        # batch_size 2
        result_data = self.predict_brpc(batch_size=2)
        print("shape:", result_data["save_infer_model/scale_0"].shape)
        # predict by http
        # TODO 必须开启压缩，否则超过默认buffer space报错
        # batch_size 1
        # result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        # print("shape:", result_data["save_infer_model/scale_0"].shape)
        # result_data = self.predict_http(mode="json", compress=True, batch_size=1)
        # print("shape:", result_data["save_infer_model/scale_0"].shape)
        # os.system("sleep 2")
        # result_data = self.predict_http(mode="grpc", compress=True, batch_size=1)
        # print("shape:", result_data["save_infer_model/scale_0"].shape)
        # 5.release
        kill_process(9393, 2)

    def test_cpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model serving_server --port 9393",
            sleep=10,
        )

        # 2.resource check
        assert count_process_num_on_port(9393) == 1

        # 3.keywords check

        # 4.predict by brpc
        # batch_size 1
        result_data = self.predict_brpc(batch_size=1)
        print("shape:", result_data["save_infer_model/scale_0"].shape)
        # batch_size 2
        result_data = self.predict_brpc(batch_size=2)
        print("shape:", result_data["save_infer_model/scale_0"].shape)
        # predict by http
        # TODO 必须开启压缩，否则超过默认buffer space报错
        # batch_size 1
        result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        print("shape:", result_data["save_infer_model/scale_0"].shape)
        result_data = self.predict_http(mode="json", compress=True, batch_size=1)
        print("shape:", result_data["save_infer_model/scale_0"].shape)

        result_data = self.predict_http(mode="grpc", compress=True, batch_size=1)
        print("shape:", result_data["save_infer_model/scale_0"].shape)
        # 5.release
        kill_process(9393)

