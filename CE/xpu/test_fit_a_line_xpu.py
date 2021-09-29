import os
import subprocess
import numpy as np
import copy
import cv2
import sys

from paddle_serving_client import Client, HttpClient
from paddle_serving_app.reader import SegPostprocess
from paddle_serving_app.reader import *
import paddle.inference as paddle_infer

sys.path.append("../")
from util import *


class TestFitALineXPU(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="fit_a_line", example_path="xpu/fit_a_line_xpu", model_dir="uci_housing_model",
                                   client_dir="uci_housing_client")
        serving_util.check_model_data_exist()
        self.serving_util = serving_util
        self.get_truth_val_by_inference(self)

    def teardown_method(self):
        print("======================stderr.log after predict======================")
        os.system("cat stderr.log")
        print("======================stdout.log after predict======================")
        os.system("cat stdout.log")
        print("====================================================================")
        kill_process(9393)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        # TODO XPU下使用预测库拿到基准值
        pass

    def predict_brpc(self, batch_size=1):
        # 1.prepare feed_data
        data = np.array(
            [0.0137, -0.1136, 0.2553, -0.0692, 0.0582, -0.0727, -0.1583, -0.0584, 0.6283, 0.4919, 0.1856, 0.0795,
             -0.0332]).astype("float32")
        fetch = ["price"]
        endpoint_list = ['127.0.0.1:9393']

        # 2.init client
        client = Client()
        client.load_client_config(self.serving_util.client_config)
        client.connect(endpoint_list)

        # 3.predict for fetch_map
        if batch_size == 1:
            fetch_map = client.predict(feed={"x": data}, fetch=fetch, batch=False)
        else:
            data = data[np.newaxis, :]
            data = np.repeat(data, repeats=batch_size, axis=0)
            fetch_map = client.predict(feed={"x": data}, fetch=fetch, batch=True)
        print(fetch_map)
        return fetch_map

    def predict_http(self, mode="proto", compress=False, batch_size=1, encryption=False):
        # 1.prepare feed_data
        data = np.array(
            [0.0137, -0.1136, 0.2553, -0.0692, 0.0582, -0.0727, -0.1583, -0.0584, 0.6283, 0.4919, 0.1856, 0.0795,
             -0.0332]).astype("float32")

        fetch = ["price"]
        endpoint_list = ['127.0.0.1:9393']

        # 2.init client
        client = HttpClient()
        client.load_client_config(self.serving_util.client_config)
        if encryption:
            # http client加密预测
            client.use_key("./key")
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
        client.connect(endpoint_list, encryption=encryption)

        # 3.predict for fetch_map
        if batch_size == 1:
            fetch_map = client.predict(feed={"x": data}, fetch=fetch, batch=False)
        else:
            data = data[np.newaxis, :]
            data = np.repeat(data, repeats=batch_size, axis=0)
            fetch_map = client.predict(feed={"x": data}, fetch=fetch, batch=True)
        print(fetch_map)
        # 转换为dict包numpy array
        result_dict = self.serving_util.parse_http_result(fetch_map)
        return result_dict

    def test_xpu_lite(self, delta=1e-3):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model uci_housing_model --port 9393 --use_lite --use_xpu --ir_optim",
            sleep=15,
        )

        # 2.resource check
        assert count_process_num_on_port(9393) == 1

        # 3.keywords check
        check_keywords_in_server_log("Running pass: __xpu__", filename="stderr.log")

        # 4.predict
        # by pybind-brpc_client
        result_data = self.predict_brpc(batch_size=1)
        print(result_data["price"].shape)
        result_data = self.predict_brpc(batch_size=2)
        print(result_data["price"].shape)

        # by HTTP-proto
        # result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        # print(result_data["price"].shape)
        # result_data = self.predict_http(mode="proto", compress=True, batch_size=2)
        # print(result_data["price"].shape)
        # # by HTTP-json
        # result_data = self.predict_http(mode="json", compress=True, batch_size=1)
        # print(result_data["price"].shape)
        # # by HTTP-grpc
        # result_data = self.predict_http(mode="grpc", compress=True, batch_size=1)
        # print(result_data["price"].shape)
        # 5.release
        kill_process(9393)

    def test_cpu_lite(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model uci_housing_model --port 9393 --use_lite --ir_optim",
            sleep=15,
        )

        # 2.resource check
        assert count_process_num_on_port(9393) == 1

        # 3.keywords check

        # 4.predict
        # by pybind-brpc_client
        result_data = self.predict_brpc(batch_size=1)
        print(result_data["price"].shape)
        result_data = self.predict_brpc(batch_size=2)
        print(result_data["price"].shape)

        # by HTTP-proto
        # result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        # print(result_data["price"].shape)
        # result_data = self.predict_http(mode="proto", compress=True, batch_size=2)
        # print(result_data["price"].shape)
        # # by HTTP-json
        # result_data = self.predict_http(mode="json", compress=True, batch_size=1)
        # print(result_data["price"].shape)
        # # by HTTP-grpc
        # result_data = self.predict_http(mode="grpc", compress=True, batch_size=1)
        # print(result_data["price"].shape)
        # 5.release
        kill_process(9393)

    def test_cpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model uci_housing_model --port 9393",
            sleep=7,
        )

        # 2.resource check
        assert count_process_num_on_port(9393) == 1

        # 3.keywords check

        # 4.predict
        # by pybind-brpc_client
        result_data = self.predict_brpc(batch_size=1)
        print(result_data["price"].shape)
        result_data = self.predict_brpc(batch_size=2)
        print(result_data["price"].shape)

        # by HTTP-proto
        # result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        # print(result_data["price"].shape)
        # result_data = self.predict_http(mode="proto", compress=True, batch_size=2)
        # print(result_data["price"].shape)
        # # by HTTP-json
        # result_data = self.predict_http(mode="json", compress=True, batch_size=1)
        # print(result_data["price"].shape)
        # # by HTTP-grpc
        # result_data = self.predict_http(mode="grpc", compress=True, batch_size=1)
        # print(result_data["price"].shape)
        # 5.release
        kill_process(9393)
