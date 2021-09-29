import os
import subprocess
import numpy as np
import sys

from paddle_serving_client import Client, HttpClient
import paddle.inference as paddle_infer

from chinese_ernie_reader import ChineseErnieReader
sys.path.append("../")
from util import *


class TestErnieXPU(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="ernie", example_path="xpu/ernie", model_dir="serving_server",
                                   client_dir="serving_client")
        self.serving_util = serving_util
        serving_util.check_model_data_exist()
        os.system("wget https://paddle-serving.bj.bcebos.com/bert_example/vocab.txt")
        self.get_truth_val_by_inference(self)

    def teardown_method(self):
        print("======================stderr.log after predict======================")
        os.system("cat stderr.log")
        print("======================stdout.log after predict======================")
        os.system("cat stdout.log")
        print("====================================================================")
        kill_process(7704)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        # TODO XPU下使用预测库拿到基准值
        pass

    def predict_brpc(self, batch_size=1):
        reader = ChineseErnieReader({"max_seq_len": 128})
        fetch = ["save_infer_model/scale_0"]
        endpoint_list = ['127.0.0.1:7704']
        client = Client()
        client.load_client_config(self.serving_util.client_config)
        client.connect(endpoint_list)

        feed_dict = reader.process("送晚了，饿得吃得很香")
        if batch_size == 1:
            for key in feed_dict.keys():
                feed_dict[key] = np.array(feed_dict[key]).reshape((128, 1))
            result = client.predict(feed=feed_dict, fetch=fetch, batch=False)
        else:
            # 多batch
            for key in feed_dict.keys():
                feed_dict[key] = np.array(feed_dict[key]).reshape((1, 128, 1))
                feed_dict[key] = np.repeat(feed_dict[key], repeats=batch_size, axis=0)
            result = client.predict(feed=feed_dict, fetch=fetch, batch=True)
        # print(result)
        return result

    def predict_http(self, mode="proto", compress=False, batch_size=1):
        reader = ChineseErnieReader({"max_seq_len": 128})
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
        client.connect(["127.0.0.1:7704"])

        feed_dict = reader.process("送晚了，饿得吃得很香")
        if batch_size == 1:
            for key in feed_dict.keys():
                feed_dict[key] = np.array(feed_dict[key]).reshape((128, 1))
            result = client.predict(feed=feed_dict, fetch=fetch, batch=False)
        else:
            # 多batch
            for key in feed_dict.keys():
                feed_dict[key] = np.array(feed_dict[key]).reshape((1, 128, 1))
                feed_dict[key] = np.repeat(feed_dict[key], repeats=batch_size, axis=0)
            result = client.predict(feed=feed_dict, fetch=fetch, batch=True)
        # 转换为dict包numpy array
        result_dict = self.serving_util.parse_http_result(result)
        return result_dict

    def test_xpu_lite(self, delta=1e-3):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model serving_server --port 7704 --use_lite --use_xpu --ir_optim",
            sleep=15,
        )

        # 2.resource check
        assert count_process_num_on_port(7704) == 1

        # 3.keywords check
        check_keywords_in_server_log("Running pass: __xpu__", filename="stderr.log")

        # 4.predict
        # by pybind-brpc_client
        result_data = self.predict_brpc(batch_size=1)
        print(result_data["save_infer_model/scale_0"].shape)
        result_data = self.predict_brpc(batch_size=2)
        print(result_data["save_infer_model/scale_0"].shape)

        # by HTTP-proto
        # TODO 必须开启压缩，否则超过默认buffer space报错(http client数据类型处理疑似有误，待排查)
        # result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        # print(result_data["save_infer_model/scale_0"].shape)
        # result_data = self.predict_http(mode="proto", compress=True, batch_size=2)
        # print(result_data["save_infer_model/scale_0"].shape)
        # result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        # print(result_data["save_infer_model/scale_0"].shape)
        # # by HTTP-json
        # result_data = self.predict_http(mode="json", compress=True, batch_size=1)
        # print(result_data["save_infer_model/scale_0"].shape)
        # # by HTTP-grpc
        # result_data = self.predict_http(mode="grpc", compress=True, batch_size=1)
        # print(result_data["save_infer_model/scale_0"].shape)
        # 5.release
        kill_process(7704)

    def cpu_lite(self):
        # todo Check failed: is_found Can't find a Cast kernel for Cast op: Tensor<host,int64_t,NCHW,0>:placeholder_3->Tensor<x86,float,NCHW,0>:cast
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model serving_server --port 7704 --use_lite --ir_optim",
            sleep=15,
        )

        # 2.resource check
        assert count_process_num_on_port(7704) == 1

        # 3.keywords check

        # 4.predict
        # by pybind-brpc_client
        result_data = self.predict_brpc(batch_size=1)
        print(result_data["save_infer_model/scale_0"].shape)
        result_data = self.predict_brpc(batch_size=2)
        print(result_data["save_infer_model/scale_0"].shape)

        # by HTTP-proto
        # TODO 必须开启压缩，否则超过默认buffer space报错(http client数据类型处理疑似有误，待排查)
        # result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        # print(result_data["save_infer_model/scale_0"].shape)
        # result_data = self.predict_http(mode="proto", compress=True, batch_size=2)
        # print(result_data["save_infer_model/scale_0"].shape)
        # result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        # print(result_data["save_infer_model/scale_0"].shape)
        # # by HTTP-json
        # result_data = self.predict_http(mode="json", compress=True, batch_size=1)
        # print(result_data["save_infer_model/scale_0"].shape)
        # # by HTTP-grpc
        # result_data = self.predict_http(mode="grpc", compress=True, batch_size=1)
        # print(result_data["save_infer_model/scale_0"].shape)
        # 5.release
        kill_process(7704)

    def test_cpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model serving_server --port 7704",
            sleep=7,
        )

        # 2.resource check
        assert count_process_num_on_port(7704) == 1

        # 3.keywords check

        # 4.predict
        # by pybind-brpc_client
        result_data = self.predict_brpc(batch_size=1)
        print(result_data["save_infer_model/scale_0"].shape)
        result_data = self.predict_brpc(batch_size=2)
        print(result_data["save_infer_model/scale_0"].shape)

        # by HTTP-proto
        # TODO 必须开启压缩，否则超过默认buffer space报错(http client数据类型处理疑似有误，待排查)
        # result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        # print(result_data["save_infer_model/scale_0"].shape)
        # result_data = self.predict_http(mode="proto", compress=True, batch_size=2)
        # print(result_data["save_infer_model/scale_0"].shape)
        # result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        # print(result_data["save_infer_model/scale_0"].shape)
        # # by HTTP-json
        # result_data = self.predict_http(mode="json", compress=True, batch_size=1)
        # print(result_data["save_infer_model/scale_0"].shape)
        # # by HTTP-grpc
        # result_data = self.predict_http(mode="grpc", compress=True, batch_size=1)
        # print(result_data["save_infer_model/scale_0"].shape)
        # 5.release
        kill_process(7704)
