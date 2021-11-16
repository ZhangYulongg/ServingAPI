import os
import subprocess
import numpy as np
import sys
import time

from paddle_serving_client import Client, HttpClient
import paddle.inference as paddle_infer

from chinese_bert_reader import ChineseBertReader
sys.path.append("../")
from util import *


class TestBertXPU(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="bert", example_path="C++/xpu/bert", model_dir="serving_server",
                                   client_dir="serving_client")
        self.serving_util = serving_util
        serving_util.check_model_data_exist()
        os.system("wget https://paddle-serving.bj.bcebos.com/bert_example/vocab.txt")
        self.get_truth_val_by_inference(self)

    def teardown_method(self):
        print_log(["stderr.log", "stdout.log",
                   "log/serving.ERROR", "PipelineServingLogs/pipeline.log"], iden="after predict")
        kill_process(7703)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        # TODO XPU下使用预测库拿到基准值
        pass

    def predict_brpc(self, batch_size=1):
        reader = ChineseBertReader({"max_seq_len": 128})
        fetch = ["save_infer_model/scale_0.tmp_1"]
        endpoint_list = ['127.0.0.1:7703']
        client = Client()
        client.load_client_config(self.serving_util.client_config)
        client.connect(endpoint_list)

        feed_dict = reader.process("送晚了，饿得吃得很香")
        if batch_size == 1:
            for key in feed_dict.keys():
                feed_dict[key] = np.array(feed_dict[key]).reshape(-1)
            result = client.predict(feed=feed_dict, fetch=fetch, batch=False)
        else:
            # 多batch
            for key in feed_dict.keys():
                feed_dict[key] = np.array(feed_dict[key]).reshape((1, 128))
                feed_dict[key] = np.repeat(feed_dict[key], repeats=batch_size, axis=0)
            result = client.predict(feed=feed_dict, fetch=fetch, batch=True)
        # print(result)
        return result

    def predict_http(self, mode="proto", compress=False, batch_size=1):
        reader = ChineseBertReader({"max_seq_len": 128})
        fetch = ["save_infer_model/scale_0.tmp_1"]
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
        client.connect(["127.0.0.1:7703"])

        feed_dict = reader.process("送晚了，饿得吃得很香")
        if batch_size == 1:
            for key in feed_dict.keys():
                feed_dict[key] = np.array(feed_dict[key]).reshape(-1)
            result = client.predict(feed=feed_dict, fetch=fetch, batch=False)
        else:
            # 多batch
            for key in feed_dict.keys():
                feed_dict[key] = np.array(feed_dict[key]).reshape((1, 128))
                feed_dict[key] = np.repeat(feed_dict[key], repeats=batch_size, axis=0)
            result = client.predict(feed=feed_dict, fetch=fetch, batch=True)
        # 转换为dict包numpy array
        result_dict = self.serving_util.parse_http_result(result)
        return result_dict

    def test_xpu_lite(self, delta=1e-3):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model serving_server --port 7703 --use_lite --use_xpu --ir_optim",
            sleep=15,
        )

        # 2.resource check
        assert count_process_num_on_port(7703) == 1

        # 3.keywords check
        check_keywords_in_server_log("Running pass: __xpu__", filename="stderr.log")

        # 4.predict
        # by pybind-brpc_client
        result_data = self.predict_brpc(batch_size=1)
        print(result_data["save_infer_model/scale_0.tmp_1"].shape)
        result_data = self.predict_brpc(batch_size=2)
        print(result_data["save_infer_model/scale_0.tmp_1"].shape)
        time.sleep(2)

        # by HTTP-proto
        # TODO 必须开启压缩，否则超过默认buffer space报错
        result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        print(result_data["save_infer_model/scale_0.tmp_1"].shape)
        time.sleep(2)
        result_data = self.predict_http(mode="proto", compress=True, batch_size=2)
        print(result_data["save_infer_model/scale_0.tmp_1"].shape)
        time.sleep(2)
        # by HTTP-json
        result_data = self.predict_http(mode="json", compress=True, batch_size=1)
        print(result_data["save_infer_model/scale_0.tmp_1"].shape)
        time.sleep(2)
        # by HTTP-grpc
        result_data = self.predict_http(mode="grpc", compress=True, batch_size=1)
        print(result_data["save_infer_model/scale_0.tmp_1"].shape)
        # 5.release
        kill_process(7703)

    def cpu_lite(self):
        # TODO Check failed: !instruct.kernels().empty() No kernels found for matmul_v2
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model serving_server --port 7703 --use_lite --ir_optim",
            sleep=15,
        )

        # 2.resource check
        assert count_process_num_on_port(7703) == 1

        # 3.keywords check

        # 4.predict
        # by pybind-brpc_client
        result_data = self.predict_brpc(batch_size=1)
        print(result_data["save_infer_model/scale_0.tmp_1"].shape)
        result_data = self.predict_brpc(batch_size=2)
        print(result_data["save_infer_model/scale_0.tmp_1"].shape)


        # by HTTP-proto
        # TODO 必须开启压缩，否则超过默认buffer space报错
        result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        print(result_data["save_infer_model/scale_0.tmp_1"].shape)
        result_data = self.predict_http(mode="proto", compress=True, batch_size=2)
        print(result_data["save_infer_model/scale_0.tmp_1"].shape)
        # by HTTP-json
        result_data = self.predict_http(mode="json", compress=True, batch_size=1)
        print(result_data["save_infer_model/scale_0.tmp_1"].shape)
        # by HTTP-grpc
        result_data = self.predict_http(mode="grpc", compress=True, batch_size=1)
        print(result_data["save_infer_model/scale_0.tmp_1"].shape)
        # 5.release
        kill_process(7703)

    def test_cpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model serving_server --port 7703",
            sleep=7,
        )

        # 2.resource check
        assert count_process_num_on_port(7703) == 1

        # 3.keywords check

        # 4.predict
        # by pybind-brpc_client
        result_data = self.predict_brpc(batch_size=1)
        print(result_data["save_infer_model/scale_0.tmp_1"].shape)
        result_data = self.predict_brpc(batch_size=2)
        print(result_data["save_infer_model/scale_0.tmp_1"].shape)

        # by HTTP-proto
        # TODO 必须开启压缩，否则超过默认buffer space报错
        result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        print(result_data["save_infer_model/scale_0.tmp_1"].shape)
        result_data = self.predict_http(mode="proto", compress=True, batch_size=2)
        print(result_data["save_infer_model/scale_0.tmp_1"].shape)
        # by HTTP-json
        result_data = self.predict_http(mode="json", compress=True, batch_size=1)
        print(result_data["save_infer_model/scale_0.tmp_1"].shape)
        # by HTTP-grpc
        result_data = self.predict_http(mode="grpc", compress=True, batch_size=1)
        print(result_data["save_infer_model/scale_0.tmp_1"].shape)
        # 5.release
        kill_process(7703)
