import os
import subprocess
import numpy as np
import copy
import cv2

from paddle_serving_client import Client, HttpClient
from paddle_serving_client.io import inference_model_to_serving
from paddle_serving_app.reader import SegPostprocess
from paddle_serving_app.reader import *
import paddle.inference as paddle_infer

from util import *


def serving_encryption():
    inference_model_to_serving(
        dirname="./uci_housing_model",
        params_filename=None,
        serving_server="encrypt_server",
        serving_client="encrypt_client",
        encryption=True)


class TestEncryption(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="encryption", example_path="Cpp/encryption", model_dir="encrypt_server",
                                   client_dir="encrypt_client")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util
        serving_encryption()

    def teardown_method(self):
        print_log(["stderr.log", "stdout.log",
                   "log/serving.ERROR", "PipelineServingLogs/pipeline.log"], iden="after predict")
        kill_process(9494)
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
        self.truth_val = output_data_dict
        print(self.truth_val, self.truth_val["fc_0.tmp_1"].shape)

    def predict_brpc(self, batch_size=1, encryption=False):
        data = np.array(
            [0.0137, -0.1136, 0.2553, -0.0692, 0.0582, -0.0727, -0.1583, -0.0584, 0.6283, 0.4919, 0.1856, 0.0795,
             -0.0332]).astype("float32")

        fetch = ["price"]
        endpoint_list = ['127.0.0.1:9494']

        client = Client()
        client.load_client_config(self.serving_util.client_config)
        if encryption:
            client.use_key("./key")
        client.connect(endpoint_list, encryption=encryption)

        if batch_size == 1:
            fetch_map = client.predict(feed={"x": data}, fetch=client.get_fetch_names(), batch=False)
        else:
            data = data[np.newaxis, :]
            data = np.repeat(data, repeats=batch_size, axis=0)
            fetch_map = client.predict(feed={"x": data}, fetch=client.get_fetch_names(), batch=True)
        print(fetch_map)
        return fetch_map

    def predict_http(self, mode="proto", compress=False, batch_size=1, encryption=False):
        data = np.array(
            [0.0137, -0.1136, 0.2553, -0.0692, 0.0582, -0.0727, -0.1583, -0.0584, 0.6283, 0.4919, 0.1856, 0.0795,
             -0.0332]).astype("float32")

        fetch = ["price"]
        endpoint_list = ['127.0.0.1:9494']

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

        if batch_size == 1:
            fetch_map = client.predict(feed={"x": data}, fetch=client.get_fetch_names(), batch=False)
        else:
            data = data[np.newaxis, :]
            data = np.repeat(data, repeats=batch_size, axis=0)
            fetch_map = client.predict(feed={"x": data}, fetch=client.get_fetch_names(), batch=True)
        print(fetch_map)
        # 转换为dict包numpy array
        result_dict = self.serving_util.parse_http_result(fetch_map)
        return result_dict

    def test_encrypt(self):
        serving_encryption()

    def test_encrypt_gpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model encrypt_server --use_encryption_model --gpu_ids 0,1 --port 9494",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9494) == 1
        # 加密部署时，client带着key请求后才起brpc-server

        # 3.keywords check

        # 4.predict by brpc
        # batch_size 1
        result_data = self.predict_brpc(batch_size=1, encryption=True)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        assert count_process_num_on_port(12000) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True
        check_keywords_in_server_log("Sync params from CPU to GPU", filename="stderr.log")
        check_keywords_in_server_log("Starting encryption server", filename="stdout.log")
        # batch_size 2
        result_data = self.predict_brpc(batch_size=2, encryption=True)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=2)
        # predict by http
        # batch_size 1
        result_data = self.predict_http(batch_size=1, encryption=True)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        # batch_size 2
        result_data = self.predict_http(batch_size=2, encryption=True)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=2)

        # 5.release
        kill_process(9494, 2)

    def test_encrypt_cpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model encrypt_server --use_encryption_model --port 9494",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9494) == 1
        # 加密部署时，client带着key请求后才起brpc-server

        # 3.keywords check

        # 4.predict
        # batch_size 1
        result_data = self.predict_brpc(batch_size=1, encryption=True)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        assert count_process_num_on_port(12000) == 1
        assert check_gpu_memory(0) is False
        check_keywords_in_server_log("Starting encryption server", filename="stdout.log")
        # batch_size 2
        result_data = self.predict_brpc(batch_size=2, encryption=True)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=2)
        # predict by http
        # batch_size 1
        result_data = self.predict_http(batch_size=1, encryption=True)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        # batch_size 2
        result_data = self.predict_http(batch_size=2, encryption=True)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=2)

        # 5.release
        kill_process(9494)

    def test_cpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model encrypt_server --thread 10 --port 9494",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9494) == 1
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
        kill_process(9494)

    def test_gpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model encrypt_server --thread 10 --port 9494 --gpu_ids 0",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9494) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is False

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU")

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
        kill_process(9494, 2)
