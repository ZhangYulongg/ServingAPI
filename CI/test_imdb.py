import os
import subprocess
import numpy as np
import copy
import cv2

from paddle_serving_client import Client, HttpClient
from paddle_serving_app.reader import Sequential, URL2Image, Resize, File2Image
from paddle_serving_app.reader import CenterCrop, RGB2BGR, Transpose, Div, Normalize
from paddle_serving_app.reader.imdb_reader import IMDBDataset
import paddle.inference as paddle_infer

from util import *


class TestIMDB(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="imdb", example_path="C++/imdb", model_dir="imdb_cnn_model",
                                   client_dir="imdb_cnn_client_conf")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util

    def teardown_method(self):
        print_log(["stderr.log", "stdout.log",
                   "log/serving.ERROR", "PipelineServingLogs/pipeline.log"], iden="after predict")
        kill_process(9292)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        imdb_dataset = IMDBDataset()
        imdb_dataset.load_resource("imdb.vocab")
        with open("test_data/part-0") as f:
            line = f.readline()
        word_ids, label = imdb_dataset.get_words_and_label(line)
        word_len = len(word_ids)
        input_dict = {
            "words": np.array(word_ids).reshape(word_len, 1),
            "words.lod": [0, word_len]
        }

        pd_config = paddle_infer.Config("imdb_cnn_model")
        pd_config.disable_gpu()
        pd_config.switch_ir_optim(False)

        predictor = paddle_infer.create_predictor(pd_config)

        input_names = predictor.get_input_names()
        for i, input_name in enumerate(input_names):
            input_handle = predictor.get_input_handle(input_name)
            # 设置变长tensor
            input_handle.set_lod([input_dict[f"{input_name}.lod"]])
            input_handle.copy_from_cpu(input_dict[input_name])

        predictor.run()

        output_data_dict = {}
        output_names = predictor.get_output_names()
        for _, output_data_name in enumerate(output_names):
            output_handle = predictor.get_output_handle(output_data_name)
            output_data = output_handle.copy_to_cpu()
            output_data_dict[output_data_name] = output_data
        # 对齐Serving output
        output_data_dict["prediction"] = output_data_dict["fc_1.tmp_2"]
        del output_data_dict["fc_1.tmp_2"]
        self.truth_val = output_data_dict
        print(self.truth_val, self.truth_val["prediction"].shape)

    def predict_brpc(self, batch_size=1):
        imdb_dataset = IMDBDataset()
        imdb_dataset.load_resource("imdb.vocab")
        with open("test_data/part-0") as f:
            line = f.readline()
        word_ids, label = imdb_dataset.get_words_and_label(line)
        word_len = len(word_ids)
        feed_dict = {
            "words": np.array(word_ids).reshape(word_len, 1),
            "words.lod": [0, word_len]
        }

        fetch = ["prediction"]
        endpoint_list = ['127.0.0.1:9292']

        client = Client()
        client.load_client_config(self.serving_util.client_config)
        client.connect(endpoint_list)

        fetch_map = client.predict(feed=feed_dict, fetch=fetch, batch=True)
        print(fetch_map)
        return fetch_map

    def predict_http(self, mode="proto", compress=False, batch_size=1):
        imdb_dataset = IMDBDataset()
        imdb_dataset.load_resource("imdb.vocab")
        with open("test_data/part-0") as f:
            line = f.readline()
        word_ids, label = imdb_dataset.get_words_and_label(line)
        word_len = len(word_ids)
        feed_dict = {
            "words": np.array(word_ids).reshape(word_len, 1),
            "words.lod": [0, word_len]
        }

        fetch = ["prediction"]

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
        client.connect(["127.0.0.1:9292"])

        fetch_map = client.predict(feed=feed_dict, fetch=fetch, batch=True)
        print(fetch_map)
        result_dict = self.serving_util.parse_http_result(fetch_map)
        print(result_dict)
        return result_dict

    def test_cpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model imdb_cnn_model --port 9292",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9292) == 1
        assert check_gpu_memory(0) is False

        # 3.keywords check

        # 4.predict by brpc
        # batch_size 1
        result_data = self.predict_brpc(batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        # predict by http
        # batch_size 1
        result_data = self.predict_http(mode="proto", batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        result_data = self.predict_http(mode="json", batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        result_data = self.predict_http(mode="grpc", batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        # compress
        result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        result_data = self.predict_http(mode="json", compress=True, batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)

        # 5.release
        kill_process(9292)

    def test_gpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model imdb_cnn_model --port 9292 --gpu_ids 0",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9292) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is False

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU", filename="stderr.log")

        # 4.predict by brpc
        # batch_size 1
        result_data = self.predict_brpc(batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        # predict by http
        # batch_size 1
        result_data = self.predict_http(mode="proto", batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        result_data = self.predict_http(mode="json", batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        result_data = self.predict_http(mode="grpc", batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        # compress
        result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        result_data = self.predict_http(mode="json", compress=True, batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)

        # 5.release
        kill_process(9292, 2)

    def test_gpu_async(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model imdb_cnn_model --thread 10 --runtime_thread_num 4 --port 9292 --gpu_ids 0",
            sleep=6,
        )

        # 2.resource check
        assert count_process_num_on_port(9292) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is False

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU", filename="stderr.log")
        check_keywords_in_server_log("Enable batch schedule framework, thread_num:4, batch_size:32, enable_overrun:0, allow_split_request:1", filename="stderr.log")

        # 4.predict by brpc
        # batch_size 1
        result_data = self.predict_brpc(batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        # predict by http
        # batch_size 1
        result_data = self.predict_http(mode="proto", batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        result_data = self.predict_http(mode="json", batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        result_data = self.predict_http(mode="grpc", batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        # compress
        result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        result_data = self.predict_http(mode="json", compress=True, batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)

        # 5.release
        kill_process(9292)
