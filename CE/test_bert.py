import os
import subprocess
import numpy as np

from paddle_serving_client import Client, HttpClient
from paddle_serving_app.reader import ChineseBertReader
import paddle.inference as paddle_infer

from util import *


class TestBert(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="bert", example_path="bert", model_dir="bert_seq128_model",
                                   client_dir="bert_seq128_client")
        self.serving_util = serving_util
        os.chdir(self.serving_util.example_path)
        print("======================cur path======================")
        os.system("pwd")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)

    def teardown_method(self):
        print("======================stderr.log after predict======================")
        os.system("cat stderr.log")
        print("======================stdout.log after predict======================")
        os.system("cat stdout.log")
        print("====================================================================")
        kill_process(9292)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        reader = ChineseBertReader({"max_seq_len": 128, "vocab_file": "vocab.txt"})
        input_dict = reader.process("送晚了，饿得吃得很香")
        for key in input_dict.keys():
            input_dict[key] = np.array(input_dict[key]).reshape((1, 128, 1))
        input_dict["input_mask"] = input_dict["input_mask"].astype("float32")
        input_dict["position_ids"] = input_dict["position_ids"].astype("int64")
        input_dict["input_ids"] = input_dict["input_ids"].astype("int64")
        input_dict["segment_ids"] = input_dict["segment_ids"].astype("int64")

        pd_config = paddle_infer.Config("bert_seq128_model")
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
        print(output_data_dict)
        output_data_dict["pooled_output"] = output_data_dict[
            "@HUB_bert_chinese_L-12_H-768_A-12@@HUB_bert_chinese_L-12_H-768_A-12@fc_72.tmp_2"]
        output_data_dict["sequence_output"] = output_data_dict[
            "@HUB_bert_chinese_L-12_H-768_A-12@@HUB_bert_chinese_L-12_H-768_A-12@layer_norm_24.tmp_2"]
        del output_data_dict["@HUB_bert_chinese_L-12_H-768_A-12@@HUB_bert_chinese_L-12_H-768_A-12@fc_72.tmp_2"], \
        output_data_dict["@HUB_bert_chinese_L-12_H-768_A-12@@HUB_bert_chinese_L-12_H-768_A-12@layer_norm_24.tmp_2"]
        self.truth_val = output_data_dict
        print(self.truth_val, self.truth_val["pooled_output"].shape, self.truth_val["sequence_output"].shape)

    def check_result(self, result_data, truth_data, batch_size=1, delta=1e-3):
        # flatten
        predict_result = {}
        truth_result = {}
        for key, value in result_data.items():
            predict_result[key] = value.flatten()
        for key, value in truth_data.items():
            truth_result[key] = np.repeat(value, repeats=batch_size, axis=0).flatten()

        # compare
        for i, data in enumerate(predict_result["pooled_output"]):
            diff = sig_fig_compare(data, truth_result["save_infer_model/scale_0.tmp_0"][i])
            assert diff < delta, f"diff is {diff} > {delta}"
        for i, data in enumerate(predict_result["sequence_output"]):
            diff = sig_fig_compare(data, truth_result["save_infer_model/scale_1.tmp_0"][i])
            assert diff < delta, f"diff is {diff} > {delta}"

    def predict_brpc(self, batch_size=1):
        reader = ChineseBertReader({"max_seq_len": 128})
        fetch = ["pooled_output", "sequence_output"]
        endpoint_list = ['127.0.0.1:9292']
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
        print(result)
        return result

    def predict_http(self, mode="proto", compress=False, batch_size=1):
        reader = ChineseBertReader({"max_seq_len": 128})
        fetch = ["pooled_output", "sequence_output"]
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
        # TODO 优化转换方式 最终转换为dict包numpy
        if isinstance(result, dict):
            pooled_output = np.array(result["outputs"][0]["tensor"][0]["float_data"]).reshape(result["outputs"][0]["tensor"][0]["shape"])
            sequence_output = np.array(result["outputs"][0]["tensor"][1]["float_data"]).reshape(result["outputs"][0]["tensor"][1]["shape"])
        else:
            pooled_output = np.array(result.outputs[0].tensor[0].float_data).reshape(result.outputs[0].tensor[0].shape)
            sequence_output = np.array(result.outputs[0].tensor[1].float_data).reshape(result.outputs[0].tensor[1].shape)
        result_dict = {}
        result_dict["pooled_output"] = pooled_output
        result_dict["sequence_output"] = sequence_output
        return result_dict

    def test_cpu(self, delta=1e-3):

        # 1.start server
        self.serving_util.start_server_by_shell(f"{self.serving_util.py_version} -m paddle_serving_server.serve --model bert_seq128_model/ --port 9292")

        # 2.resource check
        assert count_process_num_on_port(9292) == 1
        assert check_gpu_memory(0) is False

        # 3.keywords check

        # 4.predict
        # by pybind-brpc_client
        result_data = self.predict_brpc(batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        result_data = self.predict_brpc(batch_size=2)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=2)

        # by HTTP-proto
        result_data = self.predict_http(mode="proto", compress=False, batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        result_data = self.predict_http(mode="proto", compress=False, batch_size=2)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=2)
        result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        # by HTTP-json
        result_data = self.predict_http(mode="json", compress=False, batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        # by HTTP-grpc
        result_data = self.predict_http(mode="grpc", compress=False, batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)

        # 5.release
        kill_process(9292)

    def test_gpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model bert_seq128_model/ --port 9292 --gpu_ids 0",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9292) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is False

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU")

        # 4.predict
        # by pybind-brpc_client
        result_data = self.predict_brpc(batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        result_data = self.predict_brpc(batch_size=2)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=2)

        # by HTTP-proto
        result_data = self.predict_http(mode="proto", compress=False, batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        result_data = self.predict_http(mode="proto", compress=False, batch_size=2)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=2)
        result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        # by HTTP-json
        result_data = self.predict_http(mode="json", compress=False, batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)
        # by HTTP-grpc
        result_data = self.predict_http(mode="grpc", compress=False, batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)

        # 5.release
        kill_process(9292, 2)
