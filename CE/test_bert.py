import os
import subprocess
import numpy as np
import time

from paddle_serving_client import Client, HttpClient
from paddle_serving_app.reader import ChineseBertReader
import paddle.inference as paddle_infer

from util import *


class TestBert(object):
    def setup_class(self):
        code_path = os.environ.get("CODE_PATH")
        self.data_path = f"{os.environ.get('DATA_PATH')}/bert/"
        example_path = f"{code_path}/Serving/python/examples/bert/"
        self.py_version = os.environ.get("py_version")
        self.client_config = f"{example_path}/bert_seq128_client/serving_client_conf.prototxt"

        os.system(f"cd {example_path}")
        print("======================cur path======================")
        print(example_path)
        self.check_model_data_exist()
        self.get_truth_val_by_inference()

    def teardown_method(self):
        kill_process(9292, 2)

    def check_model_data_exist(self):
        if not os.path.exists("./bert_seq128_model"):
            # 软链模型数据
            dir_path, dir_names, file_names = next(os.walk(self.data_path))
            for dir_ in dir_names:
                abs_path = os.path.join(dir_path, dir_)
                os.system(f"ln -s {abs_path} {dir_}")
            for file in file_names:
                abs_path = os.path.join(dir_path, file)
                os.system(f"ln -s {abs_path} {file}")

    def get_truth_val_by_inference(self):
        reader = ChineseBertReader({"max_seq_len": 128})
        feed_dict = reader.process("送晚了，饿得吃得很香")

        for key in feed_dict.keys():
            feed_dict[key] = np.array(feed_dict[key]).reshape((1, 128, 1))

        pd_config = paddle_infer.Config("bert_seq128_model")
        pd_config.disable_gpu()
        pd_config.switch_ir_optim(False)

        predictor = paddle_infer.create_predictor(pd_config)

        input_names = predictor.get_input_names()
        input_ids = predictor.get_input_handle(input_names[0])
        position_ids = predictor.get_input_handle(input_names[1])
        segment_ids = predictor.get_input_handle(input_names[2])
        input_mask = predictor.get_input_handle(input_names[3])

        input_mask.copy_from_cpu(feed_dict["input_mask"].astype("float32"))
        position_ids.copy_from_cpu(feed_dict["position_ids"].astype("int64"))
        input_ids.copy_from_cpu(feed_dict["input_ids"].astype("int64"))
        segment_ids.copy_from_cpu(feed_dict["segment_ids"].astype("int64"))

        predictor.run()

        output_data_dict = {}
        output_names = predictor.get_output_names()
        for _, output_data_name in enumerate(output_names):
            output_handle = predictor.get_output_handle(output_data_name)
            output_data = output_handle.copy_to_cpu()
            output_data_dict[output_data_name] = output_data
        self.truth_val = output_data_dict

    def check_result(self, result_data, batch_size=1, delta=1e-3):
        # flatten
        predict_result = {}
        truth_result = {}
        for key, value in result_data.items():
            predict_result[key] = value.flatten()
        for key, value in self.truth_val.items():
            truth_result[key] = np.repeat(value, repeats=batch_size, axis=0).flatten()

        # compare
        for i, data in enumerate(predict_result["pooled_output"]):
            diff = sig_fig_compare(data, truth_result["save_infer_model/scale_0.tmp_0"][i])
            assert diff < delta, f"diff is {diff} > {delta}"
        for i, data in enumerate(predict_result["sequence_output"]):
            diff = sig_fig_compare(data, truth_result["save_infer_model/scale_1.tmp_0"][i])
            assert diff < delta, f"diff is {diff} > {delta}"

    def start_server_by_shell(self, cmd: str, sleep: int = 5):
        self.err = open("stderr.log", "w")
        self.out = open("stdout.log", "w")
        p = subprocess.Popen(cmd, shell=True, stdout=self.out, stderr=self.err)
        os.system(f"sleep {sleep}")

        print("======================stderr.log======================")
        os.system("cat stderr.log")
        print("======================stdout.log======================")
        os.system("cat stdout.log")
        print("======================================================")

    def predict_brpc(self, batch_size=1):
        reader = ChineseBertReader({"max_seq_len": 128})
        fetch = ["pooled_output", "sequence_output"]
        endpoint_list = ['127.0.0.1:9292']
        client = Client()
        client.load_client_config(self.client_config)
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
        client = HttpClient(ip='127.0.0.1', port='9292')
        client.load_client_config(self.client_config)
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
        self.start_server_by_shell(f"{self.py_version} -m paddle_serving_server.serve --model bert_seq128_model/ --port 9292")

        # 2.resource check
        assert count_process_num_on_port(9292) == 1
        assert check_gpu_memory(0) is False

        # 3.keywords check

        # 4.predict
        # by pybind-brpc_client
        result_data = self.predict_brpc(batch_size=1)
        self.check_result(result_data=result_data, batch_size=1)
        result_data = self.predict_brpc(batch_size=2)
        self.check_result(result_data=result_data, batch_size=2)

        # by HTTP-proto
        result_data = self.predict_http(mode="proto", compress=False, batch_size=1)
        self.check_result(result_data=result_data, batch_size=1)
        result_data = self.predict_http(mode="proto", compress=False, batch_size=2)
        self.check_result(result_data=result_data, batch_size=2)
        result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        self.check_result(result_data=result_data, batch_size=1)
        # by HTTP-json
        result_data = self.predict_http(mode="json", compress=False, batch_size=1)
        self.check_result(result_data=result_data, batch_size=1)
        # by HTTP-grpc
        result_data = self.predict_http(mode="grpc", compress=False, batch_size=1)
        self.check_result(result_data=result_data, batch_size=1)

        # 5.release
        kill_process(9292)

    def test_gpu(self):
        # 1.start server
        self.start_server_by_shell(
            cmd=f"{self.py_version} -m paddle_serving_server.serve --model bert_seq128_model/ --port 9292 --gpu_ids 0",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9292) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is False

        # 3.keywords check

        # 4.predict
        # by pybind-brpc_client
        result_data = self.predict_brpc(batch_size=1)
        self.check_result(result_data=result_data, batch_size=1)
        result_data = self.predict_brpc(batch_size=2)
        self.check_result(result_data=result_data, batch_size=2)

        # by HTTP-proto
        result_data = self.predict_http(mode="proto", compress=False, batch_size=1)
        self.check_result(result_data=result_data, batch_size=1)
        result_data = self.predict_http(mode="proto", compress=False, batch_size=2)
        self.check_result(result_data=result_data, batch_size=2)
        result_data = self.predict_http(mode="proto", compress=True, batch_size=1)
        self.check_result(result_data=result_data, batch_size=1)
        # by HTTP-json
        result_data = self.predict_http(mode="json", compress=False, batch_size=1)
        self.check_result(result_data=result_data, batch_size=1)
        # by HTTP-grpc
        result_data = self.predict_http(mode="grpc", compress=False, batch_size=1)
        self.check_result(result_data=result_data, batch_size=1)

        # 5.release
        kill_process(9292, 2)

if __name__ == '__main__':
    tb = TestBert()
    tb.setup_class()
    # tb.test_cpu()
    tb.test_gpu()
    # tb.teardown_method()
    # tb.get_truth_val_by_inference()