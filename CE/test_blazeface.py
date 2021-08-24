import os
import subprocess
import numpy as np

from paddle_serving_client import Client, HttpClient
from paddle_serving_app.reader import BlazeFacePostprocess
from paddle_serving_app.reader import *
import paddle.inference as paddle_infer

from util import *


class TestBlazeface(object):
    def setup_class(self):
        code_path = os.environ.get("CODE_PATH")
        self.data_path = f"{os.environ.get('DATA_PATH')}/blazeface/"
        example_path = f"{code_path}/Serving/python/examples/blazeface/"
        self.py_version = os.environ.get("py_version")
        self.client_config = f"{example_path}/serving_client/serving_client_conf.prototxt"

        os.system(f"cd {example_path}")
        print("======================cur path======================")
        print(example_path)
        self.check_model_data_exist(self)
        self.get_truth_val_by_inference(self)

    def teardown_method(self):
        print("======================stderr.log after predict======================")
        os.system("cat stderr.log")
        print("======================stdout.log after predict======================")
        os.system("cat stdout.log")
        print("======================================================")

    def check_model_data_exist(self):
        if not os.path.exists("./serving_server"):
            # 软链模型数据
            dir_path, dir_names, file_names = next(os.walk(self.data_path))
            for dir_ in dir_names:
                abs_path = os.path.join(dir_path, dir_)
                os.system(f"ln -s {abs_path} {dir_}")
            for file in file_names:
                abs_path = os.path.join(dir_path, file)
                os.system(f"ln -s {abs_path} {file}")

    def get_truth_val_by_inference(self):
        preprocess = Sequential([
            File2Image(),
            Normalize([104, 117, 123], [127.502231, 127.502231, 127.502231], False)
        ])
        postprocess = BlazeFacePostprocess("label_list.txt", "output")
        im_0 = preprocess("test.jpg")
        tmp = Transpose((2, 0, 1))
        # expand batch dim
        im = tmp(im_0)[np.newaxis, :]

        pd_config = paddle_infer.Config("serving_server")
        pd_config.disable_gpu()
        pd_config.switch_ir_optim(False)

        predictor = paddle_infer.create_predictor(pd_config)

        input_names = predictor.get_input_names()
        image = predictor.get_input_handle(input_names[0])
        image.copy_from_cpu(im.astype("float32"))

        predictor.run()

        output_data_dict = {}
        output_names = predictor.get_output_names()
        for _, output_data_name in enumerate(output_names):
            output_handle = predictor.get_output_handle(output_data_name)
            output_data = output_handle.copy_to_cpu()
            output_data_dict[output_data_name] = output_data
        self.truth_val = output_data_dict
        print(self.truth_val, self.truth_val["save_infer_model/scale_0.tmp_0"].shape)

    def check_result(self, result_data, truth_data, batch_size=1, delta=1e-3):
        # flatten
        predict_result = {}
        truth_result = {}
        for key, value in result_data.items():
            predict_result[key] = value.flatten()
        for key, value in truth_data.items():
            truth_result[key] = np.repeat(value, repeats=batch_size, axis=0).flatten()

        # compare
        for i, data in enumerate(predict_result["detection_output_0.tmp_0"]):
            diff = sig_fig_compare(data, truth_result["save_infer_model/scale_0.tmp_0"][i])
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
        preprocess = Sequential([
            File2Image(),
            Normalize([104, 117, 123], [127.502231, 127.502231, 127.502231], False)
        ])
        postprocess = BlazeFacePostprocess("label_list.txt", "output")
        im_0 = preprocess("test.jpg")
        tmp = Transpose((2, 0, 1))
        # expand batch dim
        im = tmp(im_0)[np.newaxis, :]
        fetch = ["detection_output_0.tmp_0"]
        endpoint_list = ['127.0.0.1:9494']

        client = Client()
        client.load_client_config(self.client_config)
        client.connect(endpoint_list)

        fetch_map = client.predict(feed={"image": im}, fetch=fetch, batch=True)

        print("fetch_map", fetch_map)
        print(fetch_map["detection_output_0.tmp_0"].shape)
        return fetch_map

    def test_cpu(self):
        # 1.start server
        self.start_server_by_shell(
            cmd=f"{self.py_version} -m paddle_serving_server.serve --model serving_server --port 9494",
            sleep=7,
        )

        # 2.resource check
        assert count_process_num_on_port(9494) == 1
        assert check_gpu_memory(0) is False

        # 3.keywords check

        # 4.predict
        result_data = self.predict_brpc(batch_size=1)
        self.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)

        # 5.release
        kill_process(9494)

    def test_gpu(self):
        # 1.start server
        self.start_server_by_shell(
            cmd=f"{self.py_version} -m paddle_serving_server.serve --model serving_server --port 9494 --gpu_ids 1",
            sleep=7,
        )

        # 2.resource check
        assert count_process_num_on_port(9494) == 1
        assert check_gpu_memory(0) is False
        assert check_gpu_memory(1) is True

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU")

        # 4.predict
        result_data = self.predict_brpc(batch_size=1)
        self.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)

        # 5.release
        kill_process(9494)
