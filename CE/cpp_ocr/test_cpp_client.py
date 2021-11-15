import os
import subprocess
import numpy as np
import copy
import cv2
import re
import sys

from paddle_serving_client import Client, HttpClient
from paddle_serving_client.io import inference_model_to_serving
from paddle_serving_app.reader import SegPostprocess
from paddle_serving_app.reader import *
import paddle.inference as paddle_infer

sys.path.append("../")
from util import *


class TestCPPClient(object):
    """only for CI CE拿不到client的bin"""
    def setup_class(self):
        serving_util = ServingTest(data_path="fit_a_line", example_path="C++/fit_a_line", model_dir="uci_housing_model",
                                   client_dir="uci_housing_client")
        serving_util.check_model_data_exist()
        os.system("\cp -r ${CODE_PATH}/Serving/client-build/core/general-client/simple_client ./")
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util

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
        print(output_data_dict)
        self.truth_val = output_data_dict
        print(self.truth_val, self.truth_val["fc_0.tmp_1"].shape)

    def predict_cpp(self, batch_size=1):
        self.serving_util.start_server_by_shell(
            cmd=f"./simple_client --client_conf='uci_housing_client/serving_client_conf.prototxt' --server_port='127.0.0.1:9494' --test_type='brpc' --sample_type='fit_a_line'",
            sleep=0,
            err="errout.log",
            out="out.log",
            wait=True,
        )
        with open("errout.log", "r") as f:
            words = f.readlines()
        line = words[-1].split(" ")[-1]
        result = float(re.findall(r"\d+\.\d+", line)[0])
        fetch_map = {"fc_0.tmp_1": np.array([[result]]).astype("float32")}
        print("simple_client result: ", fetch_map)
        return fetch_map

    def test_cpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model uci_housing_model --thread 10 --port 9494",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9494) == 1
        # assert check_gpu_memory(0) is False

        # 3.keywords check

        # 4.predict by brpc
        # batch_size 1
        result_data = self.predict_cpp()
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)

        # 5.release
        kill_process(9494)
