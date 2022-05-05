import os
import subprocess
import numpy as np
import copy
import cv2
import requests
import json
import sys
import yaml

from paddle_serving_server.pipeline import PipelineClient
from paddle_serving_app.reader import CenterCrop, RGB2BGR, Transpose, Div, Normalize, RCNNPostprocess
from paddle_serving_app.reader import Sequential, File2Image, Resize, Transpose, BGR2RGB, SegPostprocess
import paddle.inference as paddle_infer

sys.path.append("../../")
from util import *


class TestPPTSN_K400(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="PaddleVideo/PPTSN_K400", example_path="Pipeline/PaddleVideo/PPTSN_K400", model_dir="serving_server",
                                   client_dir="serving_client")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util
        # 读取yml文件
        with open("config.yml", "r") as file:
            dict_ = yaml.safe_load(file)
        self.default_config = dict_

    def teardown_method(self):
        print_log(["stderr.log", "stdout.log",
                   "log/serving.ERROR", "PipelineServingLogs/pipeline.log"], iden="after predict")
        kill_process(9993)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        # video模型暂不校验精度
        pass

    def predict_pipeline_rpc(self, batch_size=1):
        # TODO:rpc请求待补充
        pass

    def predict_pipeline_http(self, batch_size=1):
        url = "http://127.0.0.1:9999/ppTSN/prediction"
        video_url = "https://paddle-serving.bj.bcebos.com/huangjianhui04/example.avi"
        for i in range(4):
            data = {"key": ["filename"], "value": [video_url]}
            r = requests.post(url=url, data=json.dumps(data))
            result = r.json()
            print(result)

        return result

    def test_gpu(self):
        # 1.start server
        config = copy.deepcopy(self.default_config)
        with open("config.yml", "w") as f:
            yaml.dump(config, f)
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} web_service.py",
            sleep=10,
        )

        # 2.resource check
        assert count_process_num_on_port(9999) == 1  # gRPC Server
        assert count_process_num_on_port(18090) == 1  # gRPC gateway 代理、转发
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is False

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU", filename="stderr.log")

        # 4.predict by rpc
        # batch_size=1
        # result = self.predict_pipeline_rpc(batch_size=1)
        # self.serving_util.check_result(result_data=result, truth_data=self.truth_val, batch_size=1)
        # predict by http
        result = self.predict_pipeline_http(batch_size=1)  # batch_size=1
        assert result["value"][0] == "class: ['archery'] score: [0.99897975]"

        # 5.release
        kill_process(9998, 2)
        kill_process(18082)

