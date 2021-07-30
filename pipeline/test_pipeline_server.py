"""
test pipeline.pipeline_server module
"""
import os
import sys
import logging
import io
import base64
import numpy as np
import cv2
import pytest
import yaml
import requests
import json
from multiprocessing import Process
import subprocess
import time

from paddle_serving_app.reader import (
    Sequential,
    URL2Image,
    Resize,
    CenterCrop,
    RGB2BGR,
    Transpose,
    Div,
    Normalize,
    Base64ToImage,
)
from paddle_serving_server.web_service import WebService, Op
from paddle_serving_server import pipeline
from paddle_serving_server.pipeline import PipelineServer, PipelineClient

from resnet_pipeline import ImagenetOp, ImageService
# from pipeline.resnet_pipeline import ImagenetOp, ImageService
sys.path.append("../paddle_serving_server")
from util import default_args, kill_process, check_gpu_memory, cv2_to_base64, count_process_num_on_port


class TestPipelineServer(object):
    """test PipelineServer class"""

    def setup_class(self):
        """setup func"""
        self.dir = os.path.dirname(os.path.abspath(__file__))
        self.img_path = f"{self.dir}/../paddle_serving_server/daisy.jpg"

    def setup_method(self):
        """setup func"""
        image_service = ImageService()
        pipeline_server = PipelineServer("imagenet")

        read_op = pipeline.RequestOp()
        last_op = image_service.get_pipeline_response(read_op=read_op)
        response_op = pipeline.ResponseOp(input_ops=[last_op])
        self.read_op = read_op
        self.last_op = last_op
        self.used_op = [read_op, last_op]
        self.response_op = response_op
        self.pipeline_server = pipeline_server

        self.default_yml_dict = {
            "build_dag_each_worker": False,
            "worker_num": 1,
            "http_port": 18080,
            "rpc_port": 9993,
            "dag": {
                "is_thread_op": False
            },
            "op": {
                "imagenet": {
                    "concurrency": 1,
                    "local_service_conf": {
                        "model_config": "../paddle_serving_server/resnet_v2_50_imagenet_model/",
                        "device_type": 0,
                        "devices": "",
                        "client_type": "local_predictor",
                        "fetch_list": ["score"],
                        "ir_optim": False
                    }
                }
            }
        }

    def predict_rpc(self):
        """test predict by rpc"""
        client = PipelineClient()
        client.connect(["127.0.0.1:9993"])
        with open(self.img_path, "rb") as file:
            image_data = file.read()
        image = cv2_to_base64(image_data)

        result = client.predict(feed_dict={"image": image}, fetch=["label", "prob"])
        return result

    def predict_http(self):
        """test predict by http(grpc gateway)"""
        url = "http://127.0.0.1:18080/imagenet/prediction"
        with open(self.img_path, "rb") as file:
            image_data = file.read()
        image = cv2_to_base64(image_data)
        data = {"key": ["image"], "value": [image]}

        result = requests.post(url=url, data=json.dumps(data))
        return result.json()

    def test_set_response_op(self):
        """test set_response_op"""
        self.pipeline_server.set_response_op(self.response_op)

        used_op = list(self.pipeline_server._used_op)
        used_op.sort(key=self.used_op.index)
        assert self.pipeline_server._response_op is self.response_op
        assert isinstance(used_op[0], pipeline.operator.RequestOp)
        assert isinstance(used_op[-1], ImagenetOp)

    def test_prepare_server(self):
        """test pipeline read config.yaml"""
        self.pipeline_server.set_response_op(self.response_op)
        self.pipeline_server.prepare_server(yml_file=f"{self.dir}/config.yml")

        right_conf = {
            "worker_num": 1,
            "http_port": 18080,
            "rpc_port": 9993,
            "dag": {
                "is_thread_op": False,
                "retry": 1,
                "client_type": "brpc",
                "use_profile": False,
                "channel_size": 0,
                "tracer": {"interval_s": -1},
            },
            "op": {
                "imagenet": {
                    "concurrency": 1,
                    "local_service_conf": {
                        "model_config": "../paddle_serving_server/resnet_v2_50_imagenet_model/",
                        "device_type": 0,
                        "devices": "",
                        "client_type": "local_predictor",
                        "fetch_list": ["score"],
                        "workdir": "",
                        "thread_num": 2,
                        "mem_optim": True,
                        "ir_optim": False,
                        "precision": "fp32",
                        "use_calib": False,
                        "use_mkldnn": False,
                        "mkldnn_cache_capacity": 0,
                    },
                    "timeout": -1,
                    "retry": 1,
                    "batch_size": 1,
                    "auto_batching_timeout": -1,
                }
            },
            "build_dag_each_worker": False,
        }

        assert self.pipeline_server._rpc_port == 9993
        assert self.pipeline_server._http_port == 18080
        assert self.pipeline_server._worker_num == 1
        assert self.pipeline_server._build_dag_each_worker is False
        assert self.pipeline_server._conf == right_conf
        print(self.pipeline_server._name)

    @pytest.mark.api_pipelinePipelineServer_runServer_parameters
    def test_run_server_cpu_1proc_noir_nomkl(self):
        """test run pipeline server"""
        self.pipeline_server.set_response_op(self.response_op)
        self.pipeline_server.prepare_server(yml_dict=self.default_yml_dict)
        p = Process(target=self.pipeline_server.run_server)
        p.start()
        os.system("sleep 5")

        # TODO 封装优化
        assert count_process_num_on_port(9993) == 1
        assert check_gpu_memory(0) is False

        # predict by rpc
        result = self.predict_rpc()
        print("RPC result:\n", result)
        assert result.key == ["label", "prob"]
        assert result.value == ["['daisy']", "[0.9341403245925903]"]

        # predict by http
        result = self.predict_http()
        print("HTTP result:\n", result)
        assert result["key"] == ["label", "prob"]
        assert result["value"] == ["['daisy']", "[0.9341403245925903]"]

        kill_process(9993)
        os.system("kill -9 $(netstat -nlp | grep 'LISTENING' | awk '{print $9}' | awk -F'/' '{{ print $1 }}')")
        kill_process(18080, 1)

    @pytest.mark.api_pipelinePipelineServer_runServer_parameters
    def test_run_server_cpu_3proc_noir_nomkl(self):
        """worker_num 3  ir_optim off  mkldnn off"""
        self.pipeline_server.set_response_op(self.response_op)
        self.default_yml_dict["build_dag_each_worker"] = True
        self.default_yml_dict["worker_num"] = 3
        self.pipeline_server.prepare_server(yml_dict=self.default_yml_dict)
        p = Process(target=self.pipeline_server.run_server)
        p.start()
        os.system("sleep 5")

        assert count_process_num_on_port(9993) == 3
        assert check_gpu_memory(0) is False

        # predict by rpc
        result = self.predict_rpc()
        print("RPC result:\n", result)
        assert result.key == ["label", "prob"]
        assert result.value == ["['daisy']", "[0.9341403245925903]"]

        # predict by http
        result = self.predict_http()
        print("HTTP result:\n", result)
        assert result["key"] == ["label", "prob"]
        assert result["value"] == ["['daisy']", "[0.9341403245925903]"]

        kill_process(9993)
        os.system("kill -9 $(netstat -nlp | grep 'LISTENING' | awk '{print $9}' | awk -F'/' '{{ print $1 }}')")
        kill_process(18080, 1)

    def test_run_server_cpu_3proc_ir_mkl(self):
        """worker_num 3  ir_optim on  mkldnn on"""
        # self.pipeline_server.set_response_op(self.response_op)
        # self.default_yml_dict["build_dag_each_worker"] = True
        # self.default_yml_dict["worker_num"] = 3
        # self.default_yml_dict["op"]["imagenet"]["local_service_conf"]["ir_optim"] = True
        # self.default_yml_dict["op"]["imagenet"]["local_service_conf"]["use_mkldnn"] = True
        # self.pipeline_server.prepare_server(yml_dict=self.default_yml_dict)
        # p = Process(target=self.pipeline_server.run_server)
        # p.start()
        err = open("stderr.log", "w")
        out = open("stdout.log", "w")
        p = subprocess.Popen("python3.6 resnet_pipeline.py", shell=True, stdout=out, stderr=err)

        # print(p.stdout.read())

        os.system("sleep 5")
        kill_process(9993)
        os.system("kill -9 $(netstat -nlp | grep 'LISTENING' | awk '{print $9}' | awk -F'/' '{{ print $1 }}')")
        kill_process(18080, 1)
        err.close()
        out.close()
        # plan1
        start = time.time()
        p = subprocess.Popen("grep 'MKLDNN is enabled' stderr.log", shell=True)
        print("plan1:", p.returncode)
        end = time.time()
        time1 = end - start
        # plan2
        start = time.time()
        p = os.system("grep 'MKLDNN is enabled' stderr.log")
        print("plan2:", p)
        end = time.time()
        time2 = end - start

        print(time1, time2)



if __name__ == "__main__":
    tps = TestPipelineServer()
    tps.setup_class()
    tps.setup_method()
    # tps.test_set_response_op()
    tps.test_run_server_cpu_3proc_ir_mkl()
    # resnet_service = ImageService(name="imagenet")
    # resnet_service.prepare_pipeline_config("config.yml")
    # resnet_service.run_service()
    # tsycc = TestServerYamlConfChecker()
    # tsycc.setup_class()
    # tsycc.test_check_server_conf()
    pass
