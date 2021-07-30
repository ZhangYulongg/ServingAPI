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
from paddle_serving_server.pipeline import PipelineServer

# from resnet_pipeline import ImagenetOp, ImageService
from pipeline.resnet_pipeline import ImagenetOp, ImageService


class TestPipelineServer(object):
    """test PipelineServer class"""

    def setup_class(self):
        """setup func for pipeline server config"""
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

    def setup_method(self):
        """setup func"""
        self.dir = os.path.dirname(os.path.abspath(__file__))

        image_service = ImageService()
        pipeline_server = PipelineServer()

        read_op = pipeline.RequestOp()
        last_op = image_service.get_pipeline_response(read_op=read_op)
        response_op = pipeline.ResponseOp(input_ops=[last_op])
        self.read_op = read_op
        self.last_op = last_op
        self.used_op = [read_op, last_op]
        self.response_op = response_op
        self.pipeline_server = pipeline_server

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
    def test_run_server_with_one_grpcprocess(self):
        """test run pipeline server"""

        pass


class TestServerYamlConfChecker(object):
    """test ServerYamlConfChecker class"""

    def setup_class(self):
        """setup func"""
        self.dir = os.path.dirname(os.path.abspath(__file__))
        with io.open(f"{self.dir}/config.yml", encoding="utf-8") as f:
            conf = yaml.load(f.read())
        self.conf = conf

    def test_load_server_yaml_conf(self):
        """test pipeline server load yaml file"""

        pass

    def test_check_server_conf(self):
        """test check server config"""

        pass


if __name__ == "__main__":
    tps = TestPipelineServer()
    tps.setup_method()
    tps.test_set_response_op()
    tps.test_prepare_server()
    # resnet_service = ImageService(name="imagenet")
    # resnet_service.prepare_pipeline_config("config.yml")
    # resnet_service.run_service()
    # tsycc = TestServerYamlConfChecker()
    # tsycc.setup_class()
    # tsycc.test_check_server_conf()
    pass
