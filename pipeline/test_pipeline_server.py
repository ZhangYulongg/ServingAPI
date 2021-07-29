"""
test pipeline_server module
"""
import sys
import logging
import numpy as np
import base64
import cv2
import pytest

from paddle_serving_app.reader import Sequential, URL2Image, Resize, CenterCrop, RGB2BGR, Transpose, Div, Normalize, Base64ToImage
from paddle_serving_server.web_service import WebService, Op
from paddle_serving_server import pipeline
from paddle_serving_server.pipeline import PipelineServer

# from resnet_pipeline import ImagenetOp, ImageService
from pipeline.resnet_pipeline import ImagenetOp, ImageService


class TestPipelineServer(object):
    """test PipelineServer class"""
    def test_set_response_op(self):
        """test set_response_op"""
        image_service = ImageService()
        server = PipelineServer()

        read_op = pipeline.RequestOp()
        last_op = image_service.get_pipeline_response(read_op=read_op)
        response_op = pipeline.ResponseOp(input_ops=[last_op])
        server.set_response_op(response_op)

        used_op = list(server._used_op)
        assert server._response_op is response_op
        assert used_op[0] is read_op
        assert used_op[-1] is last_op

    def test(self):
        pass


class TestServerYamlConfChecker(object):
    """test ServerYamlConfChecker class"""
    def test_load_server_yaml_conf(self):
        """test pipeline server load yaml file"""

        pass


if __name__ == '__main__':
    tps = TestPipelineServer()
    tps.test_set_response_op()
    # resnet_service = ImageService(name="imagenet")
    # resnet_service.prepare_pipeline_config("config.yml")
    # resnet_service.run_service()
    pass
