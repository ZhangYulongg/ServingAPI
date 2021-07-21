import os
import pytest
import subprocess
import time
import numpy as np
import requests
from multiprocessing import Process
import json

from paddle_serving_server.web_service import WebService
import paddle_serving_server.serve
from paddle_serving_client import Client

from test_dag import TestOpSeqMaker
from util import kill_process, check_gpu_memory


class TestWebService(object):
    def setup(self):
        dir_name = os.path.dirname(os.path.abspath(__file__))
        self.dir = dir_name
        self.model_dir = dir_name + "/uci_housing_model"
        test_service = WebService("test_web_service")
        test_service.load_model_config(self.model_dir)
        self.test_service = test_service

    def teardown(self):
        os.system("rm -rf workdir*")
        os.system("rm -rf PipelineServingLogs")

    def predict_brpc(self):
        client = Client()
        client.load_client_config(self.dir + "/uci_housing_client/serving_client_conf.prototxt")
        client.connect(["127.0.0.1:12000"])

        data = np.array(
            [[0.0137, -0.1136, 0.2553, -0.0692, 0.0582, -0.0727, -0.1583, -0.0584, 0.6283, 0.4919, 0.1856, 0.0795,
              -0.0332]])
        fetch_map = client.predict(
            feed={"x": data}, fetch=["price"], batch=True)
        print("fetch_map:", fetch_map)
        return fetch_map['price']

    @staticmethod
    def predict_http():
        web_url = "http://127.0.0.1:9696/test_web_service/prediction"
        data = {"feed": [{"x": [0.0137, -0.1136, 0.2553, -0.0692, 0.0582, -0.0727, -0.1583, -0.0584, 0.6283, 0.4919,
                                0.1856, 0.0795, -0.0332]}], "fetch": ["price"]}

        result = requests.post(url=web_url, data=json.dumps(data))
        print(result)
        return result

    def test_load_model_config(self):
        # config_dir list
        assert self.test_service.server_config_dir_paths == [self.model_dir]
        # feed_vars
        feed_vars = self.test_service.feed_vars['x']
        assert feed_vars.name == "x"
        assert feed_vars.alias_name == "x"
        assert feed_vars.is_lod_tensor is False
        assert feed_vars.feed_type == 1
        assert feed_vars.shape == [13]
        # fetch_vars
        fetch_vars = self.test_service.fetch_vars["fc_0.tmp_1"]
        assert fetch_vars.name == "fc_0.tmp_1"
        assert fetch_vars.alias_name == "price"
        assert fetch_vars.is_lod_tensor is False
        assert fetch_vars.fetch_type == 1
        assert fetch_vars.shape == [1]
        # client config_path list
        assert self.test_service.client_config_path == [self.model_dir + '/serving_server_conf.prototxt']

    def test_prepare_server(self):
        self.test_service.prepare_server(workdir="workdir", port=9696, device="cpu")
        assert self.test_service.workdir == "workdir"
        assert self.test_service.port == 9696
        assert self.test_service.port_list == [12000]

    def test_default_rpc_service(self):
        self.test_service.prepare_server(workdir="workdir", port=9696, device="cpu")
        test_server = self.test_service.default_rpc_service(workdir="workdir", port=self.test_service.port_list[0],
                                                            gpus=-1)
        # check bRPC server params
        assert test_server.port == 12000
        assert test_server.workdir == "workdir"
        assert test_server.device == "cpu"
        # check workflows list
        workflows = test_server.workflow_conf.workflows
        assert len(workflows) == 1
        TestOpSeqMaker.check_standard_workflow(workflows[0])

    def test_create_rpc_config_with_cpu(self):
        self.test_service.prepare_server(workdir="workdir", port=9696, device="cpu")
        self.test_service.create_rpc_config()
        rpc_list = self.test_service.rpc_service_list
        assert len(rpc_list) == 1
        assert isinstance(rpc_list[0], paddle_serving_server.server.Server)

    def test_create_rpc_config_with_gpu(self):
        self.test_service.set_gpus("0,1")
        self.test_service.prepare_server(workdir="workdir", port=9696, device="gpu")
        self.test_service.create_rpc_config()
        rpc_list = self.test_service.rpc_service_list
        assert len(rpc_list) == 1
        assert isinstance(rpc_list[0], paddle_serving_server.server.Server)

    def test_set_gpus(self):
        self.test_service.set_gpus("1,2,3")
        assert self.test_service.gpus == ["1,2,3"]

    def test_run_rpc_service_with_gpu(self):
        self.test_service.set_gpus("0,1")
        self.test_service.prepare_server(workdir="workdir", port=9696, device="gpu")
        self.test_service.run_rpc_service()
        os.system("sleep 5")

        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True

        price = self.predict_brpc()
        assert price == np.array([[18.901152]], dtype=np.float32)

        kill_process(12000)

    def test_run_web_service(self):
        # TODO fix
        self.test_service.set_gpus("0,1")
        self.test_service.prepare_server(workdir="workdir", port=9696, device="gpu")
        self.test_service.run_rpc_service()
        p = Process(target=self.test_service.run_web_service)
        p.start()
        os.system("sleep 5")

        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True

        result = self.predict_http()

        kill_process(9696)
        kill_process(12000)
        pass


if __name__ == '__main__':
    tws = TestWebService()
    tws.setup()
    tws.test_run_web_service()
    # test_load_model_config()
    # test_prepare_server()
    # test_default_rpc_service()
    # test_create_rpc_config_with_cpu()
    # test_set_gpus()
    pass
