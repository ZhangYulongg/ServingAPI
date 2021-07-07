import os
import pytest
import subprocess
import paddle_serving_server.serve
import time
from paddle_serving_server.web_service import WebService


class TestWebService(object):
    def setup(self):
        dir_name = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = dir_name + "/uci_housing_model"
        test_service = WebService("test_web_service")
        test_service.load_model_config(self.model_dir)
        self.test_service = test_service

    def teardown(self):
        os.system("rm -rf workdir*")
        os.system("rm -rf PipelineServingLogs")

    def test_load_model_config(self):
        assert self.test_service.server_config_dir_paths == [self.model_dir]
        feed_vars = str(self.test_service.feed_vars['x']).split()
        assert feed_vars == ['name:', '"x"', 'alias_name:', '"x"', 'is_lod_tensor:', 'false', 'feed_type:', '1',
                             'shape:', '13']
        fetch_vars = str(self.test_service.fetch_vars['fc_0.tmp_1']).split()
        assert fetch_vars == ['name:', '"fc_0.tmp_1"', 'alias_name:', '"price"', 'is_lod_tensor:', 'false',
                              'fetch_type:', '1', 'shape:', '1']
        assert self.test_service.client_config_path == [self.model_dir + '/serving_server_conf.prototxt']

    def test_prepare_server(self):
        self.test_service.prepare_server(workdir="workdir", port=9696, device="cpu")
        assert self.test_service.workdir == "workdir"
        assert self.test_service.port == 9696
        assert self.test_service.device == "cpu"
        assert self.test_service.port_list == [12000]

    def test_default_rpc_service(self):
        self.test_service.prepare_server(workdir="workdir", port=9696, device="cpu")
        test_server = self.test_service.default_rpc_service(workdir="workdir", port=self.test_service.port_list[0],
                                                            gpus=-1)
        assert isinstance(test_server, paddle_serving_server.server.Server)
        assert test_server.port == 12000
        assert test_server.workdir == "workdir"
        assert test_server.device == "cpu"
        workflows = str(test_server.workflow_conf).split()
        assert workflows == ['workflows', '{', 'name:', '"workflow1"', 'workflow_type:',
                             '"Sequence"', 'nodes', '{', 'name:',
                             '"general_reader_0"', 'type:', '"GeneralReaderOp"', '}', 'nodes',
                             '{', 'name:', '"general_infer_0"',
                             'type:', '"GeneralInferOp"', 'dependencies', '{', 'name:',
                             '"general_reader_0"', 'mode:', '"RO"', '}',
                             '}', 'nodes', '{', 'name:', '"general_response_0"', 'type:',
                             '"GeneralResponseOp"', 'dependencies',
                             '{', 'name:', '"general_infer_0"', 'mode:', '"RO"', '}', '}', '}']

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
        assert self.test_service.gpus == "1,2,3"

    def test_run_rpc_service(self):
        pass


if __name__ == '__main__':
    # test_load_model_config()
    # test_prepare_server()
    # test_default_rpc_service()
    # test_create_rpc_config_with_cpu()
    # test_set_gpus()
    pass
