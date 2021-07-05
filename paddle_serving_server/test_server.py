import os
import pytest
import subprocess
import paddle_serving_server.serve
import time
from paddle_serving_server.server import Server
import paddle_serving_server as serving
import argparse
import collections


class TestServer(object):
    def setup(self):
        op_maker = serving.OpMaker()
        op_seq_maker = serving.OpSeqMaker()
        read_op = op_maker.create('general_reader')
        op_seq_maker.add_op(read_op)
        infer_op_name = "general_infer"
        general_infer_op = op_maker.create(infer_op_name)
        op_seq_maker.add_op(general_infer_op)
        general_response_op = op_maker.create('general_response')
        op_seq_maker.add_op(general_response_op)

        dir_name = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = dir_name + "/uci_housing_model"
        self.test_server = Server()
        self.test_server.set_op_sequence(op_seq_maker.get_op_sequence())
        self.test_server.load_model_config(self.model_dir)

    def teardown(self):
        os.system("rm -rf workdir*")
        os.system("rm -rf PipelineServingLogs")

    def test_load_model_config(self):
        test_model_conf = str(self.test_server.model_conf['general_infer_0']).split()
        assert test_model_conf == ['feed_var', '{', 'name:', '"x"', 'alias_name:', '"x"', 'is_lod_tensor:', 'false',
                                   'feed_type:', '1', 'shape:', '13', '}', 'fetch_var', '{', 'name:', '"fc_0.tmp_1"',
                                   'alias_name:', '"price"', 'is_lod_tensor:', 'false', 'fetch_type:', '1', 'shape:',
                                   '1', '}']
        model_config_paths = collections.OrderedDict([('general_infer_0', self.model_dir)])
        assert self.test_server.model_config_paths == model_config_paths
        assert self.test_server.general_model_config_fn == ['general_infer_0/general_model.prototxt']
        assert self.test_server.model_toolkit_fn == ['general_infer_0/model_toolkit.prototxt']
        assert self.test_server.subdirectory == ['general_infer_0']

    def test_port_is_available_with_unused_port(self):
        assert self.test_server.port_is_available(12000) is True

    def test_port_is_available_with_used_port(self):
        os.system("python -m SimpleHTTPServer 12000 &")
        time.sleep(2)
        assert self.test_server.port_is_available(12000) is False
        os.system("kill `ps -ef | grep SimpleHTTPServer | awk '{print $2}'` > /dev/null 2>&1")

    def test_check_avx(self):
        assert self.test_server.check_avx() is False

    def test_check_local_bin_without_defined(self):
        self.test_server.check_local_bin()
        assert self.test_server.use_local_bin is False

    def test_check_local_bin_with_defined(self):
        os.environ["SERVING_BIN"] = "/home"
        self.test_server.check_local_bin()
        assert self.test_server.use_local_bin is True
        assert self.test_server.bin_path == os.environ["SERVING_BIN"]

    def test_get_fetch_list(self):
        assert self.test_server.get_fetch_list() == ['price']

    def test_prepare_engine(self):
        self.test_server._prepare_engine(self.test_server.model_config_paths, "cpu", False)
        model_toolkit_conf = ['engines', '{', 'name:', '"general_infer_0"', 'type:', '"PADDLE_INFER"',
                              'reloadable_meta:',
                              f'"{self.model_dir}/fluid_time_file"',
                              'reloadable_type:', '"timestamp_ne"', 'model_dir:',
                              f'"{self.model_dir}"',
                              'runtime_thread_num:', '0', 'batch_infer_size:', '0', 'enable_batch_align:', '0',
                              'enable_memory_optimization:', 'false', 'enable_ir_optimization:', 'false', 'use_trt:',
                              'false', 'use_lite:', 'false', 'use_xpu:', 'false', 'use_gpu:', 'false',
                              'combined_model:', 'false', '}']
        assert str(self.test_server.model_toolkit_conf[0]).split() == model_toolkit_conf

    def test_prepare_infer_service(self):
        self.test_server._prepare_infer_service(9696)
        infer_service_conf = ['port:', '9696', 'services', '{', 'name:', '"GeneralModelService"', 'workflows:',
                              '"workflow1"', '}']
        assert str(self.test_server.infer_service_conf).split() == infer_service_conf

    def test_prepare_resource(self):
        workdir = "workdir_0"
        os.system("mkdir -p {}".format(workdir))
        for subdir in self.test_server.subdirectory:
            os.system("mkdir -p {}/{}".format(workdir, subdir))
            os.system("touch {}/{}/fluid_time_file".format(workdir, subdir))
        resource_conf = ['model_toolkit_path:', '"workdir_0"', 'model_toolkit_file:',
                         '"general_infer_0/model_toolkit.prototxt"', 'general_model_path:', '"workdir_0"',
                         'general_model_file:', '"general_infer_0/general_model.prototxt"']
        self.test_server._prepare_resource("workdir_0", None)
        assert str(self.test_server.resource_conf).split() == resource_conf

    def test_run_server(self):
        pass


if __name__ == '__main__':
    # args = argparse.ArgumentParser().parse_args()
    # args.thread = 10
    # print(args.thread, type(args.thread), type(args))
    # test_load_model_config()
    # test_port_is_available_with_used_port()
    # pytest.main(["-sv", "test_server.py"])
    # TestServer().test_get_fetch_list()
    ts = TestServer()
    ts.setup()
    ts.test_run_server()
    pass
