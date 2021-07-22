import os
import pytest
import time
import collections
from multiprocessing import Process
import numpy as np

from paddle_serving_server.server import Server, MultiLangServer
import paddle_serving_server as serving
from paddle_serving_client import Client, MultiLangClient

from util import kill_process


class TestMultiLangServer(object):
    def setup_method(self):
        op_maker = serving.OpMaker()
        op_seq_maker = serving.OpSeqMaker()
        read_op = op_maker.create('general_reader')
        op_seq_maker.add_op(read_op)
        infer_op_name = "general_infer"
        general_infer_op = op_maker.create(infer_op_name)
        op_seq_maker.add_op(general_infer_op)
        general_response_op = op_maker.create('general_response')
        op_seq_maker.add_op(general_response_op)

        dir = os.path.dirname(os.path.abspath(__file__))
        self.dir = dir
        self.model_dir = dir + "/uci_housing_model"
        self.test_server = MultiLangServer()
        self.test_server.set_op_sequence(op_seq_maker.get_op_sequence())
        self.test_server.load_model_config(self.model_dir)

    def teardown_method(self):
        os.system("rm -rf workdir*")
        os.system("rm -rf PipelineServingLogs")
        pass

    def predict(self, port=9696):
        client = MultiLangClient()
        client.connect([f"127.0.0.1:{port}"])
        client.set_rpc_timeout_ms(12000)

        data = np.array(
            [[0.0137, -0.1136, 0.2553, -0.0692, 0.0582, -0.0727, -0.1583, -0.0584, 0.6283, 0.4919, 0.1856, 0.0795,
              -0.0332]])
        fetch_map = client.predict(
            feed={"x": data}, fetch=["price"], batch=True)
        print("fetch_map:", fetch_map)
        return fetch_map['price']

    def test_load_model_config(self):
        assert self.test_server.is_multi_model_ is False
        assert self.test_server.bclient_config_path_list == [self.model_dir]

    def test_prepare_server(self):
        os.system("netstat -nlp")
        self.test_server.prepare_server(workdir="workdir", port=9696, device="gpu", use_encryption_model=False,
                                        cube_conf=None)
        assert self.test_server.device == "gpu"
        assert self.test_server.port_list_ == [12000]
        assert self.test_server.gport_ == 9696

    @pytest.mark.run(order=2)
    def test_run_server(self):
        self.test_server.set_gpuid("0,1")
        self.test_server.prepare_server(workdir="workdir", port=9697, device="gpu", use_encryption_model=False,
                                        cube_conf=None)

        p = Process(target=self.test_server.run_server)
        # print("000000000000")
        # os.system("netstat -nlp")
        # os.system("ps -ef")
        p.start()
        os.system("sleep 10")

        price = self.predict(9697)
        assert price == np.array([[18.901152]], dtype=np.float32)

        # print("1111111111")
        # os.system("netstat -nlp")
        # os.system("ps -ef")
        kill_process(9697)
        kill_process(12000, 2)


class TestServer(object):
    def setup_method(self):
        op_maker = serving.OpMaker()
        op_seq_maker = serving.OpSeqMaker()
        read_op = op_maker.create('general_reader')
        op_seq_maker.add_op(read_op)
        infer_op_name = "general_infer"
        general_infer_op = op_maker.create(infer_op_name)
        op_seq_maker.add_op(general_infer_op)
        general_response_op = op_maker.create('general_response')
        op_seq_maker.add_op(general_response_op)

        dir = os.path.dirname(os.path.abspath(__file__))
        self.dir = dir
        self.model_dir = dir + "/uci_housing_model"
        self.test_server = Server()
        self.test_server.set_op_sequence(op_seq_maker.get_op_sequence())
        self.test_server.load_model_config(self.model_dir)

    def teardown_method(self):
        os.system("rm -rf workdir*")
        os.system("rm -rf PipelineServingLogs")
        os.system("rm -rf log")
        os.system("kill `ps -ef | grep serving | awk '{print $2}'` > /dev/null 2>&1")
        kill_process(9696)

    def predict(self):
        client = Client()
        client.load_client_config(self.dir + "/uci_housing_client/serving_client_conf.prototxt")
        client.connect(["127.0.0.1:9696"])

        data = np.array(
            [[0.0137, -0.1136, 0.2553, -0.0692, 0.0582, -0.0727, -0.1583, -0.0584, 0.6283, 0.4919, 0.1856, 0.0795,
              -0.0332]])
        fetch_map = client.predict(
            feed={"x": data}, fetch=["price"], batch=True)
        print("fetch_map:", fetch_map)
        return fetch_map['price']

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
        assert self.test_server.port_is_available(12003) is True

    def test_port_is_available_with_used_port(self):
        os.system("python -m SimpleHTTPServer 12005 &")
        time.sleep(2)
        assert self.test_server.port_is_available(12005) is False
        kill_process(12005)

    def test_check_avx(self):
        assert self.test_server.check_avx() is True

    # def test_check_local_bin_without_defined(self):
    #     del os.environ["SERVING_BIN"]
    #     self.test_server.check_local_bin()
    #     assert self.test_server.use_local_bin is False
    #
    # def test_check_local_bin_with_defined(self):
    #     os.environ["SERVING_BIN"] = "/home"
    #     self.test_server.check_local_bin()
    #     assert self.test_server.use_local_bin is True
    #     assert self.test_server.bin_path == os.environ["SERVING_BIN"]
    #     del os.environ["SERVING_BIN"]

    def test_get_fetch_list(self):
        assert self.test_server.get_fetch_list() == ['price']

    def test_prepare_engine(self):
        self.test_server._prepare_engine(self.test_server.model_config_paths, "cpu", False)
        model_toolkit_conf = ['engines', '{', 'name:', '"general_infer_0"', 'type:', '"PADDLE_INFER"',
                              'reloadable_meta:',
                              f'"{self.model_dir}/fluid_time_file"',
                              'reloadable_type:', '"timestamp_ne"', 'model_dir:',
                              f'"{self.model_dir}"',
                              'gpu_ids:', '-1', 'runtime_thread_num:', '0', 'batch_infer_size:', '32',
                              'enable_batch_align:', '1', 'enable_memory_optimization:', 'false',
                              'enable_ir_optimization:', 'false', 'use_trt:', 'false', 'use_lite:', 'false', 'use_xpu:',
                              'false', 'use_gpu:', 'false', 'combined_model:', 'false', 'gpu_multi_stream:', 'false',
                              '}']
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

    def test_prepare_server(self):
        self.test_server.prepare_server("workdir", 9696, "gpu", False)
        assert os.path.isfile(self.dir + "/workdir/general_infer_0/fluid_time_file") is True
        assert os.system(f"grep -r services {self.dir}/workdir/infer_service.prototxt") == 0
        assert os.system(f"grep -r workflows {self.dir}/workdir/workflow.prototxt") == 0
        assert os.system(f"grep -r model_toolkit_file {self.dir}/workdir/resource.prototxt") == 0
        assert os.system(f"grep -r engines {self.dir}/workdir/general_infer_0/model_toolkit.prototxt") == 0

    def test_run_server_with_cpu(self):
        self.test_server.prepare_server("workdir", 9696, "cpu")
        p = Process(target=self.test_server.run_server)
        p.start()
        os.system("sleep 5")

        price = self.predict()
        assert price == np.array([[18.901152]], dtype=np.float32)

        kill_process(9696, 1)

    def test_run_server_with_gpu(self):
        self.test_server.set_gpuid("0,1")
        os.system("netstat -nlp")
        os.system("ps -ef")
        self.test_server.prepare_server("workdir_0", 9696, "gpu")
        p = Process(target=self.test_server.run_server)
        p.start()
        os.system("sleep 10")

        price = self.predict()
        assert price == np.array([[18.901152]], dtype=np.float32)

        kill_process(9696, 2)


if __name__ == '__main__':
    # test_load_model_config()
    # test_port_is_available_with_used_port()
    # pytest.main(["-sv", "test_server.py"])
    # TestServer().test_get_fetch_list()
    ts = TestMultiLangServer()
    ts.setup_method()
    ts.test_run_server()
    pass
