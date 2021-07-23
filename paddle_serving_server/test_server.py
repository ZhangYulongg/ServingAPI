import os
import pytest
import time
import collections
from multiprocessing import Process
import numpy as np

from paddle_serving_server.server import Server, MultiLangServer
import paddle_serving_server as serving
from paddle_serving_client import Client, MultiLangClient

from util import kill_process, check_gpu_memory


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
        p.start()
        os.system("sleep 10")

        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True

        price = self.predict(9697)
        assert price == np.array([[18.901152]], dtype=np.float32)

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
        # check workflow_conf (already in test_dag.py)
        # check general_infer_0 op model_conf (feed_var and fetch_var)
        # feed_var
        feed_var = self.test_server.model_conf["general_infer_0"].feed_var
        assert feed_var[0].name == "x"
        assert feed_var[0].alias_name == "x"
        assert feed_var[0].is_lod_tensor is False
        assert feed_var[0].feed_type == 1
        assert feed_var[0].shape == [13]
        # fetch_var
        fetch_var = self.test_server.model_conf["general_infer_0"].fetch_var
        assert fetch_var[0].name == "fc_0.tmp_1"
        assert fetch_var[0].alias_name == "price"
        assert fetch_var[0].is_lod_tensor is False
        assert fetch_var[0].fetch_type == 1
        assert fetch_var[0].shape == [1]
        # check model_config_paths and server config filename
        assert self.test_server.model_config_paths["general_infer_0"] == self.model_dir
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

    def test_get_fetch_list(self):
        assert self.test_server.get_fetch_list() == ['price']

    def test_prepare_engine_with_async_mode(self):
        # 生成bRPC server配置信息(model_toolkit_conf)
        # check model_toolkit_conf
        self.test_server.set_op_num(4)
        self.test_server.set_op_max_batch(64)
        self.test_server.set_gpuid(["0,1"])
        self.test_server.set_gpu_multi_stream()
        self.test_server._prepare_engine(self.test_server.model_config_paths, "gpu", False)
        model_engine_0 = self.test_server.model_toolkit_conf[0].engines[0]

        assert model_engine_0.name == "general_infer_0"
        assert model_engine_0.type == "PADDLE_INFER"
        assert model_engine_0.reloadable_meta == f"{self.model_dir}/fluid_time_file"
        assert model_engine_0.reloadable_type == "timestamp_ne"
        assert model_engine_0.model_dir == self.model_dir
        assert model_engine_0.gpu_ids == [0, 1]
        assert model_engine_0.runtime_thread_num == 4
        assert model_engine_0.batch_infer_size == 64
        assert model_engine_0.enable_batch_align == 1
        assert model_engine_0.enable_memory_optimization is False
        assert model_engine_0.enable_ir_optimization is False
        assert model_engine_0.use_trt is False
        assert model_engine_0.use_lite is False
        assert model_engine_0.use_xpu is False
        assert model_engine_0.use_gpu is True
        assert model_engine_0.combined_model is False
        assert model_engine_0.gpu_multi_stream is True

    def test_prepare_infer_service(self):
        # check infer_service_conf
        self.test_server._prepare_infer_service(9696)
        infer_service_conf = self.test_server.infer_service_conf

        assert infer_service_conf.port == 9696
        assert infer_service_conf.services[0].name == "GeneralModelService"
        assert infer_service_conf.services[0].workflows == ["workflow1"]

    def test_prepare_resource(self):
        # 生成模型feed_var,fetch_var配置文件(general_model.prototxt)，设置resource_conf属性
        # check resource_conf
        self.test_server._prepare_resource("workdir_9696", None)
        resource_conf = self.test_server.resource_conf
        assert resource_conf.model_toolkit_path == ["workdir_9696"]
        assert resource_conf.model_toolkit_file == ["general_infer_0/model_toolkit.prototxt"]
        assert resource_conf.general_model_path == ["workdir_9696"]
        assert resource_conf.general_model_file == ["general_infer_0/general_model.prototxt"]

    def test_prepare_server(self):
        # 生成bRPC server各种配置文件
        self.test_server.prepare_server("workdir_9696", 9696, "gpu", False)
        assert os.path.isfile(f"{self.dir}/workdir_9696/general_infer_0/fluid_time_file") is True
        assert os.path.isfile(f"{self.dir}/workdir_9696/infer_service.prototxt") is True
        assert os.path.isfile(f"{self.dir}/workdir_9696/workflow.prototxt") is True
        assert os.path.isfile(f"{self.dir}/workdir_9696/resource.prototxt") is True
        assert os.path.isfile(f"{self.dir}/workdir_9696/general_infer_0/model_toolkit.prototxt") is True

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

        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True

        price = self.predict()
        assert price == np.array([[18.901152]], dtype=np.float32)

        kill_process(9696, 2)


if __name__ == '__main__':
    # test_load_model_config()
    # test_port_is_available_with_used_port()
    # pytest.main(["-sv", "test_server.py"])
    # TestServer().test_get_fetch_list()
    ts = TestServer()
    ts.setup_method()
    ts.test_prepare_server()
    pass
