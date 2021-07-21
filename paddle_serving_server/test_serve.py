import argparse
import os
import pytest
from multiprocessing import Process
import numpy as np
import pynvml

from paddle_serving_server.serve import format_gpu_to_strlist
from paddle_serving_server.serve import is_gpu_mode
from paddle_serving_server.serve import start_gpu_card_model
from paddle_serving_server.serve import start_multi_card
from paddle_serving_server.serve import MainService
from paddle_serving_client import Client, MultiLangClient

from util import kill_process, check_gpu_memory


class TestServe(object):
    @staticmethod
    def default_args():
        parser = argparse.ArgumentParser()
        args = parser.parse_args([])
        args.thread = 2
        args.port = 9292
        args.device = "cpu"
        args.gpu_ids = [""]
        args.op_num = 0
        args.op_max_batch = 32
        args.model = [""]
        args.workdir = "workdir"
        args.name = "None"
        args.use_mkl = False
        args.precision = "fp32"
        args.use_calib = False
        args.mem_optim_off = False
        args.ir_optim = False
        args.max_body_size = 512 * 1024 * 1024
        args.use_encryption_model = False
        args.use_multilang = False
        args.use_trt = False
        args.use_lite = False
        args.use_xpu = False
        args.product_name = None
        args.container_id = None
        args.gpu_multi_stream = False
        return args

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

    def setup_method(self):
        dir = os.path.dirname(os.path.abspath(__file__))
        self.dir = dir
        self.model_dir = dir + "/uci_housing_model"

    def teardown_method(self):
        pass

    def test_format_gpu_to_strlist_with_int(self):
        assert format_gpu_to_strlist(2) == ["2"]

    def test_format_gpu_to_strlist_with_list(self):
        assert format_gpu_to_strlist(["3"]) == ["3"]
        assert format_gpu_to_strlist([""]) == ["-1"]
        assert format_gpu_to_strlist([]) == ["-1"]
        assert format_gpu_to_strlist([0, 1]) == ["0", "1"]
        assert format_gpu_to_strlist(["0,2", "1,3"]) == ["0,2", "1,3"]
        # with None
        assert format_gpu_to_strlist(None) == ["-1"]
        # with valid gpu id
        with pytest.raises(ValueError) as e:
            format_gpu_to_strlist(["1", "-2"])
        assert str(e.value) == "The input of gpuid error."
        with pytest.raises(ValueError) as e:
            format_gpu_to_strlist(["0,-1"])
        assert str(e.value) == "You can not use CPU and GPU in one model."

    def test_is_gpu_mode(self):
        assert is_gpu_mode(["-1"]) is False
        assert is_gpu_mode(["0,1"]) is True

    def test_start_gpu_card_model_without_model(self):
        args = self.default_args()
        args.model = ""
        with pytest.raises(SystemExit) as e:
            start_gpu_card_model(gpu_mode=False, port=args.port, args=args)
        assert str(e.value) == "-1"

    def test_start_gpu_card_model_with_single_model_cpu(self):
        args = self.default_args()
        args.model = [self.model_dir]
        args.port = 9696

        p = Process(target=start_gpu_card_model, kwargs={"gpu_mode": False, "port": args.port, "args": args})
        p.start()
        os.system("sleep 5")

        price = self.predict()
        print(price)

        kill_process(9696)

    def test_start_gpu_card_model_with_single_model_gpu(self):
        args = self.default_args()
        args.model = [self.model_dir]
        args.port = 9696
        args.gpu_ids = ["0,1"]

        p = Process(target=start_gpu_card_model, kwargs={"gpu_mode": True, "port": args.port, "args": args})
        p.start()
        os.system("sleep 7")
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True

        price = self.predict()
        print(price)

        kill_process(9696)

    def test_start_gpu_card_model_with_two_models_gpu(self):
        args = self.default_args()
        args.model = [self.model_dir, self.model_dir]
        args.port = 9696
        args.gpu_ids = ["0", "1"]

        p = Process(target=start_gpu_card_model, kwargs={"gpu_mode": True, "port": args.port, "args": args})
        p.start()
        os.system("sleep 7")
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True

        kill_process(9696)

  
if __name__ == '__main__':  
    # ts = TestMainService()
    # ts.setup_method()
    # ts.test_get_key()
    # print(format_gpu_to_strlist(["0,-1"]))
    # print(format_gpu_to_strlist(""))
    pass

