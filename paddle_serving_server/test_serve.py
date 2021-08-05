import argparse
import os
import pytest
from multiprocessing import Process
import numpy as np
import pynvml

from paddle_serving_server.serve import format_gpu_to_strlist
from paddle_serving_server.serve import is_gpu_mode
from paddle_serving_server.serve import start_gpu_card_model
from paddle_serving_client import Client, MultiLangClient
from paddle_serving_app.reader import Sequential, File2Image, Resize, CenterCrop
from paddle_serving_app.reader import RGB2BGR, Transpose, Div, Normalize

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

    def predict(self, batch=False, batch_size=1):
        client = Client()
        client.load_client_config(self.dir + "/resnet_v2_50_imagenet_client/serving_client_conf.prototxt")
        client.connect(["127.0.0.1:9696"])

        seq = Sequential([
            File2Image(), Resize(256), CenterCrop(224), RGB2BGR(), Transpose((2, 0, 1)),
            Div(255), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True)
        ])
        image_file = "daisy.jpg"
        img = seq(image_file)

        if batch:
            img_batch = img[np.newaxis, :]
            img_batch = np.repeat(img_batch, repeats=batch_size, axis=0)
            fetch_map = client.predict(feed={"image": img_batch}, fetch=["score"], batch=batch)
        else:
            fetch_map = client.predict(feed={"image": img}, fetch=["score"], batch=False)

        result_class = np.argmax(fetch_map["score"], axis=1)
        result_prob = np.max(fetch_map["score"], axis=1)
        print("fetch_map:", fetch_map)
        print("class:", result_class)
        print("prob:", result_prob)
        return result_class.tolist(), result_prob.tolist()

    def setup_method(self):
        dir = os.path.dirname(os.path.abspath(__file__))
        self.dir = dir
        self.model_dir = dir + "/resnet_v2_50_imagenet_model"

    def teardown_method(self):
        pass

    @pytest.mark.api_serverServe_formatGpuToStrList_parameters
    def test_format_gpu_to_strlist_with_int(self):
        assert format_gpu_to_strlist(2) == ["2"]

    @pytest.mark.api_serverServe_formatGpuToStrList_parameters
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

    @pytest.mark.api_serverServe_isGpuMode_parameters
    def test_is_gpu_mode(self):
        assert is_gpu_mode(["-1"]) is False
        assert is_gpu_mode(["0,1"]) is True

    @pytest.mark.api_serverServe_startGpuCardModel_exception
    def test_start_gpu_card_model_without_model(self):
        args = self.default_args()
        args.model = ""
        with pytest.raises(SystemExit) as e:
            start_gpu_card_model(gpu_mode=False, port=args.port, args=args)
        assert str(e.value) == "-1"

    @pytest.mark.api_serverServe_startGpuCardModel_parameters
    def test_start_gpu_card_model_with_single_model_cpu(self):
        args = self.default_args()
        args.model = [self.model_dir]
        args.port = 9696

        p = Process(target=start_gpu_card_model, kwargs={"gpu_mode": False, "port": args.port, "args": args})
        p.start()
        os.system("sleep 5")

        assert check_gpu_memory(0) is False

        # batch = False
        brcp_class, brpc_prob = self.predict(batch=False)
        print(brcp_class, brpc_prob)
        assert brcp_class == [985]
        assert brpc_prob == [0.9341399073600769]

        # batch_size = 2
        brcp_class, brpc_prob = self.predict(batch=True, batch_size=2)
        print(brcp_class, brpc_prob)
        assert brcp_class == [985, 985]
        assert brpc_prob == [0.9341403245925903, 0.9341403245925903]

        kill_process(9696, 1)

    @pytest.mark.api_serverServe_startGpuCardModel_parameters
    def test_start_gpu_card_model_with_single_model_gpu(self):
        args = self.default_args()
        args.model = [self.model_dir]
        args.port = 9696
        args.gpu_ids = ["0,1"]

        p = Process(target=start_gpu_card_model, kwargs={"gpu_mode": True, "port": args.port, "args": args})
        p.start()
        os.system("sleep 10")
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True

        # batch = False
        brcp_class, brpc_prob = self.predict(batch=False)
        print(brcp_class, brpc_prob)
        assert brcp_class == [985]
        assert brpc_prob == [0.9341405034065247]

        # batch_size = 2
        brcp_class, brpc_prob = self.predict(batch=True, batch_size=2)
        print(brcp_class, brpc_prob)
        assert brcp_class == [985, 985]
        assert brpc_prob == [0.9341405034065247, 0.9341405034065247]

        kill_process(9696, 3)

    @pytest.mark.api_serverServe_startGpuCardModel_parameters
    def test_start_gpu_card_model_with_two_models_gpu(self):
        args = self.default_args()
        args.model = [self.model_dir, self.model_dir]
        args.port = 9696
        args.gpu_ids = ["0", "1"]

        p = Process(target=start_gpu_card_model, kwargs={"gpu_mode": True, "port": args.port, "args": args})
        p.start()
        os.system("sleep 10")
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True

        kill_process(9696, 3)

  
if __name__ == '__main__':  
    ts = TestServe()
    ts.setup_method()
    ts.test_start_gpu_card_model_with_single_model_gpu()
    # ts.test_get_key()
    # print(format_gpu_to_strlist(["0,-1"]))
    # print(format_gpu_to_strlist(""))
    pass
