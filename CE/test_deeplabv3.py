import os
import subprocess
import numpy as np
import copy
import cv2

from paddle_serving_client import Client, HttpClient
from paddle_serving_app.reader import SegPostprocess
from paddle_serving_app.reader import *
import paddle.inference as paddle_infer

from util import *


class TestDeeplabv3(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="deeplabv3", example_path="deeplabv3", model_dir="deeplabv3_server",
                                   client_dir="deeplabv3_client")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util

    def teardown_method(self):
        print("======================stderr.log after predict======================")
        os.system("cat stderr.log")
        print("======================stdout.log after predict======================")
        os.system("cat stdout.log")
        print("====================================================================")
        kill_process(9494)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        preprocess = Sequential(
            [File2Image(), Resize(
                (512, 512), interpolation=cv2.INTER_LINEAR)])
        im = preprocess("N0060.jpg")[np.newaxis, :]
        print(im, im.shape)

        input_dict = {}
        input_dict["image"] = np.array(im).astype("float32")

        pd_config = paddle_infer.Config("deeplabv3_server")
        pd_config.disable_gpu()
        pd_config.switch_ir_optim(False)

        predictor = paddle_infer.create_predictor(pd_config)

        input_names = predictor.get_input_names()
        for i, input_name in enumerate(input_names):
            input_handle = predictor.get_input_handle(input_name)
            input_handle.copy_from_cpu(input_dict[input_name])

        predictor.run()

        output_data_dict = {}
        output_names = predictor.get_output_names()
        for _, output_data_name in enumerate(output_names):
            output_handle = predictor.get_output_handle(output_data_name)
            output_data = output_handle.copy_to_cpu()
            output_data_dict[output_data_name] = output_data
        # 对齐Serving output
        output_data_dict["output"] = output_data_dict["save_infer_model/scale_0.tmp_0"]
        del output_data_dict["save_infer_model/scale_0.tmp_0"]
        self.truth_val = output_data_dict
        # print(self.truth_val, self.truth_val["output"].shape)

    def predict_brpc(self, batch_size=1):
        preprocess = Sequential(
            [File2Image(), Resize(
                (512, 512), interpolation=cv2.INTER_LINEAR)])
        filename = "N0060.jpg"
        im = preprocess(filename)
        postprocess = SegPostprocess(2)

        fetch = ["output"]
        endpoint_list = ['127.0.0.1:9494']

        client = Client()
        client.load_client_config(self.serving_util.client_config)
        client.connect(endpoint_list)

        fetch_map = client.predict(feed={"image": im}, fetch=fetch)
        print(fetch_map)
        dict_ = copy.deepcopy(fetch_map)
        dict_["filename"] = filename
        postprocess(dict_)
        return fetch_map

    def test_gpu_multicard(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model deeplabv3_server --gpu_ids 0,1 --port 9494",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9494) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU")

        # 4.predict
        result_data = self.predict_brpc(batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)

        # 5.release
        kill_process(9494, 2)
