import os
import subprocess
import numpy as np
import copy
import cv2

from paddle_serving_client import Client, HttpClient
from paddle_serving_app.reader import CenterCrop, RGB2BGR, Transpose, Div, Normalize, RCNNPostprocess
from paddle_serving_app.reader import Sequential, File2Image, Resize, Transpose, BGR2RGB, SegPostprocess
import paddle.inference as paddle_infer

from util import *


class TestYOLOv4(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="yolov4", example_path="yolov4", model_dir="yolov4_model",
                                   client_dir="yolov4_client")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util

    def teardown_method(self):
        print("======================stderr.log after predict======================")
        os.system("cat stderr.log")
        print("======================stdout.log after predict======================")
        os.system("cat stdout.log")
        print("====================================================================")
        kill_process(9393)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        preprocess = Sequential([
            File2Image(), BGR2RGB(), Resize((608, 608), interpolation=cv2.INTER_LINEAR),
            Div(255.0), Transpose((2, 0, 1))
        ])
        filename = "000000570688.jpg"
        im = preprocess(filename)[np.newaxis, :]
        input_dict = {
            "image": im.astype("float32"),
            "im_size": np.array(im.shape[2:]).astype("int32")[np.newaxis, :]
        }

        pd_config = paddle_infer.Config("yolov4_model")
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
        # 对齐serving output
        self.truth_val = output_data_dict
        print(self.truth_val, self.truth_val["save_infer_model/scale_0.tmp_0"].shape)

        # postprocess = RCNNPostprocess("label_list.txt", "output", [608, 608])
        # output_data_dict["save_infer_model/scale_0.tmp_0.lod"] = np.array([ 0, 71], dtype="int32")
        # dict_ = copy.deepcopy(output_data_dict)
        # dict_["image"] = filename
        # postprocess(dict_)

    def predict_brpc(self, batch_size=1):
        # 1.prepare feed_data
        preprocess = Sequential([
            File2Image(), BGR2RGB(), Resize((608, 608), interpolation=cv2.INTER_LINEAR),
            Div(255.0), Transpose((2, 0, 1))
        ])
        postprocess = RCNNPostprocess("label_list.txt", "output", [608, 608])
        filename = "000000570688.jpg"
        im = preprocess(filename)
        feed_dict = {
            "image": im.astype("float32"),
            "im_size": np.array(im.shape[1:]).astype("int32")
        }

        # 2.init client
        fetch = ["save_infer_model/scale_0.tmp_0"]
        endpoint_list = ["127.0.0.1:9393"]
        client = Client()
        client.load_client_config(self.serving_util.client_config)
        client.connect(endpoint_list)

        # 3.predict for fetch_map
        fetch_map = client.predict(feed=feed_dict, fetch=fetch, batch=False)
        print(fetch_map)
        dict_ = copy.deepcopy(fetch_map)
        dict_["image"] = filename
        postprocess(dict_)
        return fetch_map

    def test_gpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model yolov4_model --port 9393 --gpu_ids 0",
            sleep=10,
        )

        # 2.resource check
        assert count_process_num_on_port(9393) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is False

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU", filename="stderr.log")

        # 4.predict
        result_data = self.predict_brpc(batch_size=1)
        # TODO cpu和gpu结果均有较大diff，待排查
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)

        # 5.release
        kill_process(9393, 2)


if __name__ == '__main__':
    sss = TestYOLOv4()
    sss.get_truth_val_by_inference()


