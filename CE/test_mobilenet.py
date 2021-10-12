import os
import subprocess
import numpy as np
import copy
import cv2

from paddle_serving_client import Client, HttpClient
from paddle_serving_app.reader import Sequential, URL2Image, Resize, File2Image
from paddle_serving_app.reader import CenterCrop, RGB2BGR, Transpose, Div, Normalize
import paddle.inference as paddle_infer

from util import *


class TestMobilenet(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="mobilenet", example_path="mobilenet", model_dir="mobilenet_v2_imagenet_model",
                                   client_dir="mobilenet_v2_imagenet_client")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util

    def teardown_method(self):
        print_log(["stderr.log", "stdout.log",
                   "log/serving.ERROR", "PipelineServingLogs/pipeline.log"], iden="after predict")
        kill_process(9393)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        seq = Sequential([
            File2Image(), Resize(256), CenterCrop(224), RGB2BGR(), Transpose((2, 0, 1)),
            Div(255), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True)
        ])
        image_file = "daisy.jpg"
        img = seq(image_file)[np.newaxis, :]
        input_dict = {
            "@HUB_mobilenet_v2_imagenet@image": img.astype("float32")
        }

        pd_config = paddle_infer.Config("mobilenet_v2_imagenet_model")
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
        output_data_dict["feature_map"] = output_data_dict["@HUB_mobilenet_v2_imagenet@pool2d_0.tmp_0"]
        del output_data_dict["@HUB_mobilenet_v2_imagenet@pool2d_0.tmp_0"]
        self.truth_val = output_data_dict
        print(self.truth_val, self.truth_val["feature_map"].shape)

    def predict_brpc(self, batch_size=1):
        seq = Sequential([
            File2Image(), Resize(256), CenterCrop(224), RGB2BGR(), Transpose((2, 0, 1)),
            Div(255), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True)
        ])
        image_file = "daisy.jpg"
        img = seq(image_file)
        feed_dict = {
            "image": img
        }

        fetch = ["feature_map"]
        endpoint_list = ["127.0.0.1:9393"]

        client = Client()
        client.load_client_config(self.serving_util.client_config)
        client.connect(endpoint_list)

        fetch_map = client.predict(feed=feed_dict, fetch=fetch, batch=False)
        print(fetch_map)
        return fetch_map

    def test_gpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model mobilenet_v2_imagenet_model --port 9393 --gpu_ids 0",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9393) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is False

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU", filename="stderr.log")

        # 4.predict
        result_data = self.predict_brpc(batch_size=1)
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)

        # 5.release
        kill_process(9393, 2)


if __name__ == '__main__':
    sss = TestMobilenet()
    sss.get_truth_val_by_inference()


