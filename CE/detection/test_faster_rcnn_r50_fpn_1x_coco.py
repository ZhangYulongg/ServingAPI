import os
import subprocess
import numpy as np
import copy
import cv2
import sys

from paddle_serving_client import Client, HttpClient
from paddle_serving_app.reader import SegPostprocess
from paddle_serving_app.reader import *
import paddle.inference as paddle_infer

sys.path.append("../")
from util import *


class TestFasterRCNN(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="detection/faster_rcnn_r50_fpn_1x_coco", example_path="C++/PaddleDetection/faster_rcnn_r50_fpn_1x_coco", model_dir="serving_server",
                                   client_dir="serving_client")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util

    def teardown_method(self):
        print_log(["stderr.log", "stdout.log",
                   "log/serving.ERROR", "PipelineServingLogs/pipeline.log"], iden="after predict")
        kill_process(9494)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        preprocess = DetectionSequential([
            DetectionFile2Image(),
            DetectionNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True),
            DetectionResize((800, 1333), True, interpolation=cv2.INTER_LINEAR),
            DetectionTranspose((2, 0, 1)),
            DetectionPadStride(128)
        ])
        file_name = "000000570688.jpg"
        im, im_info = preprocess(file_name)
        print("im_info:", im_info)
        im = im[np.newaxis, :]
        input_dict = {}
        input_dict["im_shape"] = np.array(list(im.shape[2:])).astype("float32")[np.newaxis, :]
        input_dict["image"] = im.astype("float32")
        input_dict["scale_factor"] = im_info['scale_factor'][np.newaxis, :]

        pd_config = paddle_infer.Config("serving_server/__model__", "serving_server/__params__")
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
        output_data_dict["save_infer_model/scale_0.tmp_1"] = output_data_dict["save_infer_model/scale_0.tmp_0"]
        output_data_dict["save_infer_model/scale_1.tmp_1"] = output_data_dict["save_infer_model/scale_1.tmp_0"]
        del output_data_dict["save_infer_model/scale_0.tmp_0"], output_data_dict["save_infer_model/scale_1.tmp_0"]
        self.truth_val = output_data_dict
        print(self.truth_val, self.truth_val["save_infer_model/scale_0.tmp_1"].shape, self.truth_val["save_infer_model/scale_1.tmp_1"].shape)

    def predict_brpc(self, batch_size=1):
        preprocess = DetectionSequential([
            DetectionFile2Image(),
            DetectionNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True),
            DetectionResize((800, 1333), True, interpolation=cv2.INTER_LINEAR),
            DetectionTranspose((2, 0, 1)),
            DetectionPadStride(128)
        ])
        postprocess = RCNNPostprocess("label_list.txt", "output")
        filename = "000000570688.jpg"
        im, im_info = preprocess(filename)
        print("im_info:", im_info)

        fetch = ["save_infer_model/scale_0.tmp_1", "save_infer_model/scale_1.tmp_1"]
        endpoint_list = ['127.0.0.1:9494']

        client = Client()
        client.load_client_config(self.serving_util.client_config)
        client.connect(endpoint_list)

        fetch_map = client.predict(
            feed={
                "image": im,
                "im_shape": np.array(list(im.shape[1:])).reshape(-1),
                "scale_factor": im_info['scale_factor'],
            },
            fetch=fetch,
            batch=False)
        print(fetch_map)
        dict_ = copy.deepcopy(fetch_map)
        del dict_["save_infer_model/scale_1.tmp_1"]
        dict_["image"] = filename
        postprocess(dict_)
        return fetch_map

    def test_gpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model serving_server --port 9494 --gpu_ids 1",
            sleep=8,
        )

        # 2.resource check
        assert count_process_num_on_port(9494) == 1
        assert check_gpu_memory(0) is False
        assert check_gpu_memory(1) is True

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU")

        # 4.predict
        result_data = self.predict_brpc(batch_size=1)
        # 删除lod信息
        del result_data["save_infer_model/scale_0.tmp_1.lod"]
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)

        # 5.release
        kill_process(9494, 2)


if __name__ == '__main__':
    sss = TestFasterRCNN()
    sss.get_truth_val_by_inference()