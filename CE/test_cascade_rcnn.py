import os
import subprocess
import numpy as np
import copy
import cv2

from paddle_serving_client import Client, HttpClient
from paddle_serving_app.reader import BlazeFacePostprocess
from paddle_serving_app.reader import *
import paddle.inference as paddle_infer

from util import *


class TestCascadeRCNN(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="cascade_rcnn", example_path="C++/PaddleDetection/cascade_rcnn", model_dir="serving_server",
                                   client_dir="serving_client")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util

    def teardown_method(self):
        print_log(["stderr.log", "stdout.log",
                   "log/serving.ERROR", "PipelineServingLogs/pipeline.log"], iden="after predict")
        kill_process(9292)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        preprocess = DetectionSequential([
            DetectionFile2Image(),
            DetectionResize((800, 1333), True, interpolation=2),
            DetectionNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True),
            DetectionTranspose((2, 0, 1)),
            DetectionPadStride(32)
        ])
        im, im_info = preprocess('000000570688.jpg')
        print("im_info:", im_info)
        im = im[np.newaxis, :]
        input_dict = {}
        input_dict["image"] = im.astype("float32")
        input_dict["im_shape"] = np.array(list(im.shape[2:])).reshape(-1).astype("float32")[np.newaxis, :]
        input_dict["scale_factor"] = im_info['scale_factor'][np.newaxis, :]

        pd_config = paddle_infer.Config("serving_server/__model__", "serving_server/__params__")
        pd_config.disable_gpu()
        pd_config.switch_ir_optim(True)

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
        del output_data_dict["save_infer_model/scale_0.tmp_0"]
        self.truth_val = output_data_dict
        print(self.truth_val, self.truth_val["save_infer_model/scale_0.tmp_1"].shape)

    def predict_brpc(self, batch_size=1):
        preprocess = DetectionSequential([
            DetectionFile2Image(),
            DetectionResize((800, 1333), True, interpolation=2),
            DetectionNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True),
            DetectionTranspose((2, 0, 1)),
            DetectionPadStride(32)
        ])
        postprocess = RCNNPostprocess("label_list.txt", "output")
        im, im_info = preprocess('000000570688.jpg')
        print("im_info:", im_info)

        fetch = ["save_infer_model/scale_0.tmp_1"]
        endpoint_list = ['127.0.0.1:9292']

        client = Client()
        client.load_client_config(self.serving_util.client_config)
        client.connect(endpoint_list)

        fetch_map = client.predict(
            feed={
                "image": im,
                "im_shape": np.array(list(im.shape[1:])),
                "scale_factor": im_info['scale_factor'],
            },
            fetch=["save_infer_model/scale_0.tmp_1"],
            batch=False)
        print(fetch_map)
        dict_ = copy.deepcopy(fetch_map)
        dict_["image"] = '000000570688.jpg'
        postprocess(dict_)
        return fetch_map

    def test_gpu_multicard(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model serving_server --port 9292 --gpu_ids 0,1",
            sleep=9,
        )

        # 2.resource check
        assert count_process_num_on_port(9292) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU")

        # 4.predict
        result_data = self.predict_brpc(batch_size=1)
        # 删除lod信息
        del result_data["save_infer_model/scale_0.tmp_1.lod"]
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1, delta=1e-2)

        # 5.release
        kill_process(9292, 2)
