# todo fetch取值错误示例 + 没有lod信息无法调用postprocess生成带框图片
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


class TestTTFNet(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="detection/ttfnet_darknet53_1x_coco", example_path="C++/PaddleDetection/ttfnet_darknet53_1x_coco", model_dir="serving_server",
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
            DetectionResize((512, 512), False, interpolation=cv2.INTER_LINEAR),
            DetectionNormalize([123.675, 116.28, 103.53], [58.395, 57.12, 57.375], False),
            DetectionTranspose((2, 0, 1))
        ])
        filename = "000000570688.jpg"
        im, im_info = preprocess(filename)
        print("im_info:", im_info)
        im = im[np.newaxis, :]
        input_dict = {}
        input_dict["image"] = im.astype("float32")
        input_dict["im_shape"] = np.array(list(im.shape[2:])).reshape(-1).astype("float32")
        input_dict["scale_factor"] = im_info['scale_factor'][np.newaxis, :]

        pd_config = paddle_infer.Config("serving_server/model.pdmodel", "serving_server/model.pdiparams")
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
        print(self.truth_val, self.truth_val["save_infer_model/scale_0.tmp_1"].shape,
              self.truth_val["save_infer_model/scale_1.tmp_1"].shape)

        # 输出预测库结果，框位置正确
        postprocess = RCNNPostprocess("label_list.txt", "output_infer")
        output_data_dict["save_infer_model/scale_0.tmp_1.lod"] = np.array([0, 100], dtype="int32")
        dict_ = copy.deepcopy(output_data_dict)
        del dict_["save_infer_model/scale_1.tmp_1"]
        dict_["image"] = filename
        postprocess(dict_)

    def predict_brpc(self, batch_size=1):
        preprocess = DetectionSequential([
            DetectionFile2Image(),
            DetectionResize((512, 512), False, interpolation=cv2.INTER_LINEAR),
            DetectionNormalize([123.675, 116.28, 103.53], [58.395, 57.12, 57.375], False),
            DetectionTranspose((2, 0, 1))
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
        print(fetch_map, fetch_map["save_infer_model/scale_0.tmp_1"].shape, fetch_map["save_infer_model/scale_1.tmp_1"].shape)
        dict_ = copy.deepcopy(fetch_map)
        dict_["image"] = filename
        dict_["save_infer_model/scale_0.tmp_1.lod"] = np.array([0, 100], dtype=np.int32)
        del dict_["save_infer_model/scale_1.tmp_1"]
        # TODO 未生成lod信息，暂不输出图片
        postprocess(dict_)
        return fetch_map

    def test_gpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model serving_server --port 9494 --gpu_ids 0",
            sleep=6,
        )

        # 2.resource check
        assert count_process_num_on_port(9494) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is False

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU")

        # 4.predict
        result_data = self.predict_brpc(batch_size=1)
        # TODO 未生成lod信息
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1)

        # 5.release
        kill_process(9494, 2)
