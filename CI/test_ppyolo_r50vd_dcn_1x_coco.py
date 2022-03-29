import os
import subprocess
import numpy as np
import copy
import cv2
import sys
import time
import pytest

from paddle_serving_client import Client, HttpClient
from paddle_serving_client.io import inference_model_to_serving
from paddle_serving_app.reader import SegPostprocess
from paddle_serving_app.reader import *
import paddle.inference as paddle_infer

sys.path.append("../")
from util import *


ce_name = os.environ.get("ce_name") if os.environ.get("ce_name") else ""


def serving_encryption():
    inference_model_to_serving(
        dirname="./serving_server",
        params_filename="__params__",
        serving_server="encrypt_server",
        serving_client="encrypt_client",
        encryption=True)


class TestPPYOLO(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="detection/ppyolo_r50vd_dcn_1x_coco", example_path="C++/PaddleDetection/ppyolo_r50vd_dcn_1x_coco", model_dir="serving_server",
                                   client_dir="serving_client")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util
        serving_encryption()

    def teardown_method(self):
        print_log(["stderr.log", "stdout.log",
                   "log/serving.ERROR", "PipelineServingLogs/pipeline.log"], iden="after predict")
        kill_process(9494)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        preprocess = DetectionSequential([
            DetectionFile2Image(),
            DetectionNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True),
            DetectionResize((608, 608), False, interpolation=2),
            DetectionTranspose((2, 0, 1))
        ])
        filename = "000000570688.jpg"
        im, im_info = preprocess(filename)
        print("im_info:", im_info)
        im = im[np.newaxis, :]
        input_dict = {}
        input_dict["image"] = im.astype("float32")
        input_dict["im_shape"] = np.array(im.shape[2:]).astype("float32")[np.newaxis, :]
        input_dict["scale_factor"] = im_info['scale_factor'][np.newaxis, :]

        pd_config = paddle_infer.Config("serving_server/__model__", "serving_server/__params__")
        pd_config.disable_gpu()
        pd_config.switch_ir_optim(False)
        # TRT结果能够对齐
        # pd_config.enable_use_gpu(1000, 0)
        # pd_config.enable_tensorrt_engine(
        #     workspace_size=1 << 30,
        #     max_batch_size=1,
        #     min_subgraph_size=3,
        #     precision_mode=paddle_infer.PrecisionType.Float32,
        #     use_static=False,
        #     use_calib_mode=False,
        # )

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
        print(self.truth_val, self.truth_val["save_infer_model/scale_0.tmp_1"].shape, self.truth_val["save_infer_model/scale_1.tmp_0"].shape)

        # 输出预测库结果，框位置正确
        postprocess = RCNNPostprocess("label_list.txt", "output_infer")
        output_data_dict["save_infer_model/scale_0.tmp_1.lod"] = np.array([0, 100], dtype="int32")
        dict_ = copy.deepcopy(output_data_dict)
        del dict_["save_infer_model/scale_1.tmp_0"]
        dict_["image"] = filename
        postprocess(dict_)

    def predict_brpc(self, batch_size=1):
        preprocess = DetectionSequential([
            DetectionFile2Image(),
            DetectionNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True),
            DetectionResize((608, 608), False, interpolation=2),
            DetectionTranspose((2, 0, 1))
        ])
        postprocess = RCNNPostprocess("label_list.txt", "output")
        filename = "000000570688.jpg"
        im, im_info = preprocess(filename)

        fetch = ["save_infer_model/scale_0.tmp_1"]
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
        dict_["image"] = filename
        postprocess(dict_)
        return fetch_map

    def predict_brpc_encrypt(self, batch_size=1):
        preprocess = DetectionSequential([
            DetectionFile2Image(),
            DetectionNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True),
            DetectionResize((608, 608), False, interpolation=2),
            DetectionTranspose((2, 0, 1))
        ])
        postprocess = RCNNPostprocess("label_list.txt", "output_encrypt")
        filename = "000000570688.jpg"
        im, im_info = preprocess(filename)
        print("im_info:", im_info)

        fetch = ["save_infer_model/scale_0.tmp_0"]
        endpoint_list = ['127.0.0.1:9494']

        client = Client()
        client.load_client_config("encrypt_client/serving_client_conf.prototxt")
        client.use_key("./key")
        client.connect(endpoint_list, encryption=True)
        time.sleep(70)

        fetch_map = client.predict(
            feed={
                "image": im,
                "im_shape": np.array(list(im.shape[1:])).reshape(-1),
                "scale_factor": im_info['scale_factor'],
            },
            fetch=fetch,
            batch=False)
        # 对齐serving output
        fetch_map["save_infer_model/scale_0.tmp_1"] = fetch_map["save_infer_model/scale_0.tmp_0"]
        del fetch_map["save_infer_model/scale_0.tmp_0"]
        print(fetch_map, fetch_map["save_infer_model/scale_0.tmp_1"].shape)
        dict_ = copy.deepcopy(fetch_map)
        dict_["image"] = filename
        # TODO fetch_map中缺少lod信息，手动添加后画框
        dict_["save_infer_model/scale_0.tmp_1.lod"] = np.array([0, 100], dtype="int32")
        postprocess(dict_)
        return fetch_map

    def test_gpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model serving_server --port 9494 --gpu_ids 0",
            sleep=10,
        )

        # 2.resource check
        assert count_process_num_on_port(9494) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is False

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU")

        # 4.predict
        result_data = self.predict_brpc(batch_size=1)
        # 删除 lod信息
        del result_data["save_infer_model/scale_0.tmp_1.lod"]
        self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1, delta=1e-1)

        # 5.release
        kill_process(9494, 2)

    def test_gpu_trt_fp32(self):
        if ce_name and "p4" in ce_name:
            sleep_time = 130
        else:
            sleep_time = 70
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model serving_server --port 9494 --use_trt --gpu_ids 0 | tee log.txt",
            sleep=sleep_time,
        )

        # 2.resource check
        assert count_process_num_on_port(9494) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is False

        # 3.keywords check
        check_keywords_in_server_log("Prepare TRT engine")
        check_keywords_in_server_log("Sync params from CPU to GPU")

        # 4.predict
        result_data = self.predict_brpc(batch_size=1)
        # 删除 lod信息
        del result_data["save_infer_model/scale_0.tmp_1.lod"]
        # self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1, delta=1e-1)

        # 5.release
        kill_process(9494, 2)

    def test_gpu_trt_fp32_encrypt(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model encrypt_server --use_encryption_model --port 9494 --use_trt --gpu_ids 0 2>&1",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9494) == 1
        # 加密部署时，client带着key请求后才起brpc-server(默认端口12000, --port指定的为http server端口)

        # 3.keywords check

        # 4.predict
        result_data = self.predict_brpc_encrypt(batch_size=1)
        assert count_process_num_on_port(12000) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is False
        check_keywords_in_server_log("Prepare TRT engine", "stdout.log")
        check_keywords_in_server_log("Sync params from CPU to GPU", "stdout.log")
        # 删除 lod信息 加密部署的模型fetch_map没有lod信息
        # self.serving_util.check_result(result_data=result_data, truth_data=self.truth_val, batch_size=1, delta=1e-1)

        # 5.release
        kill_process(9494, 2)
