import os
import subprocess
import numpy as np
import copy
import cv2
import requests
import json
import sys

from paddle_serving_client import Client, HttpClient
from paddle_serving_client.utils import MultiThreadRunner
from paddle_serving_client.utils import benchmark_args, show_latency
from paddle_serving_app.reader import OCRReader
from paddle_serving_app.reader import Sequential, ResizeByFactor
from paddle_serving_app.reader import Div, Normalize, Transpose, File2Image
from paddle_serving_app.reader import DBPostProcess, FilterBoxes, GetRotateCropImage, SortedBoxes
import paddle.inference as paddle_infer

sys.path.append("../")
from util import *


def single_func(idx, resource):
    total_number = 0
    latency_list = []

    client = Client()
    client.load_client_config(["ocr_det_concat_client", "ocr_rec_client"])
    client.connect(resource["endpoint"])
    start = time.time()

    while True:
        l_start = time.time()
        result = client.predict(
            feed={"x": resource["feed_data"]},
            fetch=["save_infer_model/scale_0.tmp_1"],
            batch=True)
        l_end = time.time()
        # o_start = time.time()
        latency_list.append(l_end * 1000 - l_start * 1000)
        total_number = total_number + 1
        # o_end = time.time()
        if time.time() - start > 20:
            break

    end = time.time()
    return [[end - start], latency_list, [total_number]]


def run_cmd(cmd):
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        shell=True)
    out, err = process.communicate()
    return out, process.returncode


def parse_prototxt(file):
    with open(file, "r") as f:
        lines = [i.strip().split(":") for i in f.readlines()]

    engines = {}
    for i in lines:
        if len(i) > 1:
            if i[0] in engines:
                engines[i[0]].append(i[1].strip())
            else:
                engines[i[0]] = [i[1].strip()]
    return engines


class TestOCR(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="ocr_pipe", example_path="C++/PaddleOCR/ocr", model_dir="ocr_det_model",
                                   client_dir="ocr_det_client")
        serving_util.check_model_data_exist()
        self.serving_util = serving_util
        # 检测框处理funcs
        self.filter_func = FilterBoxes(10, 10)
        self.post_func = DBPostProcess({
            "thresh": 0.3,
            "box_thresh": 0.5,
            "max_candidates": 1000,
            "unclip_ratio": 1.5,
            "min_size": 3
        })
        self.ocr_reader = OCRReader()
        self.get_truth_val_by_inference(self)

    def teardown_method(self):
        print_log(["stderr.log", "stdout.log",
                   "log/serving.ERROR", "PipelineServingLogs/pipeline.log"], iden="after predict")
        kill_process(9293)
        self.serving_util.release(keywords="server.py")

    def get_truth_val_by_inference(self):
        seq = Sequential([
            ResizeByFactor(32, 960), Div(255),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose((2, 0, 1))
        ])
        filename = "imgs/1.jpg"
        with open(filename, "rb") as f:
            image_data = f.read()
        image = cv2_to_base64(image_data)
        data = base64.b64decode(image.encode("utf8"))
        data = np.fromstring(data, np.uint8)
        im = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # 图片原始h，w
        ori_h, ori_w, _ = im.shape
        det_img = seq(im)
        # 预处理后h，w
        _, new_h, new_w = det_img.shape

        det_img = det_img[np.newaxis, :]
        input_dict = {}
        input_dict["x"] = det_img.astype("float32")

        pd_config = paddle_infer.Config("ocr_det_model/inference.pdmodel", "ocr_det_model/inference.pdiparams")
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
        # output_data_dict["concat_1.tmp_0"] = output_data_dict["save_infer_model/scale_0.tmp_0"]
        # del output_data_dict["save_infer_model/scale_0.tmp_0"]
        self.truth_val_det = output_data_dict
        print(self.truth_val_det, self.truth_val_det["save_infer_model/scale_0.tmp_1"].shape)

        # Rec result
        sorted_boxes = SortedBoxes()
        get_rotate_crop_image = GetRotateCropImage()
        ratio_list = [float(new_h) / ori_h, float(new_w) / ori_w]
        dt_boxes_list = self.post_func(output_data_dict["save_infer_model/scale_0.tmp_1"], [ratio_list])
        dt_boxes = self.filter_func(dt_boxes_list[0], [ori_h, ori_w])
        dt_boxes = sorted_boxes(dt_boxes)

        feed_list = []
        img_list = []
        max_wh_ratio = 0
        for i, dtbox in enumerate(dt_boxes):
            boximg = get_rotate_crop_image(im, dt_boxes[i])
            img_list.append(boximg)
            h, w = boximg.shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        for img in img_list:
            norm_img = self.ocr_reader.resize_norm_img(img, max_wh_ratio)
            feed_list.append(norm_img[np.newaxis, :])
        input_dict = {"x": np.concatenate(feed_list, axis=0)}

        pd_config = paddle_infer.Config("ocr_rec_model/inference.pdmodel", "ocr_rec_model/inference.pdiparams")
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
        # output_data_dict["ctc_greedy_decoder_0.tmp_0"] = output_data_dict["save_infer_model/scale_0.tmp_0"]
        # output_data_dict["softmax_0.tmp_0"] = output_data_dict["save_infer_model/scale_1.tmp_0"]
        # del output_data_dict["save_infer_model/scale_0.tmp_0"], output_data_dict["save_infer_model/scale_1.tmp_0"]
        self.truth_val_rec = output_data_dict
        print(self.truth_val_rec, self.truth_val_rec["save_infer_model/scale_0.tmp_1"].shape)

        # 后处理转文字
        # dict_ = copy.deepcopy(output_data_dict)
        # dict_["ctc_greedy_decoder_0.tmp_0.lod"] = np.array([0, 22], dtype="int32")
        # dict_["softmax_0.tmp_0.lod"] = np.array([0, 22], dtype="int32")
        # rec_result = self.ocr_reader.postprocess(dict_, with_score=True)
        # res_lst = []
        # for res in rec_result:
        #     res_lst.append(res[0])
        # res = {"res": res_lst}
        # print(res)

    def predict_brpc(self, batch_size=1):
        # 1.prepare feed_data
        filename = "imgs/1.jpg"
        with open(filename, "rb") as f:
            image_data = f.read()
        image = cv2_to_base64(image_data)
        feed_dict = {"x": image}

        # 2.init client
        client = Client()
        client.load_client_config(["ocr_det_concat_client", "ocr_rec_client"])
        client.connect(["127.0.0.1:9293"])

        # 3.predict for fetch_map
        fetch_map = client.predict(feed=feed_dict, fetch=["save_infer_model/scale_0.tmp_1"], batch=True)
        print(fetch_map)
        # 后处理转文字
        # del fetch_map["softmax_0.tmp_0"]
        rec_result = self.ocr_reader.postprocess_ocrv2(fetch_map, with_score=False)
        res_lst = []
        for res in rec_result:
            res_lst.append(res[0])
        res = {"res": res_lst}
        print(res)
        return fetch_map

    def gpu_cpp_async_concurrent(self):
        # 效率云机器原因偶hang
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model ocr_det_model ocr_rec_model --op GeneralDetectionOp GeneralInferOp --thread 4 --runtime_thread_num 2 2 --batch_infer_size 2 --gpu_ids 1 --port 9293",
            sleep=17,
        )

        # 2.resource check
        assert count_process_num_on_port(9293) == 1  # web Server
        assert check_gpu_memory(0) is False
        assert check_gpu_memory(1) is True

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU")
        check_keywords_in_server_log("BSF thread init done", "log/serving.INFO")
        check_keywords_in_server_log("runtime_thread_num: 2",
                                     "workdir_9293/GeneralDetectionOp_0/model_toolkit.prototxt")
        check_keywords_in_server_log("runtime_thread_num: 2", "workdir_9293/GeneralInferOp_0/model_toolkit.prototxt")
        detection_op = parse_prototxt("workdir_9293/GeneralDetectionOp_0/model_toolkit.prototxt")
        infer_op = parse_prototxt("workdir_9293/GeneralInferOp_0/model_toolkit.prototxt")
        assert detection_op["gpu_ids"] == ["1"]
        assert infer_op["gpu_ids"] == ["1"]

        # 内存泄露检测
        out, _ = run_cmd("ps -ef | grep serving | grep -v grep | awk '{print $2}' | awk 'END {print}'")
        server_pid = out.decode().strip().split("\n")[-1]
        print("server_pid:", server_pid)
        out, _ = run_cmd("cat /proc/" + server_pid + "/status | grep RSS | awk '{print $2}'")
        rss_start = int(out.decode().strip().split("\n")[-1])
        print("rss_start:", rss_start)

        # 4.predict by brpc 多client并发
        multi_thread_runner = MultiThreadRunner()
        endpoint_list = ["127.0.0.1:9293"]
        turns = 50
        thread_num = 30  # client并发数
        batch_size = 1
        # prepare feed_data
        image_file = "imgs/1.jpg"
        with open(image_file, 'rb') as file:
            image_data = file.read()
        image = cv2_to_base64(image_data)

        start = time.time()
        result = multi_thread_runner.run(
            single_func, thread_num, {"endpoint": endpoint_list, "turns": turns, "feed_data": image}
        )
        end = time.time()
        total_cost = end - start
        total_number = 0
        avg_cost = 0
        for i in range(thread_num):
            avg_cost += result[0][i]
            total_number += result[2][i]
        avg_cost = avg_cost / thread_num

        print("total cost-include init: {}s".format(total_cost))
        print("each thread cost: {}s. ".format(avg_cost))
        print("qps: {}samples/s".format(batch_size * total_number / avg_cost))
        print("total count: {} ".format(total_number))
        show_latency(result[1])

        # 合batch检测
        check_keywords_in_server_log("Hit auto padding", "log/serving.INFO")
        # 内存泄露检测
        out, _ = run_cmd("cat /proc/" + server_pid + "/status | grep RSS | awk '{print $2}'")
        rss_after = int(out.decode().strip().split("\n")[-1])
        print("rss_after:", rss_after)
        print("RSS diff is:", rss_after - rss_start, "KB")
        assert rss_after - rss_start <= 3145728, f"Memory Leak!, RSS diff is {rss_after - rss_start} KB"

        # 5.release
        kill_process(9293, 2)

    def test_cpu_cpp(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model ocr_det_model ocr_rec_model --op GeneralDetectionOp GeneralInferOp --port 9293",
            sleep=10,
        )

        # 2.resource check
        assert count_process_num_on_port(9293) == 1  # web Server
        assert check_gpu_memory(0) is False

        # 3.keywords check

        # 4.predict by http
        # batch_size=1
        result = self.predict_brpc(batch_size=1)
        print(result["save_infer_model/scale_0.tmp_1"].shape)
        # 删除lod信息
        # assert list(result["ctc_greedy_decoder_0.tmp_0.lod"]) == [0, 13, 22]
        # del result["ctc_greedy_decoder_0.tmp_0.lod"], result["softmax_0.tmp_0.lod"], result["ctc_greedy_decoder_0.tmp_0"]
        # self.serving_util.check_result(result_data=result, truth_data=self.truth_val_rec, batch_size=1)

        # 5.release
        kill_process(9293)

    def test_gpu_cpp(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model ocr_det_model ocr_rec_model --op GeneralDetectionOp GeneralInferOp --gpu_ids 0 --port 9293",
            sleep=17,
        )

        # 2.resource check
        assert count_process_num_on_port(9293) == 1  # web Server
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is False

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU")

        # 4.predict by http
        # batch_size=1
        result = self.predict_brpc(batch_size=1)
        print(result["save_infer_model/scale_0.tmp_1"].shape)
        # 删除lod信息
        # assert list(result["ctc_greedy_decoder_0.tmp_0.lod"]) == [0, 13, 22]
        # del result["ctc_greedy_decoder_0.tmp_0.lod"], result["softmax_0.tmp_0.lod"]
        # self.serving_util.check_result(result_data=result, truth_data=self.truth_val_rec, batch_size=1)

        # 5.release
        kill_process(9293, 2)

    def test_gpu_cpp_async(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model ocr_det_model ocr_rec_model --op GeneralDetectionOp GeneralInferOp --runtime_thread_num 2 4 --batch_infer_size 16 --gpu_ids 0,1 1 --port 9293",
            sleep=17,
        )

        # 2.resource check
        assert count_process_num_on_port(9293) == 1  # web Server
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU")
        check_keywords_in_server_log("BSF thread init done", "log/serving.INFO")
        check_keywords_in_server_log("runtime_thread_num: 2", "workdir_9293/GeneralDetectionOp_0/model_toolkit.prototxt")
        check_keywords_in_server_log("runtime_thread_num: 4", "workdir_9293/GeneralInferOp_0/model_toolkit.prototxt")
        detection_op = parse_prototxt("workdir_9293/GeneralDetectionOp_0/model_toolkit.prototxt")
        infer_op = parse_prototxt("workdir_9293/GeneralInferOp_0/model_toolkit.prototxt")
        assert detection_op["gpu_ids"] == ["0", "1"]
        assert infer_op["gpu_ids"] == ["1"]

        # 4.predict by http
        # batch_size=1
        result = self.predict_brpc(batch_size=1)
        print(result["save_infer_model/scale_0.tmp_1"].shape)
        # 删除lod信息
        # assert list(result["ctc_greedy_decoder_0.tmp_0.lod"]) == [0, 13, 22]
        # del result["ctc_greedy_decoder_0.tmp_0.lod"], result["softmax_0.tmp_0.lod"]
        # self.serving_util.check_result(result_data=result, truth_data=self.truth_val_rec, batch_size=1)

        # 5.release
        kill_process(9293, 2)

    def test_gpu_cpp_async_trt_fp32(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve "
                f"--model ocr_det_model ocr_rec_model "
                f"--op GeneralDetectionOp GeneralInferOp "
                f"--port 9293 "
                f"--use_trt "
                f"--gpu_ids 0 "
                f"--min_subgraph_size 13 3 "
                f"--runtime_thread_num 1 1",
            sleep=90,
        )

        # 2.resource check
        assert count_process_num_on_port(9293) == 1  # web Server
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is False

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU")
        check_keywords_in_server_log("BSF thread init done", "log/serving.INFO")
        check_keywords_in_server_log("runtime_thread_num: 1",
                                     "workdir_9293/GeneralDetectionOp_0/model_toolkit.prototxt")
        check_keywords_in_server_log("runtime_thread_num: 1", "workdir_9293/GeneralInferOp_0/model_toolkit.prototxt")
        detection_op = parse_prototxt("workdir_9293/GeneralDetectionOp_0/model_toolkit.prototxt")
        infer_op = parse_prototxt("workdir_9293/GeneralInferOp_0/model_toolkit.prototxt")
        assert detection_op["gpu_ids"] == ["0"]
        assert infer_op["gpu_ids"] == ["0"]

        # 4.predict by http
        # batch_size=1
        result = self.predict_brpc(batch_size=1)
        print(result["save_infer_model/scale_0.tmp_1"].shape)
        result = self.predict_brpc(batch_size=1)
        print(result["save_infer_model/scale_0.tmp_1"].shape)
        # 删除lod信息
        # assert list(result["ctc_greedy_decoder_0.tmp_0.lod"]) == [0, 13, 22]
        # del result["ctc_greedy_decoder_0.tmp_0.lod"], result["softmax_0.tmp_0.lod"]
        # self.serving_util.check_result(result_data=result, truth_data=self.truth_val_rec, batch_size=1)

        # 5.release
        kill_process(9293, 2)

