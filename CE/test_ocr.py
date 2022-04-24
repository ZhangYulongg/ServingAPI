import os
import subprocess
import numpy as np
import copy
import cv2
import requests
import json

from paddle_serving_client import Client, HttpClient
from paddle_serving_app.reader import OCRReader
from paddle_serving_app.reader import Sequential, ResizeByFactor
from paddle_serving_app.reader import Div, Normalize, Transpose, File2Image
from paddle_serving_app.reader import DBPostProcess, FilterBoxes, GetRotateCropImage, SortedBoxes
import paddle.inference as paddle_infer

from util import *


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
        os.system("sed -i '96 i \ \ \ \ \ \ \ \ np.save(\"fetch_dict_rec\", fetch_map)' ocr_debugger_server.py")
        os.system("sed -i '60 i \ \ \ \ \ \ \ \ np.save(\"fetch_dict_det\", fetch_map)' det_web_server.py")
        os.system("sed -i '86 i \ \ \ \ \ \ \ \ np.save(\"fetch_dict_rec\", fetch_map)' ocr_web_server.py")

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

    def predict_http(self, batch_size=1):
        # 1.prepare feed_data
        filename = "imgs/1.jpg"
        with open(filename, "rb") as f:
            image_data = f.read()
        image = cv2_to_base64(image_data)
        feed_dict = {
            "feed": [{"x": image}],
            "fetch": ["res"],
        }

        # 2.predict for fetch_map
        url = "http://127.0.0.1:9292/ocr/prediction"
        headers = {"Content-type": "application/json"}
        r = requests.post(url=url, headers=headers, data=json.dumps(feed_dict))
        print(r.json())
        return r.json()["result"]

    def predict_http_rec(self, batch_size=1):
        # 1.prepare feed_data
        filename = "rec_img/ch_doc3.jpg"
        with open(filename, "rb") as f:
            image_data = f.read()
        image = cv2_to_base64(image_data)
        feed_dict = {
            "feed": [{"x": image}] * batch_size,
            "fetch": ["res"],
        }
        # 2.predict for fetch_map
        url = "http://127.0.0.1:9292/ocr/prediction"
        headers = {"Content-type": "application/json"}
        r = requests.post(url=url, headers=headers, data=json.dumps(feed_dict))
        print(r.json())
        return r.json()["result"]

    def test_cpu_web_service(self):
        # python -m pytest -sv 1.py::TestOCR::test_cpu_web_service
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model ocr_det_model --port 9293",
            sleep=5,
            err="deterr.log",
            out="detout.log",
        )
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} ocr_web_server.py cpu",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9293) == 1  # det Server
        assert count_process_num_on_port(9292) == 1  # web Server
        assert count_process_num_on_port(12000) == 1  # rec Server
        assert check_gpu_memory(0) is False

        # 3.keywords check

        # 4.predict by http
        # batch_size=1
        self.predict_http(batch_size=1)
        # 从npy文件读取
        rec_result = np.load("fetch_dict_rec.npy", allow_pickle=True).item()
        # 删除文件
        os.system("rm -rf fetch_dict_rec.npy")
        # # 删除lod信息
        # del rec_result["ctc_greedy_decoder_0.tmp_0.lod"], rec_result["softmax_0.tmp_0.lod"]
        self.serving_util.check_result(result_data=rec_result, truth_data=self.truth_val_rec, batch_size=1)

        # 5.release
        kill_process(9292)
        kill_process(9293)
        kill_process(12000)

    def test_gpu_web_service(self):
        # python -m pytest -sv 1.py::TestOCR::test_cpu_web_service
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model ocr_det_model --port 9293 --gpu_ids 0",
            sleep=8,
            err="deterr.log",
            out="detout.log",
        )
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} ocr_web_server.py gpu",
            sleep=8,
        )

        # 2.resource check
        assert count_process_num_on_port(9293) == 1  # det Server
        assert count_process_num_on_port(9292) == 1  # web Server
        assert count_process_num_on_port(12000) == 1  # rec Server
        assert check_gpu_memory(0) is True

        # 3.keywords check

        # 4.predict by http
        # batch_size=1
        self.predict_http(batch_size=1)
        # 从npy文件读取
        rec_result = np.load("fetch_dict_rec.npy", allow_pickle=True).item()
        # 删除文件
        os.system("rm -rf fetch_dict_rec.npy")
        # # 删除lod信息
        # del rec_result["ctc_greedy_decoder_0.tmp_0.lod"], rec_result["softmax_0.tmp_0.lod"]
        self.serving_util.check_result(result_data=rec_result, truth_data=self.truth_val_rec, batch_size=1)

        # 5.release
        kill_process(9292)
        kill_process(9293)
        kill_process(12000)

    def test_cpu_local(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} ocr_debugger_server.py cpu",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9292) == 1  # web Server
        assert check_gpu_memory(0) is False

        # 3.keywords check

        # 4.predict by http
        # batch_size=1
        self.predict_http(batch_size=1)
        # 从npy文件读取
        rec_result = np.load("fetch_dict_rec.npy", allow_pickle=True).item()
        # 删除文件
        os.system("rm -rf fetch_dict_rec.npy")
        # # 删除lod信息
        # del rec_result["ctc_greedy_decoder_0.tmp_0.lod"], rec_result["softmax_0.tmp_0.lod"]
        self.serving_util.check_result(result_data=rec_result, truth_data=self.truth_val_rec, batch_size=1)

        # 5.release
        kill_process(9292)

    def test_gpu_local(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} ocr_debugger_server.py gpu",
            sleep=8,
        )

        # 2.resource check
        assert count_process_num_on_port(9292) == 1  # web Server
        assert check_gpu_memory(0) is False

        # 3.keywords check

        # 4.predict by http
        # batch_size=1
        self.predict_http(batch_size=1)
        # 从npy文件读取
        rec_result = np.load("fetch_dict_rec.npy", allow_pickle=True).item()
        # 删除文件
        os.system("rm -rf fetch_dict_rec.npy")
        # # 删除lod信息
        # del rec_result["ctc_greedy_decoder_0.tmp_0.lod"], rec_result["softmax_0.tmp_0.lod"]
        self.serving_util.check_result(result_data=rec_result, truth_data=self.truth_val_rec, batch_size=1)

        # 5.release
        kill_process(9292, 2)

    def test_cpu_only_det(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} det_web_server.py cpu",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9292) == 1  # web Server
        assert check_gpu_memory(0) is False

        # 3.keywords check

        # 4.predict by http
        # batch_size=1
        result = self.predict_http(batch_size=1)
        print(result, type(result))
        res = result["dt_boxes"]

        # 5.release
        kill_process(9292)

    def test_gpu_only_det(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} det_web_server.py gpu",
            sleep=8,
        )

        # 2.resource check
        assert count_process_num_on_port(9292) == 1  # web Server
        assert check_gpu_memory(0) is True

        # 3.keywords check

        # 4.predict by http
        # batch_size=1
        result = self.predict_http(batch_size=1)
        res = result["dt_boxes"]

        # 5.release
        kill_process(9292, 2)

    def test_cpu_only_det_local(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} det_debugger_server.py cpu",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9292) == 1  # web Server
        assert check_gpu_memory(0) is False

        # 3.keywords check

        # 4.predict by http
        # batch_size=1
        result = self.predict_http(batch_size=1)
        dt_boxes = result["dt_boxes"]

        # 5.release
        kill_process(9292)

    def test_gpu_only_det_local(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} det_debugger_server.py gpu",
            sleep=8,
        )

        # 2.resource check
        assert count_process_num_on_port(9292) == 1  # web Server
        assert check_gpu_memory(0) is False

        # 3.keywords check

        # 4.predict by http
        # batch_size=1
        result = self.predict_http(batch_size=1)
        res = result["dt_boxes"]

        # 5.release
        kill_process(9292, 2)

    def test_cpu_only_rec(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} rec_web_server.py cpu",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9292) == 1  # web Server
        assert check_gpu_memory(0) is False

        # 3.keywords check

        # 4.predict by http
        # batch_size=1
        result = self.predict_http_rec(batch_size=1)
        res = result["res"]

        # 5.release
        kill_process(9292)

    def test_gpu_only_rec(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} rec_web_server.py gpu",
            sleep=8,
        )

        # 2.resource check
        assert count_process_num_on_port(9292) == 1  # web Server
        assert check_gpu_memory(0) is True

        # 3.keywords check

        # 4.predict by http
        # batch_size=1
        result = self.predict_http_rec(batch_size=3)
        res = result["res"]

        # 5.release
        kill_process(9292, 2)

    def test_cpu_only_rec_local(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} rec_debugger_server.py cpu",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9292) == 1  # web Server
        assert check_gpu_memory(0) is False

        # 3.keywords check

        # 4.predict by http
        # batch_size=1
        result = self.predict_http_rec(batch_size=1)
        res = result["res"]

        # 5.release
        kill_process(9292)

    def test_gpu_only_rec_local(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} rec_debugger_server.py gpu",
            sleep=8,
        )

        # 2.resource check
        assert count_process_num_on_port(9292) == 1  # web Server
        assert check_gpu_memory(0) is False

        # 3.keywords check

        # 4.predict by http
        # batch_size=1
        result = self.predict_http_rec(batch_size=1)
        res = result["res"]

        # 5.release
        kill_process(9292, 2)
