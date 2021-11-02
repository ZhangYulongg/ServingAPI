import os
import subprocess
import numpy as np
import copy
import cv2
import requests
import json
import sys
import time

from paddle_serving_client import Client, HttpClient
from paddle_serving_app.reader import OCRReader
from paddle_serving_app.reader import Sequential, ResizeByFactor
from paddle_serving_app.reader import Div, Normalize, Transpose, File2Image
from paddle_serving_app.reader import DBPostProcess, FilterBoxes, GetRotateCropImage, SortedBoxes
import paddle.inference as paddle_infer

sys.path.append("../")
from util import *


def print_log(file_list, iden=""):
    for file in file_list:
        print(f"======================{file} {iden}=====================")
        if os.path.exists(file):
            with open(file, "r") as f:
                print(f.read())
        else:
            print(f"{file} not exist")
        print("======================================================")


class TestOCR(object):
    def setup_class(self):
        code_path = os.environ.get("CODE_PATH")
        self.data_path = f"{os.environ.get('DATA_PATH')}\\test_ocr\\"
        self.example_path = f"{code_path}\\test_ocr\\"
        self.py_version = os.environ.get("py_version")
        self.model_dir = "ocr_det_model"
        self.client_config = f"ocr_det_client\\serving_client_conf.prototxt"

        os.chdir(self.example_path)
        print("======================cur path======================")
        print(os.getcwd())
        os.system("python -m wget https://paddle-qa.bj.bcebos.com/tools/busybox64.exe")
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
        os.system("python -m paddle_serving_server.serve stop")
        os.system("sleep 2")
        # kill_process(9293)
        # os.system("kill -9 `ps -ef | grep serving | awk '{print $2}'` > /dev/null 2>&1")
        # os.system("kill -9 `ps -ef | grep server.py | awk '{print $2}'` > /dev/null 2>&1")

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
        input_dict["image"] = det_img.astype("float32")

        pd_config = paddle_infer.Config("ocr_det_model")
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
        output_data_dict["concat_1.tmp_0"] = output_data_dict["save_infer_model/scale_0.tmp_0"]
        del output_data_dict["save_infer_model/scale_0.tmp_0"]
        self.truth_val_det = output_data_dict
        print(self.truth_val_det, self.truth_val_det["concat_1.tmp_0"].shape)

        # Rec result
        sorted_boxes = SortedBoxes()
        get_rotate_crop_image = GetRotateCropImage()
        ratio_list = [float(new_h) / ori_h, float(new_w) / ori_w]
        dt_boxes_list = self.post_func(output_data_dict["concat_1.tmp_0"], [ratio_list])
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
        input_dict = {"image": np.concatenate(feed_list, axis=0)}

        pd_config = paddle_infer.Config("ocr_rec_model")
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
        output_data_dict["ctc_greedy_decoder_0.tmp_0"] = output_data_dict["save_infer_model/scale_0.tmp_0"]
        output_data_dict["softmax_0.tmp_0"] = output_data_dict["save_infer_model/scale_1.tmp_0"]
        del output_data_dict["save_infer_model/scale_0.tmp_0"], output_data_dict["save_infer_model/scale_1.tmp_0"]
        self.truth_val_rec = output_data_dict
        print(self.truth_val_rec, self.truth_val_rec["ctc_greedy_decoder_0.tmp_0"].shape,
              self.truth_val_rec["softmax_0.tmp_0"].shape)

    def predict_http(self, batch_size=1):
        # 1.prepare feed_data
        filename = "imgs/1.jpg"
        with open(filename, "rb") as f:
            image_data = f.read()
        image = cv2_to_base64(image_data)
        feed_dict = {
            "feed": [{"image": image}],
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
            "feed": [{"image": image}] * batch_size,
            "fetch": ["res"],
        }
        # 2.predict for fetch_map
        url = "http://127.0.0.1:9292/ocr/prediction"
        headers = {"Content-type": "application/json"}
        r = requests.post(url=url, headers=headers, data=json.dumps(feed_dict))
        print(r.json())
        return r.json()["result"]

    def test_cpu_local(self):
        # 1.start server
        self.err = open("stderr.log", "w")
        self.out = open("stdout.log", "w")
        subprocess.Popen("python ocr_debugger_server.py cpu", shell=True, stdout=self.out, stderr=self.err)
        time.sleep(5)
        print_log(["stderr.log", "stdout.log"])

        # 2.resource check
        command = "busybox64.exe netstat -nlp | grep :" + str(port) + " | wc -l"
        count = eval(os.popen(command).read())
        print(f"port-9292 processes num:", count)
        assert count == 1  # web Server

        # 3.keywords check

        # 4.predict by http
        # batch_size=1
        self.predict_http(batch_size=1)

        # 5.release
        # kill_process(9292)

    def test_gpu_local(self):
        # 1.start server
        self.err = open("stderr.log", "w")
        self.out = open("stdout.log", "w")
        subprocess.Popen("python ocr_debugger_server.py gpu", shell=True, stdout=self.out, stderr=self.err)
        time.sleep(8)
        print_log(["stderr.log", "stdout.log"])

        # 2.resource check
        command = "busybox64.exe netstat -nlp | grep :" + str(port) + " | wc -l"
        count = eval(os.popen(command).read())
        print(f"port-9292 processes num:", count)
        assert count == 1  # web Server

        # 3.keywords check

        # 4.predict by http
        # batch_size=1
        self.predict_http(batch_size=1)

        # 5.release
        # kill_process(9292, 2)
