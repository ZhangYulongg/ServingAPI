import os
import subprocess
import numpy as np
import copy
import cv2
import requests
import json
import yaml
import sys

from paddle_serving_server.pipeline import PipelineClient
from paddle_serving_app.reader import OCRReader
from paddle_serving_app.reader import Sequential, ResizeByFactor
from paddle_serving_app.reader import Div, Normalize, Transpose, File2Image
from paddle_serving_app.reader import DBPostProcess, FilterBoxes, GetRotateCropImage, SortedBoxes
import paddle.inference as paddle_infer

sys.path.append("../")
from util import *


class TestOCRPipeline(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="ocr_pipe", example_path="Pipeline/PaddleOCR/ocr", model_dir="ocr_det_model",
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
        os.system("sed -i '95 i \ \ \ \ \ \ \ \ np.save(\"fetch_dict_det\", fetch_dict)' web_service.py")
        os.system("sed -i '215 i \ \ \ \ \ \ \ \ np.save(\"fetch_dict_rec\", fetch_data)' web_service.py")
        # 读取yml文件
        with open("config.yml", "r", encoding="utf-8") as file:
            dict_ = yaml.safe_load(file)
        self.default_config = dict_

    def teardown_method(self):
        print_log(["stderr.log", "stdout.log",
                   "log/serving.ERROR", "PipelineServingLogs/pipeline.log"], iden="after predict")
        kill_process(9999)
        self.serving_util.release()

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

    def predict_pipeline_rpc(self, batch_size=1):
        # 1.prepare feed_data
        filename = "imgs/1.jpg"
        with open(filename, "rb") as f:
            image_data = f.read()
        image = cv2_to_base64(image_data)
        feed_dict = {"image": image}

        # 2.init client
        client = PipelineClient()
        client.connect(['127.0.0.1:18090'])

        # 3.predict for fetch_map
        ret = client.predict(feed_dict=feed_dict)
        print(ret)
        # 转换为dict
        result = {"res": eval(ret.value[0])}
        print(result)

        # 从npy文件读取
        det_result = np.load("fetch_dict_det.npy", allow_pickle=True).item()
        rec_result = np.load("fetch_dict_rec.npy", allow_pickle=True).item()
        os.system("rm -rf fetch_dict_det.npy fetch_dict_rec.npy")
        # 删除lod信息
        # del rec_result["ctc_greedy_decoder_0.tmp_0.lod"], rec_result["softmax_0.tmp_0.lod"]
        return det_result, rec_result

    def predict_pipeline_http(self, batch_size=1):
        # 1.prepare feed_data
        filename = "imgs/1.jpg"
        with open(filename, "rb") as f:
            image_data = f.read()
        image = cv2_to_base64(image_data)
        feed_dict = {"key": [], "value": []}
        for i in range(batch_size):
            feed_dict["key"].append(f"image_{i}")
            feed_dict["value"].append(image)

        # 2.predict for fetch_map
        url = "http://127.0.0.1:9999/ocr/prediction"
        r = requests.post(url=url, data=json.dumps(feed_dict))
        print(r.json())
        # 从npy文件读取
        det_result = np.load("fetch_dict_det.npy", allow_pickle=True).item()
        rec_result = np.load("fetch_dict_rec.npy", allow_pickle=True).item()
        # 删除文件
        os.system("rm -rf fetch_dict_det.npy fetch_dict_rec.npy")
        # 删除lod信息
        # del rec_result["ctc_greedy_decoder_0.tmp_0.lod"], rec_result["softmax_0.tmp_0.lod"]
        return det_result, rec_result

    def test_cpu_ir(self):
        # edit config.yml
        # 生成config.yml
        config = copy.deepcopy(self.default_config)
        config["op"]["det"]["local_service_conf"]["device_type"] = 0
        config["op"]["det"]["local_service_conf"]["devices"] = ""
        try:
            del config["op"]["det"]["local_service_conf"]["min_subgraph_size"]
            del config["op"]["rec"]["local_service_conf"]["min_subgraph_size"]
        except KeyError as e:
            print("config is default")
        config["op"]["det"]["local_service_conf"]["min_subgraph_size"] = 13
        config["op"]["rec"]["local_service_conf"]["device_type"] = 0
        config["op"]["rec"]["local_service_conf"]["devices"] = ""
        with open("config.yml", "w") as f:
            yaml.dump(config, f)
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} web_service.py",
            sleep=10,
        )

        # 2.resource check
        assert count_process_num_on_port(9999) == 1  # gRPC Server
        assert count_process_num_on_port(18090) == 1  # gRPC gateway 代理、转发
        assert check_gpu_memory(0) is False

        # 3.keywords check
        check_keywords_in_server_log("Running IR pass", filename="stderr.log")

        # 4.predict by rpc
        # batch_size=1
        det_result, rec_result = self.predict_pipeline_rpc(batch_size=1)
        print(type(det_result), type(rec_result))
        print(det_result["save_infer_model/scale_0.tmp_1"].shape)
        print(rec_result["save_infer_model/scale_0.tmp_1"].shape)
        self.serving_util.check_result(result_data=det_result, truth_data=self.truth_val_det, batch_size=1)
        self.serving_util.check_result(result_data=rec_result, truth_data=self.truth_val_rec, batch_size=1)
        # predict by http
        det_result, rec_result = self.predict_pipeline_http(batch_size=1)  # batch_size=1
        self.serving_util.check_result(result_data=det_result, truth_data=self.truth_val_det, batch_size=1)
        self.serving_util.check_result(result_data=rec_result, truth_data=self.truth_val_rec, batch_size=1)
        # det_result, rec_result = self.predict_pipeline_http(batch_size=2)

        # 5.release
        kill_process(9999)
        kill_process(18090)

    def test_gpu_trt_fp32(self):
        # edit config.yml
        # 生成config.yml
        config = copy.deepcopy(self.default_config)
        config["op"]["det"]["local_service_conf"]["device_type"] = 2
        config["op"]["det"]["local_service_conf"]["devices"] = "0"
        config["op"]["det"]["local_service_conf"]["min_subgraph_size"] = 13
        config["op"]["rec"]["local_service_conf"]["device_type"] = 2
        config["op"]["rec"]["local_service_conf"]["devices"] = "1"
        config["op"]["rec"]["local_service_conf"]["min_subgraph_size"] = 3
        with open("config.yml", "w") as f:
            yaml.dump(config, f)
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} web_service.py",
            sleep=120,
        )

        # 2.resource check
        assert count_process_num_on_port(9999) == 1  # gRPC Server
        assert count_process_num_on_port(18090) == 1  # gRPC gateway 代理、转发
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is True

        # 3.keywords check
        check_keywords_in_server_log("Prepare TRT engine", filename="stderr.log")

        # 4.predict by rpc
        # batch_size=1
        det_result, rec_result = self.predict_pipeline_rpc(batch_size=1)
        print(type(det_result), type(rec_result))
        print(det_result["save_infer_model/scale_0.tmp_1"].shape)
        print(rec_result["save_infer_model/scale_0.tmp_1"].shape)
        self.serving_util.check_result(result_data=det_result, truth_data=self.truth_val_det, batch_size=1)
        # self.serving_util.check_result(result_data=rec_result, truth_data=self.truth_val_rec, batch_size=1)
        # predict by http
        det_result, rec_result = self.predict_pipeline_http(batch_size=1)  # batch_size=1
        self.serving_util.check_result(result_data=det_result, truth_data=self.truth_val_det, batch_size=1)
        # self.serving_util.check_result(result_data=rec_result, truth_data=self.truth_val_rec, batch_size=1)

        # 5.release
        kill_process(9999)
        kill_process(18090)
