import os
import subprocess
import numpy as np
import copy
import cv2
import faiss
import pickle

from paddle_serving_client import Client, HttpClient
from paddle_serving_app.reader import Sequential, URL2Image, Resize, File2Image
from paddle_serving_app.reader import CenterCrop, RGB2BGR, Transpose, Div, Normalize
import paddle.inference as paddle_infer

from util import *


def init_index(index_dir):
    assert os.path.exists(os.path.join(
        index_dir, "vector.index")), "vector.index not found ..."
    assert os.path.exists(os.path.join(
        index_dir, "id_map.pkl")), "id_map.pkl not found ... "

    searcher = faiss.read_index(
        os.path.join(index_dir, "vector.index"))

    with open(os.path.join(index_dir, "id_map.pkl"), "rb") as fd:
        id_map = pickle.load(fd)
    return searcher, id_map


#get box
def nms_to_rec_results(results, thresh=0.1):
    filtered_results = []

    x1 = np.array([r["bbox"][0] for r in results]).astype("float32")
    y1 = np.array([r["bbox"][1] for r in results]).astype("float32")
    x2 = np.array([r["bbox"][2] for r in results]).astype("float32")
    y2 = np.array([r["bbox"][3] for r in results]).astype("float32")
    scores = np.array([r["rec_scores"] for r in results])

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    while order.size > 0:
        i = order[0]
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
        filtered_results.append(results[i])
    return filtered_results


def postprocess(fetch_dict,
                feature_normalize,
                det_boxes,
                searcher,
                id_map,
                return_k,
                rec_score_thres,
                rec_nms_thresold):
    batch_features = fetch_dict["features"]

    #do feature norm
    if feature_normalize:
        feas_norm = np.sqrt(
            np.sum(np.square(batch_features), axis=1, keepdims=True))
        batch_features = np.divide(batch_features, feas_norm)

    scores, docs = searcher.search(batch_features, return_k)

    results = []
    for i in range(scores.shape[0]):
        pred = {}
        if scores[i][0] >= rec_score_thres:
            pred["bbox"] = [int(x) for x in det_boxes[i,2:]]
            pred["rec_docs"] = id_map[docs[i][0]].split()[1]
            pred["rec_scores"] = scores[i][0]
            results.append(pred)

    #do nms
    results = nms_to_rec_results(results, rec_nms_thresold)
    return results


class TestPPShitu(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="pp_shitu", example_path="C++/PaddleClas/pp_shitu", model_dir="picodet_PPLCNet_x2_5_mainbody_lite_v2.0_serving",
                                   client_dir="picodet_PPLCNet_x2_5_mainbody_lite_v2.0_client")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util

    def teardown_method(self):
        print_log(["stderr.log", "stdout.log",
                   "log/serving.ERROR", "PipelineServingLogs/pipeline.log"], iden="after predict")
        kill_process(9400)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        # 含特殊OP，暂不取inference结果
        pass

    def predict_brpc(self, batch_size=1):
        rec_nms_thresold = 0.05
        rec_score_thres = 0.5
        feature_normalize = True
        return_k = 1
        index_dir = "./drink_dataset_v1.0/index"
        # 1.prepare feed_data
        is_batch = False
        filename = "./drink_dataset_v1.0/test_images/nongfu_spring.jpeg"
        im = cv2.imread(filename)
        im_shape = np.array(im.shape[:2]).reshape(-1)
        if batch_size > 1:
            is_batch = True
            im = im[np.newaxis, :]
            im = np.repeat(im, repeats=batch_size, axis=0)
            im_shape = im_shape[np.newaxis, :]
            im_shape = np.repeat(im_shape, repeats=batch_size, axis=0)
        feed_dict = {
            "image": im,
            "im_shape": im_shape
        }

        fetch = ["features", "boxes"]
        endpoint_list = ['127.0.0.1:9400']

        # 2.init client
        client = Client()
        client.load_client_config(["picodet_PPLCNet_x2_5_mainbody_lite_v2.0_client", "general_PPLCNet_x2_5_lite_v2.0_client"])
        client.connect(endpoint_list)

        # 3.predict for fetch_map
        print(im.shape, is_batch)
        fetch_map = client.predict(feed=feed_dict, fetch=fetch, batch=is_batch)
        print("features.shape:", fetch_map["features"].shape)
        print("boxes.shape:", fetch_map["boxes"].shape)
        # postprocess
        det_boxes = fetch_map["boxes"]
        searcher, id_map = init_index(index_dir)
        results = postprocess(fetch_map, feature_normalize, det_boxes, searcher, id_map, return_k, rec_score_thres,
                              rec_nms_thresold)
        print(results)
        return fetch_map

    def predict_http(self, mode="proto", compress=False, batch_size=1):
        rec_nms_thresold = 0.05
        rec_score_thres = 0.5
        feature_normalize = True
        return_k = 1
        index_dir = "./drink_dataset_v1.0/index"
        # 1.prepare feed_data
        is_batch = False
        filename = "./drink_dataset_v1.0/test_images/nongfu_spring.jpeg"
        im = cv2.imread(filename)
        im_shape = np.array(im.shape[:2]).reshape(-1)
        if batch_size > 1:
            is_batch = True
            im = im[np.newaxis, :]
            im = np.repeat(im, repeats=batch_size, axis=0)
            im_shape = im_shape[np.newaxis, :]
            im_shape = np.repeat(im_shape, repeats=batch_size, axis=0)
        feed_dict = {
            "image": im,
            "im_shape": im_shape
        }

        fetch = ["features", "boxes"]
        endpoint_list = ['127.0.0.1:9400']

        client = HttpClient()
        client.load_client_config(["picodet_PPLCNet_x2_5_mainbody_lite_v2.0_client", "general_PPLCNet_x2_5_lite_v2.0_client"])
        if mode == "proto":
            client.set_http_proto(True)
        elif mode == "json":
            client.set_http_proto(False)
        elif mode == "grpc":
            client.set_use_grpc_client(True)
        else:
            exit(-1)
        if compress:
            client.set_response_compress(True)
            client.set_request_compress(True)
        client.connect(endpoint_list)

        fetch_map = client.predict(feed=feed_dict, fetch=fetch, batch=is_batch)
        # 转换为dict包numpy array
        result_dict = self.serving_util.parse_http_result(fetch_map)
        print(result_dict)
        return result_dict

    def test_cpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve "
                f"--model picodet_PPLCNet_x2_5_mainbody_lite_v2.0_serving general_PPLCNet_x2_5_lite_v2.0_serving "
                f"--op GeneralPicodetOp GeneralFeatureExtractOp "
                f"--port 9400",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9400) == 1
        assert check_gpu_memory(0) is False

        # 3.keywords check

        # 4.predict by brpc
        # batch_size 1
        result_data = self.predict_brpc(batch_size=1)
        assert result_data["features"].shape == (6, 512)
        # predict by http
        # TODO HTTP client暂不支持uint8类型输入
        # batch_size 1
        # result_data = self.predict_http(batch_size=1)

        # 5.release
        stop_server()
        assert count_process_num_on_port(9400) == 0

    def test_gpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve "
                f"--model picodet_PPLCNet_x2_5_mainbody_lite_v2.0_serving general_PPLCNet_x2_5_lite_v2.0_serving "
                f"--op GeneralPicodetOp GeneralFeatureExtractOp "
                f"--gpu_ids 0 "
                f"--port 9400",
            sleep=8,
        )

        # 2.resource check
        assert count_process_num_on_port(9400) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is False

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU", filename="stderr.log")

        # 4.predict by brpc
        # batch_size 1
        result_data = self.predict_brpc(batch_size=1)
        assert result_data["features"].shape == (6, 512)
        # predict by http
        # batch_size 1
        # result_data = self.predict_http(batch_size=1)

        # 5.release
        stop_server()
        assert count_process_num_on_port(9400) == 0
