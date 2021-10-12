import os
import subprocess
import numpy as np
import copy
import cv2
import requests
import json
import sys
import yaml

from paddle_serving_server.pipeline import PipelineClient
from paddle_serving_app.reader import CenterCrop, RGB2BGR, Transpose, Div, Normalize, RCNNPostprocess
from paddle_serving_app.reader import Sequential, File2Image, Resize, Transpose, BGR2RGB, SegPostprocess
import paddle.inference as paddle_infer

sys.path.append("../../")
from util import *


class TestPPYOLO_mbv3(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="detection/ppyolo_mbv3", example_path="pipeline/PaddleDetection/ppyolo_mbv3", model_dir="serving_server",
                                   client_dir="serving_client")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util
        # TODO 为校验精度将模型输出存入npy文件，通过修改server端代码实现，考虑更优雅的方法
        os.system("sed -i '61 i \ \ \ \ \ \ \ \ np.save(\"fetch_dict\", fetch_dict)' web_service.py")
        # 读取yml文件
        with open("config.yml", "r") as file:
            dict_ = yaml.safe_load(file)
        dict_["op"]["ppyolo_mbv3"]["local_service_conf"]["devices"] = "0"
        self.default_config = dict_

    def teardown_method(self):
        print_log(["stderr.log", "stdout.log",
                   "log/serving.ERROR", "PipelineServingLogs/pipeline.log"], iden="after predict")
        kill_process(9993)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        preprocess = Sequential([
            File2Image(), BGR2RGB(), Div(255.0),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], False),
            Resize((320, 320)), Transpose((2, 0, 1))
        ])
        file_name = "000000570688.jpg"
        im = preprocess(file_name)[np.newaxis, :]
        input_dict = {}
        input_dict["im_shape"] = np.array(list(im.shape[2:])).astype("float32")[np.newaxis, :]
        input_dict["image"] = im.astype("float32")
        input_dict["scale_factor"] = np.array([1.0, 1.0]).astype("float32")[np.newaxis, :]

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
        del output_data_dict["save_infer_model/scale_0.tmp_0"]
        del output_data_dict["save_infer_model/scale_1.tmp_0"]
        self.truth_val = output_data_dict
        print(self.truth_val, self.truth_val["save_infer_model/scale_0.tmp_1"].shape,
              self.truth_val["save_infer_model/scale_1.tmp_1"].shape)

    def predict_pipeline_rpc(self, batch_size=1):
        # 1.prepare feed_data
        with open("000000570688.jpg", "rb") as f:
            image_data = f.read()
        image = cv2_to_base64(image_data)
        feed_dict = {f"image_{i}": image for i in range(batch_size)}

        # 2.init client
        # fetch = ["label"]
        client = PipelineClient()
        client.connect(['127.0.0.1:9998'])

        # 3.predict for fetch_map
        ret = client.predict(feed_dict=feed_dict)
        print(ret)
        # 从npy文件读取 .item()转换为dict
        result = np.load("fetch_dict.npy", allow_pickle=True).item()

        postprocess = RCNNPostprocess("label_list.txt", "output", [320, 320])
        # output_data_dict["save_infer_model/scale_0.tmp_0.lod"] = np.array([ 0, 71], dtype="int32")
        dict_ = copy.deepcopy(result)
        dict_["image"] = "000000570688.jpg"
        postprocess(dict_)

        # 删除lod信息
        print(result["save_infer_model/scale_0.tmp_1"].shape)
        del result["save_infer_model/scale_0.tmp_1.lod"]
        return result

    def predict_pipeline_http(self, batch_size=1):
        # 1.prepare feed_data
        with open("000000570688.jpg", "rb") as f:
            image_data = f.read()
        image = cv2_to_base64(image_data)
        feed_dict = {"key": [], "value": []}
        for i in range(batch_size):
            feed_dict["key"].append(f"image_{i}")
            feed_dict["value"].append(image)

        # 2.predict for fetch_map
        url = "http://127.0.0.1:18082/ppyolo_mbv3/prediction"
        r = requests.post(url=url, data=json.dumps(feed_dict))
        print(r.json())
        # 从npy文件读取 .item()转换为dict
        result = np.load("fetch_dict.npy", allow_pickle=True).item()
        # 删除lod信息
        print(result["save_infer_model/scale_0.tmp_1"].shape)
        del result["save_infer_model/scale_0.tmp_1.lod"]
        return result

    def test_gpu(self):
        # 1.start server
        config = copy.deepcopy(self.default_config)
        with open("config.yml", "w") as f:
            yaml.dump(config, f)
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} web_service.py",
            sleep=10,
        )

        # 2.resource check
        assert count_process_num_on_port(9998) == 1  # gRPC Server
        assert count_process_num_on_port(18082) == 1  # gRPC gateway 代理、转发
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is False

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU", filename="stderr.log")

        # 4.predict by rpc
        # batch_size=1
        result = self.predict_pipeline_rpc(batch_size=1)
        self.serving_util.check_result(result_data=result, truth_data=self.truth_val, batch_size=1)
        # batch_size=2
        # TODO 多batch报错
        # result = self.predict_pipeline_rpc(batch_size=2)
        # self.serving_util.check_result(result_data=result, truth_data=self.truth_val, batch_size=2)
        # predict by http
        result = self.predict_pipeline_http(batch_size=1)  # batch_size=1
        self.serving_util.check_result(result_data=result, truth_data=self.truth_val, batch_size=1)
        # TODO 多batch报错
        # result = self.predict_pipeline_http(batch_size=2)  # batch_size=2
        # self.serving_util.check_result(result_data=result, truth_data=self.truth_val, batch_size=2)

        # 5.release
        kill_process(9998, 2)
        kill_process(18082)


if __name__ == '__main__':
    sss = TestPPYOLO_mbv3()
    sss.predict_pipeline_rpc(batch_size=2)

