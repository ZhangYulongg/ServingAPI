import os
import subprocess
import numpy as np
import copy
import cv2
import requests
import json
import yaml

from paddle_serving_server.pipeline import PipelineClient
from paddle_serving_app.reader import CenterCrop, RGB2BGR, Transpose, Div, Normalize, RCNNPostprocess
from paddle_serving_app.reader import ChineseBertReader
import paddle.inference as paddle_infer

from util import *


class TestBertPipeline(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="bert", example_path="bert", model_dir="bert_seq128_model",
                                   client_dir="bert_seq128_client")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util
        # TODO 为校验精度将模型输出存入npy文件，通过修改server端代码实现，考虑更优雅的方法
        os.system("sed -i '47 i \ \ \ \ \ \ \ \ np.save(\"fetch_dict\", fetch_dict)' web_service.py")
        # 读取yml文件
        with open("config.yml", "r") as file:
            dict_ = yaml.safe_load(file)
        dict_["op"]["bert"]["local_service_conf"]["devices"] = "0"
        self.default_config = dict_

    def teardown_method(self):
        print("======================stderr.log after predict======================")
        os.system("cat stderr.log")
        print("======================stdout.log after predict======================")
        os.system("cat stdout.log")
        print("====================================================================")
        kill_process(9998)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        reader = ChineseBertReader({"max_seq_len": 128, "vocab_file": "vocab.txt"})
        input_dict = reader.process("送晚了，饿得吃得很香")
        for key in input_dict.keys():
            input_dict[key] = np.array(input_dict[key]).reshape((1, 128, 1))
        input_dict["input_mask"] = input_dict["input_mask"].astype("float32")
        input_dict["position_ids"] = input_dict["position_ids"].astype("int64")
        input_dict["input_ids"] = input_dict["input_ids"].astype("int64")
        input_dict["segment_ids"] = input_dict["segment_ids"].astype("int64")

        pd_config = paddle_infer.Config("bert_seq128_model")
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
        output_data_dict["pooled_output"] = output_data_dict["@HUB_bert_chinese_L-12_H-768_A-12@@HUB_bert_chinese_L-12_H-768_A-12@fc_72.tmp_2"]
        output_data_dict["sequence_output"] = output_data_dict["@HUB_bert_chinese_L-12_H-768_A-12@@HUB_bert_chinese_L-12_H-768_A-12@layer_norm_24.tmp_2"]
        del output_data_dict["@HUB_bert_chinese_L-12_H-768_A-12@@HUB_bert_chinese_L-12_H-768_A-12@fc_72.tmp_2"], output_data_dict["@HUB_bert_chinese_L-12_H-768_A-12@@HUB_bert_chinese_L-12_H-768_A-12@layer_norm_24.tmp_2"]
        self.truth_val = output_data_dict
        print(self.truth_val, self.truth_val["pooled_output"].shape, self.truth_val["sequence_output"].shape)

    def predict_pipeline_rpc(self, batch_size=1):
        # 1.prepare feed_data
        sentence = "送晚了，饿得吃得很香"
        # pipeline client 仅支持key值为bytes或unicode str
        feed_dict = {str(i): sentence for i in range(batch_size)}

        # 2.init client
        client = PipelineClient()
        client.connect(['127.0.0.1:9998'])

        # 3.predict for fetch_map
        ret = client.predict(feed_dict=feed_dict)
        print(ret)
        # 从npy文件读取 .item()转换为dict
        result = np.load("fetch_dict.npy", allow_pickle=True).item()
        os.system("rm -rf fetch_dict.npy")
        print(result["pooled_output"].shape, result["sequence_output"].shape)
        return result

    def predict_pipeline_http(self, batch_size=1):
        # 1.prepare feed_data
        sentence = "送晚了，饿得吃得很香"
        feed_dict = {
            "key": [str(i) for i in range(batch_size)],
            "value": [sentence for i in range(batch_size)],
        }

        # 2.predict for fetch_map
        url = "http://127.0.0.1:18082/bert/prediction"
        r = requests.post(url=url, data=json.dumps(feed_dict))
        print(r.json())
        # 从npy文件读取 .item()转换为dict
        result = np.load("fetch_dict.npy", allow_pickle=True).item()
        os.system("rm -rf fetch_dict.npy")
        print(result["pooled_output"].shape, result["sequence_output"].shape)
        return result

    def test_gpu(self):
        # 1.start server
        # 生成config.yml
        config = copy.deepcopy(self.default_config)
        config["op"]["bert"]["concurrency"] = 1
        config["op"]["bert"]["local_service_conf"]["device_type"] = 1
        config["op"]["bert"]["local_service_conf"]["devices"] = "0"
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
        result = self.predict_pipeline_rpc(batch_size=2)
        self.serving_util.check_result(result_data=result, truth_data=self.truth_val, batch_size=2)
        # # predict by http
        result = self.predict_pipeline_http(batch_size=1)  # batch_size=1
        self.serving_util.check_result(result_data=result, truth_data=self.truth_val, batch_size=1)
        result = self.predict_pipeline_http(batch_size=2)  # batch_size=2
        self.serving_util.check_result(result_data=result, truth_data=self.truth_val, batch_size=2)

        # 5.release
        kill_process(9998)
        kill_process(18082)


if __name__ == '__main__':
    sss = TestBertPipeline()
    sss.predict_pipeline_http(batch_size=1)
    sss.predict_pipeline_http(batch_size=2)





