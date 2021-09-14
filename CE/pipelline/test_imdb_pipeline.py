import os
import subprocess
import numpy as np
import copy
import cv2
import requests
import json
import sys

from paddle_serving_server.pipeline import PipelineClient
from paddle_serving_app.reader import CenterCrop, RGB2BGR, Transpose, Div, Normalize, RCNNPostprocess
from paddle_serving_app.reader.imdb_reader import IMDBDataset
import paddle.inference as paddle_infer

sys.path.append("../")
from util import *


class TestIMDBPipeline(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="imdb", example_path="pipeline/imdb", model_dir="imdb_cnn_model",
                                   client_dir="imdb_cnn_client_conf")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util

    def teardown_method(self):
        print("======================stderr_cnn.log after predict======================")
        os.system("cat stderr_cnn.log")
        print("======================stdout_cnn.log after predict======================")
        os.system("cat stdout_cnn.log")
        print("====================================================================")
        print("======================stderr_bow.log after predict======================")
        os.system("cat stderr_bow.log")
        print("======================stdout_bow.log after predict======================")
        os.system("cat stdout_bow.log")
        print("====================================================================")
        kill_process(18070)
        self.serving_util.release(keywords="pipeline_server.py")

    def get_truth_val_by_inference(self):
        imdb_dataset = IMDBDataset()
        imdb_dataset.load_resource("imdb.vocab")
        line = "i am very sad | 0"
        word_ids, label = imdb_dataset.get_words_and_label(line)
        word_len = len(word_ids)
        input_dict = {
            "words": np.array(word_ids).reshape(word_len, 1),
            "words.lod": [0, word_len]
        }

        pd_config = paddle_infer.Config("imdb_cnn_model")
        pd_config.disable_gpu()
        pd_config.switch_ir_optim(False)

        predictor = paddle_infer.create_predictor(pd_config)

        input_names = predictor.get_input_names()
        for i, input_name in enumerate(input_names):
            input_handle = predictor.get_input_handle(input_name)
            # 设置变长tensor
            input_handle.set_lod([input_dict[f"{input_name}.lod"]])
            input_handle.copy_from_cpu(input_dict[input_name])

        predictor.run()

        output_data_dict = {}
        output_names = predictor.get_output_names()
        for _, output_data_name in enumerate(output_names):
            output_handle = predictor.get_output_handle(output_data_name)
            output_data = output_handle.copy_to_cpu()
            output_data_dict[output_data_name] = output_data
        # 对齐Serving output
        output_data_dict["prediction"] = output_data_dict["fc_1.tmp_2"]
        del output_data_dict["fc_1.tmp_2"]
        self.truth_val = output_data_dict
        print(self.truth_val, self.truth_val["prediction"].shape)

    def predict_pipeline_rpc(self, batch_size=1):
        # 1.prepare feed_data
        words = 'i am very sad | 0'
        feed_dict = {
            "words": words,
            "logid": 10000
        }
        # TODO 原示例不支持batch

        # 2.init client
        # fetch = ["label", "prob"]
        client = PipelineClient()
        client.connect(['127.0.0.1:18070'])

        # 3.predict for fetch_map
        ret = client.predict(feed_dict=feed_dict, asyn=True, profile=False)
        result = ret.result()
        print(result)
        # 转换为dict
        # TODO 通过截取str转为list，更优雅的方法？
        result = {"prediction": np.array(eval(result.value[0][7:-17]), dtype=np.float32)}
        print(result)
        return result

    def test_cpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model imdb_cnn_model --port 9292",
            sleep=5,
            err="stderr_cnn.log",
            out="stdout_cnn.log",
        )
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model imdb_bow_model --port 9393",
            sleep=5,
            err="stderr_bow.log",
            out="stdout_bow.log",
        )
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} test_pipeline_server.py",
            sleep=5,
            err="stderr_com.log",
            out="stdout_com.log",
        )

        # 2.resource check
        assert count_process_num_on_port(18070) == 1  # gRPC Server
        assert count_process_num_on_port(18071) == 1  # gRPC gateway 代理、转发
        assert check_gpu_memory(0) is False

        # 3.keywords check
        check_keywords_in_server_log("MKLDNN is enabled", filename="stderr.log")

        # 4.predict by rpc
        # batch_size=1
        result = self.predict_pipeline_rpc(batch_size=1)
        self.serving_util.check_result(result_data=result, truth_data=self.truth_val, batch_size=1, delta=5e-2)

        # 5.release
        kill_process(18070)
        kill_process(18071)
        kill_process(9292)
        kill_process(9393)
