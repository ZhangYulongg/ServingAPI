import os
import subprocess
import numpy as np
import copy

from paddle_serving_client import Client, HttpClient
from paddle_serving_app.reader import BlazeFacePostprocess
from paddle_serving_app.reader import *
import paddle.inference as paddle_infer

from util import *


class CriteoReader(object):
    def __init__(self, sparse_feature_dim):
        self.cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cont_max_ = [
            20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50
        ]
        self.cont_diff_ = [
            20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50
        ]
        self.hash_dim_ = sparse_feature_dim
        # here, training data are lines with line_index < train_idx_
        self.train_idx_ = 41256555
        self.continuous_range_ = range(1, 14)
        self.categorical_range_ = range(14, 40)

    def process_line(self, line):
        features = line.rstrip('\n').split('\t')
        dense_feature = []
        sparse_feature = []
        for idx in self.continuous_range_:
            if features[idx] == '':
                dense_feature.append(0.0)
            else:
                dense_feature.append((float(features[idx]) - self.cont_min_[idx - 1]) / \
                                     self.cont_diff_[idx - 1])
        for idx in self.categorical_range_:
            sparse_feature.append(
                [hash(str(idx) + features[idx]) % self.hash_dim_])

        return sparse_feature


class TestCriteoCtr(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="criteo_ctr", example_path="criteo_ctr", model_dir="ctr_serving_model",
                                   client_dir="ctr_client_conf")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util

    def teardown_method(self):
        print("======================stderr.log after predict======================")
        os.system("cat stderr.log")
        print("======================stdout.log after predict======================")
        os.system("cat stdout.log")
        print("====================================================================")
        kill_process(9292)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        reader = CriteoReader(1000001)
        f = open("raw_data/part-0", "r")
        data = reader.process_line(f.readline())
        input_dict = {}
        for i in range(1, 27):
            input_dict[f"C{i}"] = np.array(data[i - 1]).reshape((-1, 1)).astype("int64")
            input_dict[f"C{i}.lod"] = [0, len(data[i - 1])]
        f.close()

        pd_config = paddle_infer.Config("ctr_serving_model")
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
        output_data_dict["prob"] = output_data_dict["fc_3.tmp_2"]
        del output_data_dict["fc_3.tmp_2"]
        self.truth_val = output_data_dict
        print(self.truth_val, self.truth_val["prob"].shape)

    def predict_brpc(self, batch_size=1):
        reader = CriteoReader(1000001)
        f = open("raw_data/part-0", "r")
        data = reader.process_line(f.readline())
        feed_dict = {}
        for i in range(1, 27):
            feed_dict[f"sparse_{i - 1}"] = np.array(data[i - 1]).reshape(-1)
            feed_dict[f"sparse_{i - 1}.lod"] = [0, len(data[i - 1])]
        f.close()

        fetch = ["prob"]
        endpoint_list = ['127.0.0.1:9292']

        client = Client()
        client.load_client_config(self.serving_util.client_config)
        client.connect(endpoint_list)

        fetch_map = client.predict(feed=feed_dict, fetch=fetch)
        print(fetch_map)
        return fetch_map

    def test_cpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model ctr_serving_model/ --port 9292",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9292) == 1
        assert check_gpu_memory(0) is False
        assert check_gpu_memory(1) is False

        # 3.keywords check

        # 4.predict
        result = self.predict_brpc(batch_size=1)
        self.serving_util.check_result(result_data=result, truth_data=self.truth_val, batch_size=1)

        # 5.release
        kill_process(9292)

    def test_gpu(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model ctr_serving_model/ --port 9292 --gpu_ids 0",
            sleep=5,
        )

        # 2.resource check
        assert count_process_num_on_port(9292) == 1
        assert check_gpu_memory(0) is True
        assert check_gpu_memory(1) is False

        # 3.keywords check
        check_keywords_in_server_log("Sync params from CPU to GPU")

        # 4.predict
        result = self.predict_brpc(batch_size=1)
        self.serving_util.check_result(result_data=result, truth_data=self.truth_val, batch_size=1)

        # 5.release
        kill_process(9292, 1)
