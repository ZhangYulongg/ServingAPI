import os
import subprocess
import numpy as np
import copy

from paddle_serving_client import Client, HttpClient
from paddle_serving_app.reader import BlazeFacePostprocess
from paddle_serving_app.reader import *
import paddle.inference as paddle_infer

import criteo_reader as criteo
from util import *


class TestCriteoCtr(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="criteo_ctr_with_cube", example_path="criteo_ctr_with_cube", model_dir="ctr_serving_model_kv",
                                   client_dir="ctr_client_conf")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util

    def teardown_method(self):
        print_log(["stderr.log", "stdout.log",
                   "log/serving.ERROR", "PipelineServingLogs/pipeline.log"], iden="after predict")
        kill_process(8027)
        kill_process(9292)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        # cube 功能模型暂不做精度校验
        pass

    def predict_brpc(self, batch_size=1):
        # 1.prepare feed_data
        batch = 1
        buf_size = 100
        dataset = criteo.CriteoDataset()
        dataset.setup(1000001)
        test_filelists = ["raw_data/part-0"]
        reader = dataset.infer_reader(test_filelists, batch, buf_size)
        label_list = []
        prob_list = []

        data = reader().__next__()
        feed_dict = {}
        feed_dict['dense_input'] = np.array(data[0][0]).reshape(1, len(data[0][0]))

        for i in range(1, 27):
            feed_dict["embedding_{}.tmp_0".format(i - 1)] = np.array(data[0][i]).reshape(len(data[0][i]))
            feed_dict["embedding_{}.tmp_0.lod".format(i - 1)] = [0, len(data[0][i])]

        # 2.init client
        client = Client()
        client.load_client_config(self.serving_util.client_config)
        client.connect(["127.0.0.1:9292"])
        fetch = ["prob"]

        # 3.predict for fetch_map
        fetch_map = client.predict(feed=feed_dict, fetch=fetch, batch=True)
        print(fetch_map)
        prob_list.append(fetch_map['prob'][0][1])
        label_list.append(data[0][-1][0])

        return fetch_map

    def test_cpu(self):
        # prepare
        os.system("rm -rf cube")
        os.system(f"\cp -rf {self.serving_util.data_path}/cube ./")
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"sh cube_prepare.sh &",
            sleep=5,
            err="cube_err.log",
            out="cube_out.log",
        )
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} test_server.py ctr_serving_model_kv",
            sleep=7,
        )

        # 2.resource check
        assert count_process_num_on_port(9292) == 1
        # assert check_gpu_memory(0) is False
        # assert check_gpu_memory(1) is False

        # 3.keywords check

        # 4.predict
        result = self.predict_brpc(batch_size=1)
        # print(result["prob"].shape, type(result["prob"].shape))
        assert result["prob"].shape == (1, 2)
        # self.serving_util.check_result(result_data=result, truth_data=self.truth_val, batch_size=1)

        # 5.release
        kill_process(8027)
        kill_process(9292)
