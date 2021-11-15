import os
import subprocess
import numpy as np
import copy

from paddle_serving_client import Client, HttpClient
from paddle_serving_app.reader import BlazeFacePostprocess
from paddle_serving_app.reader import *
import paddle.inference as paddle_infer
import paddle.fluid.incubate.data_generator as dg

from util import *


class CriteoDataset(dg.MultiSlotDataGenerator):
    def setup(self, sparse_feature_dim):
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

    def _process_line(self, line):
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

        return dense_feature, sparse_feature, [int(features[0])]

    def infer_reader(self, filelist, batch, buf_size):
        def local_iter():
            for fname in filelist:
                with open(fname.strip(), "r") as fin:
                    for line in fin:
                        dense_feature, sparse_feature, label = self._process_line(
                            line)
                        #yield dense_feature, sparse_feature, label
                        yield [dense_feature] + sparse_feature + [label]

        import paddle
        batch_iter = paddle.batch(
            paddle.reader.shuffle(
                local_iter, buf_size=buf_size),
            batch_size=batch)
        return batch_iter

    def generate_sample(self, line):
        def data_iter():
            dense_feature, sparse_feature, label = self._process_line(line)
            feature_name = ["dense_input"]
            for idx in self.categorical_range_:
                feature_name.append("C" + str(idx - 13))
            feature_name.append("label")
            yield zip(feature_name, [dense_feature] + sparse_feature + [label])

        return data_iter


class TestCriteoCtrWithCube(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="criteo_ctr_with_cube", example_path="C++/PaddleRec/criteo_ctr_with_cube", model_dir="ctr_serving_model_kv",
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
        dataset = CriteoDataset()
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
