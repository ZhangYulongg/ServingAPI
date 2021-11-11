import os
import subprocess
import numpy as np
import copy
import cv2

from paddle_serving_client import Client, HttpClient
from paddle_serving_app.reader import Sequential, URL2Image, Resize, File2Image
from paddle_serving_app.reader import CenterCrop, RGB2BGR, Transpose, Div, Normalize
from paddle_serving_app.reader.imdb_reader import IMDBDataset
import paddle.inference as paddle_infer

from util import *


class TestIMDBABTest(object):
    def setup_class(self):
        serving_util = ServingTest(data_path="imdb", example_path="imdb", model_dir="imdb_cnn_model",
                                   client_dir="imdb_cnn_client_conf")
        serving_util.check_model_data_exist()
        self.get_truth_val_by_inference(self)
        self.serving_util = serving_util
        os.system(f"{self.serving_util.py_version} abtest_get_data.py")

    def teardown_method(self):
        print_log(["stderr.log", "stdout.log",
                   "log/serving.ERROR", "PipelineServingLogs/pipeline.log"], iden="after predict")
        kill_process(8001)
        self.serving_util.release()

    def get_truth_val_by_inference(self):
        # test for AB test
        pass

    def predict_brpc(self, batch_size=1):
        # 1.prepare feed_data

        # 2.init client
        client = Client()
        client.load_client_config('imdb_bow_client_conf/serving_client_conf.prototxt')
        client.add_variant("bow", ["127.0.0.1:8001"], 10)
        client.add_variant("lstm", ["127.0.0.1:9000"], 90)
        client.connect()

        with open('processed.data', encoding="utf-8") as f:
            cnt = {"bow": {'acc': 0, 'total': 0}, "lstm": {'acc': 0, 'total': 0}}
            count = 0
            for line in f:
                word_ids, label = line.split(';')
                word_ids = [int(x) for x in word_ids.split(',')]
                word_len = len(word_ids)
                feed = {
                    "words": np.array(word_ids).reshape(word_len, 1),
                    "words.lod": [0, word_len]
                }
                fetch = ["acc", "cost", "prediction"]
                [fetch_map, tag] = client.predict(feed=feed, fetch=fetch, need_variant_tag=True, batch=True)
                if (float(fetch_map["prediction"][0][1]) - 0.5) * (float(label[0]) - 0.5) > 0:
                    cnt[tag]['acc'] += 1
                cnt[tag]['total'] += 1
                count += 1
                if count >= 100:
                    break

            for tag, data in cnt.items():
                print(f"[{tag}](total: {data['total']}) acc: {float(data['acc'])}")

        return cnt

    def test_cpu_abtest(self):
        # 1.start server
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model imdb_bow_model --port 8001",
            sleep=10,
            err="bowerr.log",
            out="bowout.log",
        )
        self.serving_util.start_server_by_shell(
            cmd=f"{self.serving_util.py_version} -m paddle_serving_server.serve --model imdb_lstm_model --port 9000",
            sleep=10,
        )

        # 2.resource check
        assert count_process_num_on_port(8001) == 1
        assert count_process_num_on_port(9000) == 1
        # assert check_gpu_memory(0) is False

        # 3.keywords check

        # 4.predict by brpc
        # batch_size 1
        result = self.predict_brpc()
        print(result)

        # 5.release
        kill_process(8001)
        kill_process(9000)
