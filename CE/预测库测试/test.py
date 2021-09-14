import sys
from paddle_serving_client import Client
from paddle_serving_client.utils import benchmark_args
from chinese_ernie_reader import ChineseErnieReader
import numpy as np
import paddle.inference as paddle_infer


data = "送晚了，饿得吃得很香"
reader = ChineseErnieReader({"max_seq_len": 40})

feed_dict = reader.process(data)
for k, v in feed_dict.items():
    print(k, len(v), v)
