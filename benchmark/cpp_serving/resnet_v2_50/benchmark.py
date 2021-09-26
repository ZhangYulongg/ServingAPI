# -*- coding: utf-8 -*-
from __future__ import unicode_literals, absolute_import
import os
import sys
import time
import json
import requests
import numpy as np
from paddle_serving_client import Client
from paddle_serving_client.utils import MultiThreadRunner
from paddle_serving_client.utils import benchmark_args, show_latency
from paddle_serving_app.reader import Sequential, File2Image, Resize, CenterCrop
from paddle_serving_app.reader import RGB2BGR, Transpose, Div, Normalize

args = benchmark_args()

seq = Sequential([
                    File2Image(), Resize(256), CenterCrop(224), RGB2BGR(),
                    Transpose((2, 0, 1)), Div(255), Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True)
                ])
image_file = "daisy.jpg"
img = seq(image_file)
feed_data = np.array(img)
feed_data = np.expand_dims(feed_data, 0).repeat(
    args.batch_size, axis=0)


def single_func(idx, resource):
    total_number = 0
    latency_list = []

    client = Client()
    client.load_client_config(args.model)
    client.connect(resource["endpoint"])
    start = time.time()
    for i in range(turns):
        if args.batch_size >= 1:
            l_start = time.time()
            result = client.predict(
                feed={"image": feed_data},
                fetch=["save_infer_model/scale_0.tmp_0"],
                batch=True)
            l_end = time.time()
            # o_start = time.time()
            latency_list.append(l_end * 1000 - l_start * 1000)
            total_number = total_number + 1
            # o_end = time.time()
        else:
            print("unsupport batch size {}".format(args.batch_size))

    end = time.time()
    return [[end - start], latency_list, [total_number]]


if __name__ == '__main__':
    multi_thread_runner = MultiThreadRunner()
    endpoint_list = ["127.0.0.1:9393"]
    turns = 100
    start = time.time()
    result = multi_thread_runner.run(
        single_func, args.thread, {"endpoint": endpoint_list,
                                   "turns": turns})
    end = time.time()
    total_cost = end - start
    total_number = 0
    avg_cost = 0
    for i in range(args.thread):
        avg_cost += result[0][i]
        total_number += result[2][i]
    avg_cost = avg_cost / args.thread

    print("Total cost: {}s".format(total_cost))
    print("Each thread cost: {}s. ".format(avg_cost))
    print("AVG_QPS: {} samples/s".format(args.batch_size * total_number / avg_cost))
    print("qps(request): {}samples/s".format(total_number / (avg_cost *
                                                             args.thread)))
    print("Total count: {}. ".format(total_number))
    if os.getenv("FLAGS_serving_latency"):
        show_latency(result[1])