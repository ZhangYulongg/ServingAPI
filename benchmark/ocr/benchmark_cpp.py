# -*- coding: utf-8 -*-
from __future__ import unicode_literals, absolute_import
import os
import sys
import time
import json
import requests
import numpy as np
import base64
from paddle_serving_client import Client
from paddle_serving_client.utils import MultiThreadRunner
from paddle_serving_client.utils import benchmark_args, show_latency
from paddle_serving_app.reader import Sequential, File2Image, Resize, CenterCrop
from paddle_serving_app.reader import RGB2BGR, Transpose, Div, Normalize

args = benchmark_args()


def cv2_to_base64(image):
    return base64.b64encode(image) #data.tostring()).decode('utf8')


image_file = "imgs/1.jpg"
with open(image_file, 'rb') as file:
    image_data = file.read()
image = cv2_to_base64(image_data)


def single_func(idx, resource):
    total_number = 0
    latency_list = []

    client = Client()
    client.load_client_config(["ocr_det_client", "ocr_rec_client"])
    client.connect(resource["endpoint"])
    start = time.time()

    while True:
        l_start = time.time()
        result = client.predict(
            feed={"image": image},
            fetch=["ctc_greedy_decoder_0.tmp_0", "softmax_0.tmp_0"],
            batch=True)
        l_end = time.time()
        # o_start = time.time()
        latency_list.append(l_end * 1000 - l_start * 1000)
        total_number = total_number + 1
        # o_end = time.time()
        if time.time() - start > 20:
            break

    end = time.time()
    return [[end - start], latency_list, [total_number]]


if __name__ == '__main__':
    multi_thread_runner = MultiThreadRunner()
    endpoint_list = ["127.0.0.1:9293"]
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