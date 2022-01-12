# -*- coding: utf-8 -*-
from __future__ import unicode_literals, absolute_import
import os
import sys
import time
import json
import base64
import requests
import numpy as np
from paddle_serving_server.pipeline import PipelineClient
from paddle_serving_client.utils import MultiThreadRunner
from paddle_serving_client.utils import benchmark_args, show_latency
from paddle_serving_app.reader import Sequential, File2Image, Resize, CenterCrop
from paddle_serving_app.reader import RGB2BGR, Transpose, Div, Normalize

args = benchmark_args()


def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')


with open("imgs/1.jpg", 'rb') as file:
    image_data = file.read()
image = cv2_to_base64(image_data)
feed_dict = {"image": image}
fetch = ["res"]
# http request
keys, values = [], []
for i in range(args.batch_size):
    keys.append("image_{}".format(i))
    values.append(image)
data = {"key": keys, "value": values}


def run_rpc(idx, resource):
    total_number = 0
    latency_list = []

    client = PipelineClient()
    client.connect(resource["endpoint"])
    start = time.time()
    for i in range(resource["turns"]):
        if args.batch_size >= 1:
            l_start = time.time()
            result = client.predict(
                feed_dict=feed_dict,
                fetch=fetch)
            # print(result)
            l_end = time.time()
            # o_start = time.time()
            latency_list.append(l_end * 1000 - l_start * 1000)
            total_number = total_number + 1
            # o_end = time.time()
            if time.time() - start > 20:
                break
        else:
            print("unsupport batch size {}".format(args.batch_size))

    end = time.time()
    return [[end - start], latency_list, [total_number]]


def run_http(idx, resource):
    total_number = 0
    latency_list = []
    # url = "http://127.0.0.1:9999/ocr/prediction"
    url = "http://127.0.0.1:9998/ocr/prediction"

    start = time.time()
    while True:
        l_start = time.time()
        r = requests.post(url=url, data=json.dumps(data))
        print(r.json())
        l_end = time.time()
        latency_list.append(l_end * 1000 - l_start * 1000)
        total_number += 1
        if time.time() - start > 20:
            break
    end = time.time()
    return [[end - start], latency_list, [total_number]]


if __name__ == '__main__':
    multi_thread_runner = MultiThreadRunner()
    # endpoint_list = ["127.0.0.1:18090"]
    endpoint_list = ["127.0.0.1:9999"]
    turns = 100
    start = time.time()
    if args.request == "rpc":
        result = multi_thread_runner.run(run_rpc, args.thread, {"endpoint": endpoint_list, "turns": turns})
    elif args.request == "http":
        result = multi_thread_runner.run(run_http, args.thread, {"endpoint": endpoint_list, "turns": turns})
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