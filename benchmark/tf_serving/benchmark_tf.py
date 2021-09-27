# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=doc-string-missing

from __future__ import unicode_literals, absolute_import, print_function
import os
import sys
import time
import requests
import json
import base64
from paddle_serving_client import Client
from paddle_serving_client.utils import MultiThreadRunner
from paddle_serving_client.utils import benchmark_args, show_latency
from paddle_serving_app.reader import Sequential, File2Image, Resize
from paddle_serving_app.reader import CenterCrop, RGB2BGR, Transpose, Div, Normalize

import grpc
import requests
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import numpy as np

args = benchmark_args()

seq_preprocess = Sequential([
    File2Image(), Resize(256), CenterCrop(224), RGB2BGR(), Transpose((2, 0, 1)),
    Div(255), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True)
])


img_file = "cat.jpg"
img = seq_preprocess(img_file)
img = img.transpose((1, 2, 0))

endpoint_list = [
    "127.0.0.1:8500",
]
channel = grpc.insecure_channel(endpoint_list[0])

def single_func(idx, resource):
    turns = resource["turns"]
    total_number = 0
    latency_flags = False
    if os.getenv("FLAGS_serving_latency"):
        latency_flags = True
        latency_list = []
    profile_flags = False
    if "FLAGS_profile_client" in os.environ and os.environ[
            "FLAGS_profile_client"]:
        profile_flags = True

    if args.request == "rpc":
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'serving_default'
        request.model_spec.signature_name = 'serving_default'

        batch = []
        for i in range(args.batch_size):
            batch.append(img)

        request.inputs['input'].CopyFrom(
            tf.make_tensor_proto(np.array(batch), shape=np.array(batch).shape))

        start = time.time()

        for i in range(turns):
            if args.batch_size >= 1:
                l_start = time.time()
                result = stub.Predict(request, 30.0)
                l_end = time.time()
                if latency_flags:
                    latency_list.append(l_end * 1000 - l_start * 1000)
                total_number = total_number + 1
            else:
                print("unsupport batch size {}".format(args.batch_size))

    end = time.time()
    if latency_flags:
        return [[end - start], latency_list, [total_number]]
    return [[end - start]]


if __name__ == '__main__':
    multi_thread_runner = MultiThreadRunner()
    turns = 10
    start = time.time()
    result = multi_thread_runner.run(
            single_func, args.thread, {"turns": turns})
    #result = single_func(0, {"endpoint": endpoint_list})
    end = time.time()
    total_cost = end - start
    total_number = 0
    avg_cost = 0
    for i in range(args.thread):
        avg_cost += result[0][i]
        total_number += result[2][i]
    avg_cost = avg_cost / args.thread
    print("Total cost: {}s".format(end - start))
    print("Each thread cost: {}s.".format(avg_cost))
    print("AVG_QPS: {} samples/s".format(args.batch_size * total_number /
                                    avg_cost))
    print("Total count: {}.".format(total_number))
    if os.getenv("FLAGS_serving_latency"):
        show_latency(result[1])