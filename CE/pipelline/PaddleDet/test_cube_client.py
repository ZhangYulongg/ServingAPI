from paddle_serving_client import Client
import sys
import os
import time
import random
import numpy as np
from collections import Counter

client = Client()
client.load_client_config(sys.argv[1])
client.connect(["127.0.0.1:9494"])

def file_2_dict(filename="", mydict={}):
    line_count = 0
    with open(filename, "r+", encoding='utf-8') as f:
        for line in f.readlines():
            line_count += 1
            line_split = line[:-1].split(' , ', 1)
            if len(line_split) != 2:
                print("Invalid line! line number:{}".format(line_count))
                break

            key_word = line_split[0].split(': ', 1)
            if len(key_word) != 2:
                print("Invalid key word! line number:{}, key_word:{}".format(line_count, key_word))
                continue
            key = key_word[1]
            #print("read key:{}".format(key))

            value_word = line_split[1].split(": ", 1)
            if len(value_word) != 2:
                print("Invalid value word! line number:{} value_word:{}".format(line_count, value_word))
                continue
            value = value_word[1]
            #print("read value:{}".format(value))
            #set key/value to mydict
            mydict[key] = value
            if line_count > 50000:
                break

def gen_rand_keys(mydict, count):
    int64_keys = []
    rand_keys = random.sample(mydict.keys(), count)
    #print("generate random keys:{}".format(rand_keys))
    #for one_key in rand_keys:
    #    int64_keys.append(int64(one_key))
    #print("after int64 translated, random keys:{}".format(int64_keys))
    return rand_keys

cube_filename = "cube_file"
cache_filename = "cache_file"
cache_dict = {}
cube_dict = {}

# Reading files
print("Reading file {} ...".format(cube_filename))
file_2_dict(cube_filename, cube_dict)
print("Reading done! cube dict len:{}\n".format(len(cube_dict)))

print("Reading file {} ...".format(cache_filename))
file_2_dict(cache_filename, cache_dict)
print("Reading done! cache dict len:{}\n".format(len(cache_dict)))

print(len(cube_dict.keys()), len(cache_dict.keys()))

# Assembling Tensors
feed_dict = {}
feed_len = 72
total_keys = int(sys.argv[2])
cache_ratio = float(sys.argv[3])
cache_keys = int(total_keys * cache_ratio)
cube_keys = int(total_keys * (1-cache_ratio))

rand_keys = gen_rand_keys(cache_dict, cache_keys) + gen_rand_keys(cube_dict, cube_keys)

for i in range(feed_len):
    feed_key = "embedding_{}.tmp_0".format(i % feed_len)
    feed_dict[feed_key] = []

for i, a_key in enumerate(rand_keys):
    feed_key = "embedding_{}.tmp_0".format(i % feed_len)
    int64_key = int(a_key)
    feed_dict[feed_key].append(int64_key)

for i in range(feed_len):
    feed_key = "embedding_{}.tmp_0".format(i % feed_len)
    feed_key_len = len(feed_dict[feed_key])
    feed_lod = [0, feed_key_len]
    feed_dict[feed_key] = np.array(feed_dict[feed_key], dtype=np.int64).reshape(feed_key_len, 1)
    feed_dict["{}.lod".format(feed_key)] = feed_lod

print("feed_dict: {}".format(feed_dict))

latency_list = []
result_list = []
turns = 5000
for i in range(turns):
    start = time.time()
    fetch_map = client.predict(feed=feed_dict, fetch=["join_similarity_norm.tmp_0"], batch=True)
    end = time.time()
    latency_list.append(end - start)
    print(fetch_map)
    result = fetch_map["prob"][0, 0]
    result_list.append(result)

latency_array = np.array(latency_list)

print(Counter(result_list))
result_list = np.array(result_list)
failed_nums = np.sum(result_list == 0)
print(f"turns: {turns}")
print(f"total keys: {total_keys}")
print(f"cache ratio: {cache_ratio}")
print(f"mean: {np.mean(latency_array) * 1000:.2f} ms")
print(f"90 %: {np.percentile(latency_array, 90) * 1000:.2f} ms")
print(f"99 %: {np.percentile(latency_array, 99) * 1000:.2f} ms")
print(f"Timeout rate: {failed_nums / turns * 100:.2f}%")
print(f"Success rate: {(turns - failed_nums) / turns * 100:.2f}%")
