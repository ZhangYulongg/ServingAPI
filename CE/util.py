import os
import pynvml
import argparse
import base64
import subprocess


def kill_process(port, sleep_time=0):
    command = "kill -9 $(netstat -nlp | grep :" + str(port) + " | awk '{print $7}' | awk -F'/' '{{ print $1 }}')"
    os.system(command)
    # 解决端口占用
    os.system(f"sleep {sleep_time}")


def check_gpu_memory(gpu_id):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    mem_used = mem_info.used / 1024 ** 2
    print(f"GPU-{gpu_id} memory used:", mem_used)
    return mem_used > 100


def count_process_num_on_port(port):
    command = "netstat -nlp | grep :" + str(port) + " | wc -l"
    count = eval(os.popen(command).read())
    print(f"port-{port} processes num:", count)
    return count


def check_keywords_in_server_log(words: str):
    p = subprocess.Popen(f"grep '{words}' stderr.log", shell=True)
    p.wait()
    return p.returncode == 0


def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')


def sig_fig_compare(num0, num1, delta=5):
    difference = num0 - num1
    num0_int_length = len(str(int(num0)))
    num1_int_length = len(str(int(num1)))
    num0_int = int(num0)
    num1_int = int(num1)
    if num0 < 1 and num1 < 1 and difference < 1:
        return difference
    elif num0_int_length == num1_int_length:
        if num0_int_length >= 5:
            return abs(num0_int - num1_int)
        else:
            scale = 5 - num1_int_length
            num0_padding = num0 * scale
            num1_padding = num1 * scale
            return abs(num0_padding - num1_padding) / (10 * scale)
    elif num0_int_length != num1_int_length:
        return difference


def default_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    args.thread = 2
    args.port = 9292
    args.device = "cpu"
    args.gpu_ids = [""]
    args.op_num = 0
    args.op_max_batch = 32
    args.model = [""]
    args.workdir = "workdir"
    args.use_mkl = False
    args.precision = "fp32"
    args.use_calib = False
    args.mem_optim_off = False
    args.ir_optim = False
    args.max_body_size = 512 * 1024 * 1024
    args.use_encryption_model = False
    args.use_multilang = False
    args.use_trt = False
    args.use_lite = False
    args.use_xpu = False
    args.product_name = None
    args.container_id = None
    args.gpu_multi_stream = False
    return args
