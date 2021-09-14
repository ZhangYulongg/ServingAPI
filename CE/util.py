import os
import pynvml
import argparse
import base64
import subprocess
import numpy as np


class ServingTest(object):
    def __init__(self, data_path: str, example_path: str, model_dir: str, client_dir: str):
        """
        需设置环境变量
        CODE_PATH: repo上一级目录
        DATA_PATH: 数据集根目录
        py_version: python版本 python3.6~3.8
        """
        code_path = os.environ.get("CODE_PATH")
        self.data_path = f"{os.environ.get('DATA_PATH')}/{data_path}/"
        self.example_path = f"{code_path}/Serving/python/examples/{example_path}/"
        self.py_version = os.environ.get("py_version")
        self.model_dir = model_dir
        self.client_config = f"{client_dir}/serving_client_conf.prototxt"

        self.check_model_data_exist()

    def check_model_data_exist(self):
        if not os.path.exists(f"./{self.model_dir}"):
            # 软链模型数据
            dir_path, dir_names, file_names = next(os.walk(self.data_path))
            for dir_ in dir_names:
                abs_path = os.path.join(dir_path, dir_)
                os.system(f"ln -s {abs_path} {dir_}")
            for file in file_names:
                abs_path = os.path.join(dir_path, file)
                os.system(f"ln -s {abs_path} {file}")

    def start_server_by_shell(self, cmd: str, sleep: int = 5, err="stderr.log", out="stdout.log"):
        self.err = open(err, "w")
        self.out = open(out, "w")
        p = subprocess.Popen(cmd, shell=True, stdout=self.out, stderr=self.err)
        os.system(f"sleep {sleep}")

        print(f"======================{err}======================")
        os.system(f"cat {err}")
        print(f"======================{out}======================")
        os.system(f"cat {out}")
        print("======================================================")

    @staticmethod
    def check_result(result_data: dict, truth_data: dict, batch_size=1, delta=1e-3):
        # flatten
        predict_result = {}
        truth_result = {}
        for key, value in result_data.items():
            predict_result[key] = value.flatten()
        for key, value in truth_data.items():
            truth_result[key] = np.repeat(value, repeats=batch_size, axis=0).flatten()
        # print("预测值:", predict_result)
        # print("真实值:", truth_result)

        # compare
        for key in predict_result.keys():
            for i, data in enumerate(predict_result[key]):
                diff = sig_fig_compare(data, truth_result[key][i])
                assert diff < delta, f"data:{data} truth:{truth_result[key][i]} diff is {diff} > {delta}, index:{i}"

    @staticmethod
    def parse_http_result(output):
        # 转换http client返回的proto格式数据，统一为dict包numpy array
        # todo 仅支持float_data
        result_dict = {}
        if isinstance(output, dict):
            for tensor in output["outputs"][0]["tensor"]:
                result_dict[tensor["alias_name"]] = np.array(tensor["float_data"]).reshape(tensor["shape"])
        else:
            for tensor in output.outputs[0].tensor:
                result_dict[tensor.alias_name] = np.array(tensor.float_data).reshape(tensor.shape)
        return result_dict

    @staticmethod
    def release(keywords="web_service.py"):
        os.system("kill -9 `ps -ef | grep serving | awk '{print $2}'` > /dev/null 2>&1")
        os.system("kill -9 `ps -ef | grep " + keywords + " | awk '{print $2}'` > /dev/null 2>&1")


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


def check_keywords_in_server_log(words: str, filename="stderr.log"):
    p = subprocess.Popen(f"grep '{words}' {filename} | head", shell=True)
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
