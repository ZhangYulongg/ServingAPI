import os
import pynvml


def kill_process(port):
    command = "kill -9 $(netstat -nlp | grep :"+str(port)+" | awk '{print $7}' | awk -F'/' '{{ print $1 }}')"
    os.system(command)
    # 解决端口占用
    os.system("ls > /dev/null 2>&1")


def check_gpu_memory(gpu_id):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    mem_used = mem_info.used / 1024 ** 2
    print(f"GPU-{gpu_id} memory used:", mem_used)
    return mem_used > 100
