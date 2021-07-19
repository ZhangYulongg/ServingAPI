#!/bin/bash
# 输出颜色
RED_COLOR='\E[1;31m'  #红
GREEN_COLOR='\E[1;32m' #绿
YELOW_COLOR='\E[1;33m' #黄
RES='\E[0m'

function check_gpu_memory() {
    gpu_memory=`nvidia-smi --id=$1 --format=csv,noheader --query-gpu=memory.used | awk '{print $1}'`
    echo -e "${GREEN_COLOR}-------id-$1 gpu_memory_used: ${gpu_memory}${RES}\n"
    if [ ${gpu_memory} -le 100 ]; then
        echo "----GPU not used"
        status="GPU not used"
    else
        echo "----GPU_memory used is expected"
    fi
}

check_gpu_memory 0
check_gpu_memory 1
