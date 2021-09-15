#!/bin/bash

dir=${CODE_PATH}/
py_version=python3.6
# 脚本目录
shell_path=${CODE_PATH}/CE/
# 输出颜色
RED_COLOR='\E[1;31m'  #红
GREEN_COLOR='\E[1;32m' #绿
YELOW_COLOR='\E[1;33m' #黄
RES='\E[0m'

# 获取运行参数
if [ $# == 2 ]
then
  echo -e "${YELOW_COLOR}Selected python version is py$1${RES}"
else
  echo "Usage: `basename $0` first secend"
  echo "first(py_version): 36 37 38"
  echo "second(cuda_version): cpu 101 102 110"
  echo "You provided $# parameters,but 2 are required."
  exit
fi

function set_env () {
  if [ $1 == 36 ]; then
    export PYTHONROOT=/usr/local/
    echo PYTHONROOT=/usr/local/ >> ${dir}env.sh
    echo PYTHON_INCLUDE_DIR=$PYTHONROOT/include/python3.6m >> ${dir}env.sh
    echo PYTHON_LIBRARIES=$PYTHONROOT/lib/libpython3.6m.so >> ${dir}env.sh
    echo PYTHON_EXECUTABLE=$PYTHONROOT/bin/python3.6 >> ${dir}env.sh
    py_version="python3.6"
  elif [ $1 == 37 ]; then
    export PYTHONROOT=/usr/local/
    echo PYTHONROOT=/usr/local/ >> ${dir}env.sh
    echo PYTHON_INCLUDE_DIR=$PYTHONROOT/include/python3.7m >> ${dir}env.sh
    echo PYTHON_LIBRARIES=$PYTHONROOT/lib/libpython3.7m.so >> ${dir}env.sh
    echo PYTHON_EXECUTABLE=$PYTHONROOT/bin/python3.7 >> ${dir}env.sh
    py_version="python3.7"
  elif [ $1 == 38 ]; then
    export PYTHONROOT=/usr/local/
    echo PYTHONROOT=/usr/local/ >> ${dir}env.sh
    echo PYTHON_INCLUDE_DIR=$PYTHONROOT/include/python3.8 >> ${dir}env.sh
    echo PYTHON_LIBRARIES=$PYTHONROOT/lib/libpython3.8.so >> ${dir}env.sh
    echo PYTHON_EXECUTABLE=$PYTHONROOT/bin/python3.8 >> ${dir}env.sh
    py_version="python3.8"
  else
    echo -e "${RED_COLOR}Error py version$1${RES}"
    exit
  fi
  echo CUDA_PATH='/usr/local/cuda' >> ${dir}env.sh
  echo CUDNN_LIBRARY='/usr/local/cuda/lib64/' >> ${dir}env.sh
  echo CUDA_CUDART_LIBRARY="/usr/local/cuda/lib64/" >> ${dir}env.sh
  if [ $2 == 101 ]; then
    echo TENSORRT_LIBRARY_PATH="/usr/local/TensorRT6-cuda10.1-cudnn7/targets/x86_64-linux-gnu/" >> ${dir}env.sh
  elif [ $2 == 102 ]; then
    echo TENSORRT_LIBRARY_PATH="/usr/local/TensorRT-7.1.3.4/targets/x86_64-linux-gnu/" >> ${dir}env.sh
  elif [ $2 == 110 ]; then
    echo TENSORRT_LIBRARY_PATH="/usr/local/TensorRT-7.1.3.4/targets/x86_64-linux-gnu/" >> ${dir}env.sh
  elif [ $2 == "cpu" ]; then
    echo TENSORRT_LIBRARY_PATH="/usr/local/TensorRT6-cuda9.0-cudnn7/targets/x86_64-linux-gnu" >> ${dir}env.sh
  fi
}

rm -rf ${dir}/env.sh
set_env $1 $2
cd ${dir}
echo -e "${GREEN_COLOR}-----------env.sh: ${RES}"
cat env.sh
