#!/bin/bash

# 进入serving目录
serving_dir=${CODE_PATH}/Serving/
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

function set_proxy() {
  echo -e "${YELOW_COLOR}---------set proxy---------${RES}"
  export https_proxy=${proxy}
  export http_proxy=${proxy}
}

function unset_proxy() {
  echo -e "${YELOW_COLOR}---------unset proxy---------${RES}"
  unset http_proxy
  unset https_proxy
}

# 设置python版本
function set_py () {
  if [ $1 == 36 ]; then
    py_version="python3.6"
    pip_version="pip3.6"
  elif [ $1 == 37 ]; then
    py_version="python3.7"
    pip_version="pip3.7"
  elif [ $1 == 38 ]; then
    py_version="python3.8"
    pip_version="pip3.8"
  else
    echo -e "${RED_COLOR}Error py version$1${RES}"
    exit
  fi
}

# 安装python依赖包
function py_requirements () {
  echo -e "${YELOW_COLOR}---------install python requirements---------${RES}"
  echo "---------Python Version: $py_version"
  set_proxy
  $py_version -m pip install --upgrade pip==21.1.3
  unset_proxy
  $py_version -m pip install -r python/requirements.txt -i https://mirror.baidu.com/pypi/simple
  $py_version -m pip install paddlehub -i https://mirror.baidu.com/pypi/simple
  if [ $2 == 101 ]; then
    if [ $1 == 36 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddlepaddle_gpu-2.1.0.post101-cp36-cp36m-linux_x86_64.whl > /dev/null 2>&1
    elif [ $1 == 37 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddlepaddle_gpu-2.1.0.post101-cp37-cp37m-linux_x86_64.whl > /dev/null 2>&1
    elif [ $1 == 38 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddlepaddle_gpu-2.1.0.post101-cp38-cp38-linux_x86_64.whl > /dev/null 2>&1
    fi
  elif [ $2 == 102 ]; then
    if [ $1 == 36 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda10.2-cudnn8-mkl-gcc8.2/paddlepaddle_gpu-2.1.0-cp36-cp36m-linux_x86_64.whl > /dev/null 2>&1
    elif [ $1 == 37 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda10.2-cudnn8-mkl-gcc8.2/paddlepaddle_gpu-2.1.0-cp37-cp37m-linux_x86_64.whl > /dev/null 2>&1
    elif [ $1 == 38 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda10.2-cudnn8-mkl-gcc8.2/paddlepaddle_gpu-2.1.0-cp38-cp38-linux_x86_64.whl > /dev/null 2>&1
    fi
  elif [ $2 == 110 ]; then
    if [ $1 == 36 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda11.0-cudnn8-mkl-gcc8.2/paddlepaddle_gpu-2.1.0.post110-cp36-cp36m-linux_x86_64.whl > /dev/null 2>&1
    elif [ $1 == 37 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda11.0-cudnn8-mkl-gcc8.2/paddlepaddle_gpu-2.1.0.post110-cp37-cp37m-linux_x86_64.whl > /dev/null 2>&1
    elif [ $1 == 38 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda11.0-cudnn8-mkl-gcc8.2/paddlepaddle_gpu-2.1.0.post110-cp38-cp38-linux_x86_64.whl > /dev/null 2>&1
    fi
  elif [ $2 == "cpu" ]; then
    if [ $1 == 36 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.1.0-cpu-avx-mkl/paddlepaddle-2.1.0-cp36-cp36m-linux_x86_64.whl > /dev/null 2>&1
    elif [ $1 == 37 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.1.0-cpu-avx-mkl/paddlepaddle-2.1.0-cp37-cp37m-linux_x86_64.whl > /dev/null 2>&1
    elif [ $1 == 38 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.1.0-cpu-avx-mkl/paddlepaddle-2.1.0-cp38-cp38-linux_x86_64.whl > /dev/null 2>&1
    fi
  else
    echo -e "${RED_COLOR}Error cuda version$1${RES}"
    exit
  fi
  set_proxy
  $py_version -m pip install paddlepaddle*
  rm -rf paddlepaddle*
  echo -e "${YELOW_COLOR}---------complete---------\n${RES}"
}

function pip_install_serving() {
  unset_proxy
  if [ ${test_branch} == "develop" ]; then
    version=0.0.0
  else
    version=${test_branch: 1}
  fi
  # whl包数组
  whl_list=()
  whl_list[0]=app-${version}-py3
  if [ $1 == 36 ]; then
    whl_list[1]=client-${version}-cp36
  elif [ $1 == 37 ]; then
    whl_list[1]=client-${version}-cp37
  elif [ $1 == 38 ]; then
    whl_list[1]=client-${version}-cp38
  fi
  if [ $2 == 101 ]; then
    whl_list[2]=server_gpu-${version}.post101-py3
  elif [ $2 == 102 ]; then
    whl_list[2]=server_gpu-${version}.post102-py3
  elif [ $2 == 110 ]; then
    whl_list[2]=server_gpu-${version}.post11-py3
  elif [ $2 == "cpu" ]; then
    whl_list[2]=server-${version}-py3
  fi
  echo "----------whl_list: "
  echo ${whl_list[*]}
  cd ${CODE_PATH}
  rm -rf whl_packages
  mkdir whl_packages && cd whl_packages
  echo "----------cur path: `pwd`"
  for whl_item in ${whl_list[@]}
  do
      wget -q https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_${whl_item}-none-any.whl
      if [ $? -eq 0 ]; then
          echo "--------------download ${whl_item} succ"
      else
          echo "--------------download ${whl_item} failed"
      fi
  done
  if [ $1 == 38 ]; then
    $py_version -m pip install sentencepiece
  fi
  set_proxy
  $py_version -m pip install *
}

function pip_install_test() {
  cd ${CODE_PATH}
  rm -rf whl_packages
  mkdir whl_packages && cd whl_packages
  wget -q https://paddle-serving.bj.bcebos.com/test-dev/whl/0.6.1/paddle_serving_app-0.6.1-cp36-cp36m-linux_x86_64.whl
  if [ $1 == 36 ]; then
      wget -q https://paddle-serving.bj.bcebos.com/test-dev/whl/0.6.1/paddle_serving_client-0.6.1-cp36-cp36m-linux_x86_64.whl
  elif [ $1 == 37 ]; then
      wget -q https://paddle-serving.bj.bcebos.com/test-dev/whl/0.6.1/paddle_serving_client-0.6.1-cp37-cp37m-linux_x86_64.whl
  elif [ $1 == 38 ]; then
      wget -q https://paddle-serving.bj.bcebos.com/test-dev/whl/0.6.1/paddle_serving_client-0.6.1-cp38-cp38-linux_x86_64.whl
  fi
  if [ $2 == "cpu" ]; then
      wget -q https://paddle-serving.bj.bcebos.com/test-dev/whl/0.6.1/paddle_serving_server-0.6.1-cp36-cp36m-linux_x86_64.whl
  elif [ $2 == 101 ]; then
      wget -q https://paddle-serving.bj.bcebos.com/test-dev/whl/0.6.1/paddle_serving_server_gpu-0.6.1.post101-cp36-cp36m-linux_x86_64.whl
  elif [ $2 == 102 ]; then
      wget -q https://paddle-serving.bj.bcebos.com/test-dev/whl/0.6.1/paddle_serving_server_gpu-0.6.1.post102-cp36-cp36m-linux_x86_64.whl
  elif [ $2 == 110 ]; then
      wget -q https://paddle-serving.bj.bcebos.com/test-dev/whl/0.6.1/paddle_serving_server_gpu-0.6.1.post11-cp36-cp36m-linux_x86_64.whl
  fi
  set_proxy
  $py_version -m pip install *
}

cd $serving_dir
set_proxy
git submodule update --init --recursive
echo "-----------cur path: `pwd`"
echo -e "${GREEN_COLOR}-----------env lists: ${RES}"
env | grep -E "PYTHONROOT|PYTHON_INCLUDE_DIR|PYTHON_LIBRARIES|PYTHON_EXECUTABLE|CUDA_PATH|CUDA_PATH|CUDNN_LIBRARY|CUDA_CUDART_LIBRARY|TENSORRT_LIBRARY_PATH"

set_py $1
py_requirements $1 $2
echo "--------pip list: "
${pip_version} list

pip_install_serving $1 $2

echo "--------pip list after pip: "
${pip_version} list
# whl包检查
if [ `${pip_version} list | grep -c paddlepaddle` != 1 ]; then
    py_requirements $1 $2
    if [ `${pip_version} list | grep -c paddlepaddle` != 1 ]; then
        echo "----------paddle install failed!----------"
        exit 2
    fi
fi

if [ `${pip_version} list | egrep "paddle-serving" | wc -l` -eq 3 ]; then
    echo "-----------whl_packages succ"
else
    echo "----------whl_packages failed"
    pip_install_serving $1 $2
    if [ `${pip_version} list | egrep "paddle-serving" | wc -l` -eq 3 ]; then
       echo "-----------whl_packages succ ---2"
    else
        echo "----------whl_packages failed ---2"
        exit 1
    fi
fi
