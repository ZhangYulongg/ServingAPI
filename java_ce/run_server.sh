#!/bin/bash

# 进入serving目录
serving_dir=${CODE_PATH}/Serving/
py_version=python3.6
# 日志目录
log_dir=${CODE_PATH}/logs/
demo_dir=${CODE_PATH}/Serving/examples/
# 数据目录
data=/mnt/serving/dataset/data/
# 输出颜色
RED_COLOR='\E[1;31m'  #红
GREEN_COLOR='\E[1;32m' #绿
YELOW_COLOR='\E[1;33m' #黄
RES='\E[0m'
# 异常词
error_words="Fail|DENIED|None"

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
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddlepaddle_gpu-2.2.0.post101-cp36-cp36m-linux_x86_64.whl
    elif [ $1 == 37 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddlepaddle_gpu-2.2.0.post101-cp37-cp37m-linux_x86_64.whl
    elif [ $1 == 38 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddlepaddle_gpu-2.2.0.post101-cp38-cp38-linux_x86_64.whl
    fi
  elif [ $2 == 1027 ]; then
    if [ $1 == 36 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.0/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.0-cp36-cp36m-linux_x86_64.whl
    elif [ $1 == 37 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.0/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.0-cp37-cp37m-linux_x86_64.whl
    elif [ $1 == 38 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.0/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.0-cp38-cp38-linux_x86_64.whl
    fi
  elif [ $2 == 1028 ]; then
    if [ $1 == 36 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.0-cp36-cp36m-linux_x86_64.whl
    elif [ $1 == 37 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.0-cp37-cp37m-linux_x86_64.whl
    elif [ $1 == 38 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.0-cp38-cp38-linux_x86_64.whl
    fi
  elif [ $2 == 112 ]; then
    if [ $1 == 36 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.0/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.0.post112-cp36-cp36m-linux_x86_64.whl
    elif [ $1 == 37 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.0/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.0.post112-cp37-cp37m-linux_x86_64.whl
    elif [ $1 == 38 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.0/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.0.post112-cp38-cp38-linux_x86_64.whl
    fi
  elif [ $2 == "cpu" ]; then
    if [ $1 == 36 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.0/linux/linux-cpu-mkl-avx/paddlepaddle-2.2.0-cp36-cp36m-linux_x86_64.whl
    elif [ $1 == 37 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.0/linux/linux-cpu-mkl-avx/paddlepaddle-2.2.0-cp37-cp37m-linux_x86_64.whl
    elif [ $1 == 38 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.0/linux/linux-cpu-mkl-avx/paddlepaddle-2.2.0-cp38-cp38-linux_x86_64.whl
    fi
  else
    echo -e "${RED_COLOR}Error cuda version$1${RES}"
    exit
  fi
  $py_version -m pip install paddlepaddle* -i https://mirror.baidu.com/pypi/simple
  rm -rf paddlepaddle*
  set_proxy
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
  elif [ $2 == 1027 ]; then
    whl_list[2]=server_gpu-${version}.post102-py3
  elif [ $2 == 1028 ]; then
    whl_list[2]=server_gpu-${version}.post1028-py3
  elif [ $2 == 112 ]; then
    whl_list[2]=server_gpu-${version}.post112-py3
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

# 起server
# 链接模型数据
function link_data () {
  for file in $1*
  do
  	if [ ! -h ${file##*/} ]
  	then
  	  ln -s ${file} ./${file##*/}
  	fi
  done
}

function prepare () {
  # 准备
  dir=${log_dir}$2/$1/
  enter_dir $1/
  check_dir ${dir}
}

# 进入目录
function enter_dir () {
  cd ${demo_dir}$1
  pwd
}

# 判断目录是否存在
function check_dir () {
  if [ ! -d "$1" ]
  then
  	mkdir -p $1
  fi
}

# kill进程
function kill_process () {
  kill `ps -ef | grep serving | awk '{print $2}'` > /dev/null 2>&1
  kill `ps -ef | grep python | awk '{print $2}'` > /dev/null 2>&1
  echo -e "${GREEN_COLOR}process killed...${RES}"
}

# check命令并保存日志
function check_save () {
  if [ $? == 0 ]
  then
  	echo -e "${GREEN_COLOR}$1 execute normally${RES}"
  	# server端
  	if [ $1 == "server" ]
  	then
      sleep $2
      cat ${dir}server_log.txt | tee -a ${log_dir}server_total.txt
    fi
    # client端
    if [ $1 == "client" ]
    then
      cat ${dir}client_log.txt | tee -a ${log_dir}client_total.txt
      # 检查日志异常词
      grep -E "${error_words}" ${dir}client_log.txt > /dev/null
      if [ $? == 0 ]; then
        # 包含关键词，预测报错
        echo -e "${RED_COLOR}$1 error command${RES}\n" | tee -a ${log_dir}server_total.txt ${log_dir}client_total.txt
        # 记录模型、部署方式、环境信息
        error_log $2
      else
        # 不含关键词，正常预测
        echo -e "${GREEN_COLOR}$2${RES}\n" | tee -a ${log_dir}server_total.txt ${log_dir}client_total.txt
      fi
    fi
  else
    echo -e "${RED_COLOR}$1 error command${RES}\n" | tee -a ${log_dir}server_total.txt ${log_dir}client_total.txt
    cat ${dir}client_log.txt | tee -a ${log_dir}client_total.txt
    # 记录模型、部署方式、环境信息
    error_log $2
  fi
}

# 提取错误日志
function error_log () {
  # 记录模型、部署方式、环境信息
  arg=${1//\//_} # 将所有/替换为_
  echo "-----------------------------" | tee -a ${log_dir}error_models.txt
  arg=${arg%% *}
  arr=(${arg//_/ })
  if [ ${arr[@]: -1} == 1 -o ${arr[@]: -1} == 2 ]; then
    model=${arr[*]:0:${#arr[*]}-3}
    deployment=${arr[*]: -3}
  else
    model=${arr[*]:0:${#arr[*]}-2}
    deployment=${arr[*]: -2}
  fi
  echo "model: ${model// /_}" | tee -a ${log_dir}error_models.txt
  echo "deployment: ${deployment// /_}" | tee -a ${log_dir}error_models.txt
  echo "py_version: ${py_version}" | tee -a ${log_dir}error_models.txt
  echo "cuda_version: ${cuda_version}" | tee -a ${log_dir}error_models.txt
  echo "status: Failed" | tee -a ${log_dir}error_models.txt
  echo -e "-----------------------------\n\n" | tee -a ${log_dir}error_models.txt
  # 日志汇总
  prefix=${arg//\//_}
  for file in ${dir}*
  do
  	cp ${file} ${log_dir}error/${prefix}_${file##*/}
  done
}

function pre_loading() {
  cd ${demo_dir}/C++/PaddleNLP/bert
  # 链接模型数据
  data_dir=${data}bert/
  link_data ${data_dir}
  if [ $1 == "cpu" ]; then
    $py_version -m paddle_serving_server.serve --model bert_seq128_model/ --port 9200 > ${log_dir}/cpu_pre_loading.txt 2>&1 &
    n=1
    while [ ${n} -le 12 ]; do
      sleep 10
      tail ${log_dir}/cpu_pre_loading.txt
      # 检查日志
      echo "-----check for next step n=${n}-----"
      tail ${log_dir}/cpu_pre_loading.txt | grep -E "ir_graph_to_program_pass" ${log_dir}/cpu_pre_loading.txt > /dev/null
      if [ $? == 0 ]; then
        # 预加载成功
        echo "----------cpu_server loaded----------"
        head ${log_dir}/cpu_pre_loading.txt
        break
      elif [ $n == 12 ]; then
        cat ${log_dir}/cpu_pre_loading.txt | grep -E "failed|Error" ${log_dir}/cpu_pre_loading.txt > /dev/null
        if [ $? == 0 ]; then
          # 下载出错
          exit 1
        fi
        n=10
      fi
      n=$((n + 1))
    done
    head ${log_dir}/cpu_pre_loading.txt
    echo "----------cpu_server loaded----------"
#    kill_process
  fi
}

function pipeline_imdb_model_ensemble_cpu_pipeline () {
	# 准备
  dir=${log_dir}java/python_pipeline_server/imdb_model_ensemble/
  enter_dir Pipeline/imdb_model_ensemble/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}imdb/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}pipeline_imdb_model_ensemble_CPU_PIPELINE server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model imdb_cnn_model --port 9292 > ${dir}cnn_log.txt 2>&1 &
  # 单独检查
  if [ $? == 0 ]
  then
  	echo -e "${GREEN_COLOR}pipeline_imdb_model_ensemble_CNN_CPU_RPC execute normally${RES}"
  	sleep 5
    cat ${dir}cnn_log.txt | tee -a ${log_dir}server_total.txt
  fi
  $py_version -m paddle_serving_server.serve --model imdb_bow_model --port 9393 > ${dir}bow_log.txt 2>&1 &
  if [ $? == 0 ]
  then
  	echo -e "${GREEN_COLOR}pipeline_imdb_model_ensemble_BOW_CPU_RPC execute normally${RES}"
  	sleep 5
    cat ${dir}bow_log.txt | tee -a ${log_dir}server_total.txt
  fi
  $py_version test_pipeline_server.py > ${dir}server_log.txt 2>&1 &
  if [ $? == 0 ]
  then
  	echo -e "${GREEN_COLOR}pipeline_imdb_model_ensemble_PIPELINE_SERVER execute normally${RES}"
  	sleep 5
    cat ${dir}bow_log.txt | tee -a ${log_dir}server_total.txt
  fi
}

function pipeline_simple_web_service_cpu_pipeline () {
	# 准备
  dir=${log_dir}java/brpc_server/simple_web_service/
  enter_dir Pipeline/simple_web_service/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}fit_a_line/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}pipeline_simple_web_service_CPU_PIPELINE server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version web_service_java.py > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
}

function brpc_server_fit_a_line_cpu() {
    dir=${log_dir}java/python_pipeline_server/simple_web_service/
    enter_dir C++/fit_a_line/
    check_dir ${dir}
    data_dir=${data}fit_a_line/
    link_data ${data_dir}
    # 启动服务
    echo -e "${GREEN_COLOR}brpc_server_fit_a_line_GPU_BRPC server started${RES}" | tee -a ${log_dir}server_total.txt
    $py_version -m paddle_serving_server.serve --model uci_housing_model --thread 10 --port 9898 > ${dir}server_log.txt 2>&1 &
    # 命令检查
    check_save server 10
}

unset http_proxy
unset https_proxy
# 设置py版本
set_py $1
check_dir ${log_dir}
pre_loading $2
pipeline_imdb_model_ensemble_cpu_pipeline
pipeline_simple_web_service_cpu_pipeline
brpc_server_fit_a_line_cpu

netstat -nlp

sleep 1000
