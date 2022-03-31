#!/bin/bash

# 进入serving目录
serving_dir=${CODE_PATH}/Serving/
py_version=python3.9
# 日志目录
log_dir=${CODE_PATH}/logs/
demo_dir=${CODE_PATH}/Serving/examples/
# 数据目录
data=${DATA_PATH}
# 输出颜色
RED_COLOR='\E[1;31m'  #红
GREEN_COLOR='\E[1;32m' #绿
YELOW_COLOR='\E[1;33m' #黄
RES='\E[0m'
# 异常词
error_words="Fail|DENIED|None|empty|failed|not match"

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

function pip_install_serving () {
  cd ${serving_dir}
  unset_proxy
  $py_version -m pip install app_build/python/dist/paddle_serving* -i https://mirror.baidu.com/pypi/simple
  $py_version -m pip install client-build/python/dist/paddle_serving* -i https://mirror.baidu.com/pypi/simple
  $py_version -m pip install server-build-gpu-opencv/python/dist/paddle_serving* -i https://mirror.baidu.com/pypi/simple
}

# 安装python依赖包
function py_requirements () {
  cd $serving_dir
  echo -e "${YELOW_COLOR}---------install python requirements---------${RES}"
  echo "---------Python Version: $py_version"
  set_proxy
  $py_version -m pip install --upgrade pip
  unset_proxy
  $py_version -m pip install -r python/requirements.txt -i https://mirror.baidu.com/pypi/simple
  if [ $2 == 101 ]; then
    if [ $1 == 36 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddlepaddle_gpu-2.2.2.post101-cp36-cp36m-linux_x86_64.whl
    elif [ $1 == 37 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddlepaddle_gpu-2.2.2.post101-cp37-cp37m-linux_x86_64.whl
    elif [ $1 == 38 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddlepaddle_gpu-2.2.2.post101-cp38-cp38-linux_x86_64.whl
    elif [ $1 == 39 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddlepaddle_gpu-2.2.2.post101-cp39-cp39-linux_x86_64.whl
    fi
  elif [ $2 == 1027 ]; then
    if [ $1 == 36 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn7.6.5_trt6.0.1.5/paddlepaddle_gpu-2.2.2-cp36-cp36m-linux_x86_64.whl
    elif [ $1 == 37 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn7.6.5_trt6.0.1.5/paddlepaddle_gpu-2.2.2-cp37-cp37m-linux_x86_64.whl
    elif [ $1 == 38 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn7.6.5_trt6.0.1.5/paddlepaddle_gpu-2.2.2-cp38-cp38-linux_x86_64.whl
    elif [ $1 == 39 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn7.6.5_trt6.0.1.5/paddlepaddle_gpu-2.2.2-cp39-cp39-linux_x86_64.whl
    fi
  elif [ $2 == 1028 ]; then
    if [ $1 == 36 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.2-cp36-cp36m-linux_x86_64.whl
    elif [ $1 == 37 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.2-cp37-cp37m-linux_x86_64.whl
    elif [ $1 == 38 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.2-cp38-cp38-linux_x86_64.whl
    elif [ $1 == 39 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.2-cp39-cp39-linux_x86_64.whl
    fi
  elif [ $2 == 112 ]; then
    if [ $1 == 36 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddlepaddle_gpu-2.2.2.post112-cp36-cp36m-linux_x86_64.whl
    elif [ $1 == 37 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddlepaddle_gpu-2.2.2.post112-cp37-cp37m-linux_x86_64.whl
    elif [ $1 == 38 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddlepaddle_gpu-2.2.2.post112-cp38-cp38-linux_x86_64.whl
    elif [ $1 == 39 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddlepaddle_gpu-2.2.2.post112-cp39-cp39-linux_x86_64.whl
    fi
  elif [ $2 == "cpu" ]; then
    if [ $1 == 36 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.2/linux/linux-cpu-mkl-avx/paddlepaddle-2.2.2-cp36-cp36m-linux_x86_64.whl
    elif [ $1 == 37 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.2/linux/linux-cpu-mkl-avx/paddlepaddle-2.2.2-cp37-cp37m-linux_x86_64.whl
    elif [ $1 == 38 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.2/linux/linux-cpu-mkl-avx/paddlepaddle-2.2.2-cp38-cp38-linux_x86_64.whl
    elif [ $1 == 39 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.2/linux/linux-cpu-mkl-avx/paddlepaddle-2.2.2-cp39-cp39-linux_x86_64.whl
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
      tail ${dir}server_log.txt | tee -a ${log_dir}server_total.txt
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
        echo "$2 failed" >> ${log_dir}/result.txt
      else
        # 不含关键词，正常预测
        echo -e "${GREEN_COLOR}$2${RES}\n" | tee -a ${log_dir}server_total.txt ${log_dir}client_total.txt
      fi
    fi
  else
    echo -e "${RED_COLOR}$1 error command${RES}\n" | tee -a ${log_dir}server_total.txt ${log_dir}client_total.txt
    cat ${dir}client_log.txt | tee -a ${log_dir}client_total.txt
    # 记录模型、部署方式、环境信息
    echo "$2 failed" >> ${log_dir}/result.txt
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
  cd $serving_dir/java/examples/target
  java -cp paddle-serving-sdk-java-examples-0.0.1-jar-with-dependencies.jar PipelineClientExample string_imdb_predict > ${dir}client_log.txt 2>&1
  check_save client "pipeline_imdb_model_ensemble_sync_java_client_CPU_PIPELINE"
  java -cp paddle-serving-sdk-java-examples-0.0.1-jar-with-dependencies.jar PipelineClientExample asyn_predict > ${dir}client_log.txt 2>&1
  check_save client "pipeline_imdb_model_ensemble_asyn_java_client_CPU_PIPELINE"
  kill_process
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
  cd $serving_dir/java/examples/target
  java -cp paddle-serving-sdk-java-examples-0.0.1-jar-with-dependencies.jar PipelineClientExample indarray_predict > ${dir}client_log.txt 2>&1
  check_save client "pipeline_simple_web_service_indarray_sync_java_client_CPU_PIPELINE"
  kill_process
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
    # 拷贝模型配置文件
    cd $serving_dir/java/examples/target
    cp -r ${DATA_PATH}/fit_a_line/uci_housing_client/serving_client_conf.prototxt  ./fit_a_line.prototxt
    java -cp paddle-serving-sdk-java-examples-0.0.1-jar-with-dependencies.jar PaddleServingClientExample http_proto fit_a_line.prototxt > ${dir}client_log.txt 2>&1
    check_save client "fit_a_line_java_http_proto_GPU_BRPC"

    java -cp paddle-serving-sdk-java-examples-0.0.1-jar-with-dependencies.jar PaddleServingClientExample http_json fit_a_line.prototxt > ${dir}client_log.txt 2>&1
    check_save client "fit_a_line_java_http_json_GPU_BRPC"

    java -cp paddle-serving-sdk-java-examples-0.0.1-jar-with-dependencies.jar PaddleServingClientExample grpc fit_a_line.prototxt > ${dir}client_log.txt 2>&1
    check_save client "fit_a_line_java_grpc_GPU_BRPC"

    java -cp paddle-serving-sdk-java-examples-0.0.1-jar-with-dependencies.jar PaddleServingClientExample compress fit_a_line.prototxt > ${dir}client_log.txt 2>&1
    check_save client "fit_a_line_java_compress_GPU_BRPC"
    kill_process
}

function prepare_java_client() {
    cd $serving_dir/java
    set_proxy
    # 修改示例设置端口
    sed -i "s/9393/9898/g" examples/src/main/java/PaddleServingClientExample.java
    mvn compile
    mvn install
    cd examples
    mvn compile
    mvn install
    unset_proxy
}

unset http_proxy
unset https_proxy
# 设置py版本
cd $serving_dir
py_requirements $1 $2
pip_install_serving
check_dir ${log_dir}
prepare_java_client
rm -rf ${log_dir}/result.txt
unset_proxy
env
pipeline_imdb_model_ensemble_cpu_pipeline
pipeline_simple_web_service_cpu_pipeline
brpc_server_fit_a_line_cpu

if [ -f ${log_dir}result.txt ]; then
  cat ${log_dir}result.txt
  exit 1
fi
