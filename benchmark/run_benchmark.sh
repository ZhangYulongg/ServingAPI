#!/bin/bash

# 日志目录
log_dir=${CODE_PATH}/logs/
demo_dir=${CODE_PATH}/Serving/python/examples/
shell_dir=${CODE_PATH}/benchmark/
dir=`pwd`
export SERVING_BIN=${CODE_PATH}/Serving/server-build-gpu-opencv/core/general-server/serving
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
    export PYTHONROOT=/usr/local/
    export PYTHON_INCLUDE_DIR=$PYTHONROOT/include/python3.6m
    export PYTHON_LIBRARIES=$PYTHONROOT/lib/libpython3.6m.so
    export PYTHON_EXECUTABLE=$PYTHONROOT/bin/python3.6
    py_version="python3.6"
  elif [ $1 == 37 ]; then
    export PYTHONROOT=/usr/local/
    export PYTHON_INCLUDE_DIR=$PYTHONROOT/include/python3.7m
    export PYTHON_LIBRARIES=$PYTHONROOT/lib/libpython3.7m.so
    export PYTHON_EXECUTABLE=$PYTHONROOT/bin/python3.7
    py_version="python3.7"
  elif [ $1 == 38 ]; then
    export PYTHONROOT=/usr/local/
    export PYTHON_INCLUDE_DIR=$PYTHONROOT/include/python3.8
    export PYTHON_LIBRARIES=$PYTHONROOT/lib/libpython3.8.so
    export PYTHON_EXECUTABLE=$PYTHONROOT/bin/python3.8
    py_version="python3.8"
  fi
}

# kill进程
function kill_process () {
  kill `ps -ef | grep serving | awk '{print $2}'` > /dev/null 2>&1
  kill `ps -ef | grep python | awk '{print $2}'` > /dev/null 2>&1
  echo -e "${GREEN_COLOR}process killed...${RES}"
}

# 判断目录是否存在
function check_dir () {
  if [ ! -d "$1" ]
  then
  	mkdir -p $1
  fi
}

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

# check命令并保存日志
function check_save () {
  if [ $? == 0 ]
  then
  	echo -e "${GREEN_COLOR}$1 execute normally${RES}"
  	# server端
  	if [ $1 == "server" ]
  	then
      sleep $2
      tail ${dir}server_log.txt
    fi
    # client端
    if [ $1 == "client" ]
    then
      tail ${dir}client_log.txt
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
    tail ${dir}client_log.txt | tee -a ${log_dir}client_total.txt
    # 记录模型、部署方式、环境信息
    error_log $2
  fi
}

# 提取错误日志
function error_log() {
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

function generate_logs() {
  # 获取commit_id
  cd ${CODE_PATH}/Serving/
  commit_id=`git log | head -1`
  commit_id=${commit_id// /_} # 将空格替换为_
  commit_id=${commit_id:0:14} # 截取
  echo "-------commit_id: ${commit_id}"
  export TZ='Asia/Shanghai'
  today=`date +%m%d`
  echo "-------today: ${today}"
  # 生成日志
  cd ${CODE_PATH}
  mkdir -p ../logs_output/daily_logs/${ce_name}/benchmark_excel/
  if [ -d "../logs_output/${today}_${commit_id}/${ce_name}/logs_$1_$2" ]; then
    rm -rf ../logs_output/${today}_${commit_id}/${ce_name}/logs_$1_$2
  fi
  mkdir -p ../logs_output/${today}_${commit_id}/${ce_name}/
  # 汇总信息
  rm -rf ../logs_output/daily_logs/${ce_name}/benchmark_excel/benchmark_excel.xlsx
  cp logs/benchmark_excel/benchmark_excel.xlsx ../logs_output/daily_logs/${ce_name}/benchmark_excel/
  # 详细日志
  mv logs ../logs_output/${today}_${commit_id}/${ce_name}/logs_$1_$2
}

function pipeline_darknet53() {
    cd ${demo_dir}/pipeline/PaddleClas/DarkNet53
    dir=${log_dir}/DarkNet53/
    check_dir $dir
    # 链接模型数据
    data_dir=${data}pipeline/DarkNet53/
    link_data ${data_dir}
    data_dir=${data}pipeline/images/
    link_data ${data_dir}
    # 启动服务
    echo -e "${GREEN_COLOR}pipeline_clas_darknet53_GPU_pipeline server started${RES}"
    $py_version resnet50_web_service.py > ${dir}server_log.txt 2>&1 &
    # 命令检查
    check_save server 8
    # benchmark
    sed -i "s/python3.6/${py_version}/g" benchmark.sh
    sed -i "s/id=3/id=0/g" benchmark.sh
    sed -i "s/CPU_UTIL/CPU_UTILIZATION/g" benchmark.sh
    sed -i "s/GPU_MEM/MAX_GPU_MEMORY/g" benchmark.sh
    sed -i 's/1000/100/g' benchmark.sh
    sed -i "s/GPU_UTIL/GPU_UTILIZATION/g" benchmark.sh
    sh benchmark.sh
    tail -n 16 profile_log_clas-DarkNet53
    # 日志处理
    cp ${shell_dir}/benchmark_utils.py ./
    cp ${shell_dir}/parse_profile.py ./
    sed -i 's/runtime_device: "cpu"/runtime_device: "gpu"/g' benchmark_cfg.yaml
    sed -i 's/model_name: "imagenet"/model_name: "DarkNet53"/g' benchmark_cfg.yaml
    $py_version parse_profile.py --benchmark_cfg benchmark_cfg.yaml --benchmark_log profile_log_clas-DarkNet53 > ${dir}/client_log.txt 2>&1
    tail -n 31 ${dir}/client_log.txt
    cp -r benchmark_logs ${log_dir}/benchmark_logs/DarkNet53
    kill_process
}

function pipeline_PaddleClas() {
    cd ${demo_dir}/pipeline/PaddleClas/$1
    dir=${log_dir}/$1/
    check_dir $dir
    # 链接模型数据
    data_dir=${data}pipeline/$1/
    link_data ${data_dir}
    data_dir=${data}pipeline/images/
    link_data ${data_dir}
    # 启动服务
    echo -e "${GREEN_COLOR}pipeline_clas_$1_GPU_pipeline server started${RES}"
    $py_version resnet50_web_service.py > ${dir}server_log.txt 2>&1 &
    # 命令检查
    check_save server 8
    # benchmark
    sed -i "s/python3.6/${py_version}/g" benchmark.sh
    sed -i "s/id=3/id=0/g" benchmark.sh
#    sed -i "s/1 2 4 8 12 16/8 12/g" benchmark.sh
    sed -i 's/----/#----/g' benchmark.sh
    sed -i 's/1000/100/g' benchmark.sh
    sed -i "s/AVG QPS/AVG_QPS/g" benchmark.py
    sed -i "s/18000/18080/g" benchmark.py
    if [ $1 == "ResNet_V2_50" ]; then
        sed -i "s/clas-ResNet_v2_50/clas-ResNet_V2_50/g" benchmark.sh
    fi
    sh benchmark.sh
    tail -n 16 profile_log_clas-$1
    # 日志处理
    cp -rf ${shell_dir}/benchmark_utils.py ./
    cp -rf ${shell_dir}/parse_profile.py ./
    cp -rf ${shell_dir}/benchmark_cfg.yaml ./
    sed -i 's/runtime_device: "cpu"/runtime_device: "gpu"/g' benchmark_cfg.yaml
    sed -i "s/imagenet/$1/g" benchmark_cfg.yaml
    $py_version parse_profile.py --benchmark_cfg benchmark_cfg.yaml --benchmark_log profile_log_clas-$1 > ${dir}/client_log.txt 2>&1
    tail -n 31 ${dir}/client_log.txt
    cp -r benchmark_logs ${log_dir}/benchmark_logs/pipeline/$1
    kill_process
}

function pipeline_ocr() {
    cd ${demo_dir}/pipeline/ocr
    dir=${log_dir}/ocr/
    check_dir $dir
    # 链接模型数据
    data_dir=${data}ocr/
    link_data ${data_dir}
    # 启动服务
    echo -e "${GREEN_COLOR}pipeline_OCR_GPU_pipeline server started${RES}"
    sed -i 's/devices: ""/devices: "0"/g' config.yml
    $py_version web_service.py > ${dir}server_log.txt 2>&1 &
    # 命令检查
    check_save server 8
    # benchmark
    sed -i "s/python3.7/${py_version}/g" benchmark.sh
    sed -i "s/id=3/id=0/g" benchmark.sh
#    sed -i "s/1 2 4 8 12 16/8 12/g" benchmark.sh
    sed -i 's/----/#----/g' benchmark.sh
    sed -i 's/1000/100/g' benchmark.sh
    sed -i "s/AVG QPS/AVG_QPS/g" benchmark.py
    sed -i "76d" benchmark.py
    sh benchmark.sh
    tail -n 16 profile_log_ocr
    # 日志处理
    cp -rf ${shell_dir}/benchmark_utils.py ./
    cp -rf ${shell_dir}/parse_profile.py ./
    cp -rf ${shell_dir}/benchmark_cfg.yaml ./
    sed -i 's/runtime_device: "cpu"/runtime_device: "gpu"/g' benchmark_cfg.yaml
    sed -i "s/imagenet/OCR/g" benchmark_cfg.yaml
    $py_version parse_profile.py --benchmark_cfg benchmark_cfg.yaml --benchmark_log profile_log_ocr > ${dir}/client_log.txt 2>&1
    tail -n 31 ${dir}/client_log.txt
    cp -r benchmark_logs ${log_dir}/benchmark_logs/pipeline/ocr
    kill_process
}

function cpp_sync_resnet_v2_50() {
    cd ${demo_dir}/resnet_v2_50/
    dir=${log_dir}/cpp/resnet_v2_50/
    check_dir $dir
    # 链接模型数据
    data_dir=${data}resnet_v2_50/
    link_data ${data_dir}
    # 拷贝shell
    \cp -r ${shell_dir}/cpp_serving/resnet_v2_50/* ./
    # 启动服务
    echo -e "${GREEN_COLOR}cpp_ResNet_V2_50_GPU_pipeline server started${RES}"
    ${py_version} -m paddle_serving_server.serve --model resnet_v2_50_imagenet_model --port 9393 --thread 16 --gpu_ids 1 > ${dir}server_log.txt 2>&1 &
    check_save server 15
    bash benchmark.sh
    tail -n 31 profile_log_resnet_v2_50
    # 日志处理
    cp -rf ${shell_dir}/benchmark_utils.py ./
    cp -rf ${shell_dir}/parse_profile.py ./
    cp -rf ${shell_dir}/benchmark_cfg.yaml ./
    sed -i 's/runtime_device: "cpu"/runtime_device: "gpu"/g' benchmark_cfg.yaml
    sed -i "s/imagenet/ResNet_V2_50/g" benchmark_cfg.yaml
    $py_version parse_profile.py --benchmark_cfg benchmark_cfg.yaml --benchmark_log profile_log_resnet_v2_50 > ${dir}/client_log.txt 2>&1
    tail -n 31 ${dir}/client_log.txt
    cp -r benchmark_logs ${log_dir}/benchmark_logs/cpp/resnet_v2_50
    kill_process
}

function cpp_sync_ocr() {
    cd ${demo_dir}/ocr/
    dir=${log_dir}/cpp/ocr/
    check_dir $dir
    # 链接模型数据
    data_dir=${data}ocr/
    link_data ${data_dir}
    # 修改feed_var
    cp -r ocr_det_client/ ./ocr_det_client_cp
    rm -rf ocr_det_client
    mv ocr_det_client_cp ocr_det_client
    sed -i "s/feed_type: 1/feed_type: 20/g" ocr_det_client/serving_client_conf.prototxt
    sed -i "s/shape: 3/shape: 1/g" ocr_det_client/serving_client_conf.prototxt
    sed -i '7,8d' ocr_det_client/serving_client_conf.prototxt
    # 拷贝shell
    \cp -r ${shell_dir}/cpp_serving/ocr/* ./
    # 启动服务
    echo -e "${GREEN_COLOR}cpp_OCR_GPU_pipeline server started${RES}"
    ${py_version} -m paddle_serving_server.serve --model ocr_det_model ocr_rec_model --port 9293 --gpu_ids 1 > ${dir}server_log.txt 2>&1 &
    check_save server 15
    bash benchmark.sh
    tail -n 31 profile_log_ocr
    # 日志处理
    cp -rf ${shell_dir}/benchmark_utils.py ./
    cp -rf ${shell_dir}/parse_profile.py ./
    cp -rf ${shell_dir}/benchmark_cfg.yaml ./
    sed -i 's/runtime_device: "cpu"/runtime_device: "gpu"/g' benchmark_cfg.yaml
    sed -i "s/imagenet/OCR/g" benchmark_cfg.yaml
    $py_version parse_profile.py --benchmark_cfg benchmark_cfg.yaml --benchmark_log profile_log_ocr > ${dir}/client_log.txt 2>&1
    tail -n 31 ${dir}/client_log.txt
    cp -r benchmark_logs ${log_dir}/benchmark_logs/cpp/profile_log_ocr
    kill_process
}

# 创建日志目录
check_dir ${log_dir}/benchmark_excel
check_dir ${log_dir}/benchmark_logs/cpp
check_dir ${log_dir}/benchmark_logs/pipeline
# 设置py版本
set_py $1
env | grep -E "PYTHONROOT|PYTHON_INCLUDE_DIR|PYTHON_LIBRARIES|PYTHON_EXECUTABLE"

# 性能测试
unset_proxy
#pipeline_darknet53
#pipeline_PaddleClas HRNet_W18_C
#pipeline_PaddleClas MobileNetV1
#pipeline_PaddleClas MobileNetV2
#pipeline_PaddleClas MobileNetV3_large_x1_0
#pipeline_PaddleClas ResNeXt101_vd_64x4d
#pipeline_PaddleClas ResNet50_vd
#pipeline_PaddleClas ResNet50_vd_FPGM
#pipeline_PaddleClas ResNet50_vd_KL
#pipeline_PaddleClas ResNet50_vd_PACT
pipeline_PaddleClas ResNet_V2_50
#pipeline_PaddleClas ShuffleNetV2_x1_0
pipeline_ocr
cpp_sync_resnet_v2_50
cpp_sync_ocr

# 生成excel
cd ${CODE_PATH}/benchmark/
$py_version benchmark_analysis.py --log_path ${log_dir}/benchmark_logs/pipeline --server_mode Pipeline --output_name benchmark_excel_pipeline.xlsx --output_html_name benchmark_data_pipeline.html
$py_version benchmark_analysis.py --log_path ${log_dir}/benchmark_logs/cpp --server_mode Pipeline --output_name benchmark_excel_cpp.xlsx --output_html_name benchmark_data_cpp.html
cp *.xlsx ${log_dir}/benchmark_excel
cp *.html ${log_dir}/benchmark_excel
# 写入数据库
$py_version benchmark_backend.py --log_path=${log_dir}/benchmark_logs/pipeline --post_url=${post_url} --frame_name=paddle --api=python --framework_version=ffa88c31c2da5090c6f70e8e9b523356d7cd5e7f --cuda_version=10.2 --cudnn_version=7.6.5 --trt_version=6.0.1.5 --device_name=gpu --server_mode Pipeline
$py_version benchmark_backend.py --log_path=${log_dir}/benchmark_logs/cpp --post_url=${post_url} --frame_name=paddle --api=python --framework_version=ffa88c31c2da5090c6f70e8e9b523356d7cd5e7f --cuda_version=10.2 --cudnn_version=7.6.5 --trt_version=6.0.1.5 --device_name=gpu --server_mode CPP

generate_logs $1 $2
