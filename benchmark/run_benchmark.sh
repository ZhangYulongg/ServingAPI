#!/bin/bash

# 日志目录
log_dir=${CODE_PATH}/logs/
demo_dir=${CODE_PATH}/Serving/python/examples/
dir=`pwd`
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

function pre_loading() {
  cd ${demo_dir}/bert
  # 链接模型数据
  data_dir=${data}bert/
  link_data ${data_dir}
  if [ $1 == "cpu" ]; then
    $py_version -m paddle_serving_server.serve --model bert_seq128_model/ --port 9292 > ${log_dir}/cpu_pre_loading.txt 2>&1 &
    n=1
    while [ ${n} -le 12 ]; do
      sleep 10
      tail -2 ${log_dir}/cpu_pre_loading.txt
      # 检查日志
      echo "-----check for next step n=${n}-----"
      tail ${log_dir}/cpu_pre_loading.txt | grep -E "ir_graph_to_program_pass" ${log_dir}/cpu_pre_loading.txt > /dev/null
      if [ $? == 0 ]; then
        # 预加载成功
        echo "----------gpu_server loaded----------"
        break
      elif [ $n == 12 ]; then
        tail ${log_dir}/cpu_pre_loading.txt | grep -E "failed|Error" ${log_dir}/cpu_pre_loading.txt > /dev/null
        if [ $? == 0 ]; then
          # 下载出错
          exit
        fi
        n=10
      fi
      n=$((n + 1))
    done
    tail ${log_dir}/cpu_pre_loading.txt
    echo "----------cpu_server loaded----------"
    kill_process
  else
    $py_version -m paddle_serving_server.serve --model bert_seq128_model/ --port 9292 --gpu_ids 0 > ${log_dir}/gpu_pre_loading.txt 2>&1 &
    n=1
    while [ ${n} -le 50 ]; do
      sleep 10
      tail -2 ${log_dir}/gpu_pre_loading.txt
      # 检查日志
      echo "-----check for next step n=${n}-----"
  #    tail ${log_dir}/gpu_pre_loading.txt | grep -E "Succ load model" ${log_dir}/gpu_pre_loading.txt > /dev/null
      tail ${log_dir}/gpu_pre_loading.txt | grep -E "ir_graph_to_program_pass" ${log_dir}/gpu_pre_loading.txt > /dev/null
      if [ $? == 0 ]; then
        # 预加载成功
        echo "----------gpu_server loaded----------"
        head ${log_dir}/gpu_pre_loading.txt
        break
      elif [ $n == 50 ]; then
        tail ${log_dir}/cpu_pre_loading.txt | grep -E "failed|Error" ${log_dir}/cpu_pre_loading.txt > /dev/null
        if [ $? == 0 ]; then
          # 下载出错
          exit
        fi
        n=45
      fi
      n=$((n + 1))
    done
    tail ${log_dir}/gpu_pre_loading.txt
    echo "----------gpu_server loaded----------"
    kill_process
  fi
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

# 安装python依赖包
function py_requirements () {
  cd ${CODE_PATH}/Serving
  echo -e "${YELOW_COLOR}---------install python requirements---------${RES}"
  unset_proxy
  echo "---------Python Version: $py_version"
  set_proxy
  $py_version -m pip install --upgrade pip==21.1.3
  unset_proxy
  $py_version -m pip install -r python/requirements.txt -i https://mirror.baidu.com/pypi/simple
  if [ $2 == 101 ]; then
    if [ $1 == 36 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddlepaddle_gpu-2.1.0.post101-cp36-cp36m-linux_x86_64.whl
    elif [ $1 == 37 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddlepaddle_gpu-2.1.0.post101-cp37-cp37m-linux_x86_64.whl
    elif [ $1 == 38 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddlepaddle_gpu-2.1.0.post101-cp38-cp38-linux_x86_64.whl
    fi
  elif [ $2 == 102 ]; then
    if [ $1 == 36 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda10.2-cudnn8-mkl-gcc8.2/paddlepaddle_gpu-2.1.0-cp36-cp36m-linux_x86_64.whl
    elif [ $1 == 37 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda10.2-cudnn8-mkl-gcc8.2/paddlepaddle_gpu-2.1.0-cp37-cp37m-linux_x86_64.whl
    elif [ $1 == 38 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda10.2-cudnn8-mkl-gcc8.2/paddlepaddle_gpu-2.1.0-cp38-cp38-linux_x86_64.whl
    fi
  elif [ $2 == 110 ]; then
    if [ $1 == 36 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda11.0-cudnn8-mkl-gcc8.2/paddlepaddle_gpu-2.1.0.post110-cp36-cp36m-linux_x86_64.whl
    elif [ $1 == 37 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda11.0-cudnn8-mkl-gcc8.2/paddlepaddle_gpu-2.1.0.post110-cp37-cp37m-linux_x86_64.whl
    elif [ $1 == 38 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda11.0-cudnn8-mkl-gcc8.2/paddlepaddle_gpu-2.1.0.post110-cp38-cp38-linux_x86_64.whl
    fi
  elif [ $2 == "cpu" ]; then
    if [ $1 == 36 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.1.0-cpu-avx-mkl/paddlepaddle-2.1.0-cp36-cp36m-linux_x86_64.whl
    elif [ $1 == 37 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.1.0-cpu-avx-mkl/paddlepaddle-2.1.0-cp37-cp37m-linux_x86_64.whl
    elif [ $1 == 38 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.1.0-cpu-avx-mkl/paddlepaddle-2.1.0-cp38-cp38-linux_x86_64.whl
    fi
  else
    echo -e "${RED_COLOR}Error cuda version$1${RES}"
    exit
  fi
  set_proxy
  $py_version -m pip install paddlepaddle*
  ${py_version} -m pip install psutil
  ${py_version} -m pip install pandas
  ${py_version} -m pip install openpyxl
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
  set_proxy
  if [ $1 == 38 ]; then
    $py_version -m pip install sentencepiece
  fi
  $py_version -m pip install *
  ${py_version} -m pip install pandas
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
    sh benchmark.sh
    tail -n 16 profile_log_clas-DarkNet53
    # 日志处理
    cp /mnt/serving/zyl/shell_zyl/benchmark_utils.py ./
    cp /mnt/serving/zyl/shell_zyl/parse_profile.py ./
    sed -i 's/runtime_device: "cpu"/runtime_device: "gpu"/g' benchmark_cfg.yaml
    sed -i 's/model_name: "imagenet"/model_name: "DarkNet53"/g' benchmark_cfg.yaml
    $py_version parse_profile.py --benchmark_cfg benchmark_cfg.yaml --benchmark_log profile_log_clas-DarkNet53 > ${dir}/client_log.txt 2>&1
    tail -n 31 ${dir}/client_log.txt
    cp -r benchmark_logs ${log_dir}/benchmark_logs/DarkNet53
    kill_process
}

function pipeline_HRNet_W18_C() {
    cd ${demo_dir}/pipeline/PaddleClas/HRNet_W18_C
    # 链接模型数据
    data_dir=${data}pipeline/HRNet_W18_C/
    link_data ${data_dir}
    data_dir=${data}pipeline/images/
    link_data ${data_dir}
    # 启动服务
    echo -e "${GREEN_COLOR}pipeline_clas_HRNet_W18_C_GPU_pipeline server started${RES}"
    $py_version resnet50_web_service.py > ${dir}server_log.txt 2>&1 &
    # 命令检查
    check_save server 8
    # benchmark
    sed -i "s/python3.6/${py_version}/g" benchmark.sh
    sed -i "s/id=3/id=0/g" benchmark.sh
    sh benchmark.sh
    tail profile_log_clas-HRNet_W18_C
    # 日志处理
    cp /mnt/serving/zyl/shell_zyl/benchmark_utils.py ./
    cp /mnt/serving/zyl/shell_zyl/parse_profile.py ./
#    cp /mnt/serving/zyl/shell_zyl/benchmark_analysis.py ./
    cp /mnt/serving/zyl/shell_zyl/benchmark_cfg.yaml ./
    sed -i "s/runtime_device: "cpu"/runtime_device: "gpu"/g" benchmark_cfg.yaml
    sed -i "s/model_name: "imagenet"/model_name: "HRNet_W18_C"/g" benchmark_cfg.yaml
    $py_version parse_profile.py --benchmark_cfg benchmark_cfg.yaml --benchmark_log profile_log_clas-HRNet_W18_C
#    $py_version benchmark_analysis.py --log_path ${CODE_PATH}/benchmark_logs
#    cp benchmark_excel.xlsx ${log_dir}/benchmark_excel
    cp -r benchmark_logs/* ${log_dir}
    kill_process
}

function pipeline_MobileNetV1() {
    cd ${demo_dir}/pipeline/PaddleClas/MobileNetV1
    # 链接模型数据
    data_dir=${data}pipeline/MobileNetV1/
    link_data ${data_dir}
    data_dir=${data}pipeline/images/
    link_data ${data_dir}
    # 启动服务
    echo -e "${GREEN_COLOR}pipeline_clas_MobileNetV1_GPU_pipeline server started${RES}"
    $py_version resnet50_web_service.py > ${dir}server_log.txt 2>&1 &
    # 命令检查
    check_save server 8
    # benchmark
    sed -i "s/python3.6/${py_version}/g" benchmark.sh
    sed -i "s/id=3/id=0/g" benchmark.sh
    sh benchmark.sh
    tail profile_log_clas-MobileNetV1
    # 日志处理
    cp /mnt/serving/zyl/shell_zyl/benchmark_utils.py ./
    cp /mnt/serving/zyl/shell_zyl/parse_profile.py ./
    cp /mnt/serving/zyl/shell_zyl/benchmark_cfg.yaml ./
    sed -i 's/runtime_device: "cpu"/runtime_device: "gpu"/g' benchmark_cfg.yaml
    sed -i 's/model_name: "imagenet"/model_name: "MobileNetV1"/g' benchmark_cfg.yaml
    $py_version parse_profile.py --benchmark_cfg benchmark_cfg.yaml --benchmark_log profile_log_clas-MobileNetV1
#    cp -r benchmark_logs/* ${log_dir}
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
    sed -i "s/CPU_UTILIZATION/CPU_UTIL/g" benchmark.sh
    sed -i "s/MAX_GPU_MEMORY/GPU_MEM/g" benchmark.sh
    sed -i "s/GPU_UTILIZATION/GPU_UTIL/g" benchmark.sh
    sed -i "s/AVG QPS/AVG_QPS/g" benchmark.py
    sed -i "s/18000/18080/g" benchmark.py
    if [ $1 == "ResNet_V2_50" ]; then
        sed -i "s/clas-ResNet_v2_50/clas-ResNet_V2_50/g" benchmark.sh
    fi
    sh benchmark.sh
    tail -n 16 profile_log_clas-$1
    # 日志处理
    cp -rf /mnt/serving/zyl/shell_zyl/benchmark_utils.py ./
    cp -rf /mnt/serving/zyl/shell_zyl/parse_profile.py ./
    cp -rf /mnt/serving/zyl/shell_zyl/benchmark_cfg.yaml ./
    sed -i 's/runtime_device: "cpu"/runtime_device: "gpu"/g' benchmark_cfg.yaml
    sed -i "s/imagenet/$1/g" benchmark_cfg.yaml
    $py_version parse_profile.py --benchmark_cfg benchmark_cfg.yaml --benchmark_log profile_log_clas-$1 > ${dir}/client_log.txt 2>&1
    tail -n 31 ${dir}/client_log.txt
    cp -r benchmark_logs ${log_dir}/benchmark_logs/$1
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
    sed -i "s/CPU_UTILIZATION/CPU_UTIL/g" benchmark.sh
    sed -i "s/MAX_GPU_MEMORY/GPU_MEM/g" benchmark.sh
    sed -i "s/GPU_UTILIZATION/GPU_UTIL/g" benchmark.sh
    sed -i "s/AVG QPS/AVG_QPS/g" benchmark.py
    sed -i "78d" benchmark.py
    sh benchmark.sh
    tail -n 16 profile_log_ocr
    # 日志处理
    cp -rf /mnt/serving/zyl/shell_zyl/benchmark_utils.py ./
    cp -rf /mnt/serving/zyl/shell_zyl/parse_profile.py ./
    cp -rf /mnt/serving/zyl/shell_zyl/benchmark_cfg.yaml ./
    sed -i 's/runtime_device: "cpu"/runtime_device: "gpu"/g' benchmark_cfg.yaml
    sed -i "s/imagenet/ocr/g" benchmark_cfg.yaml
    $py_version parse_profile.py --benchmark_cfg benchmark_cfg.yaml --benchmark_log profile_log_ocr > ${dir}/client_log.txt 2>&1
    tail -n 31 ${dir}/client_log.txt
    cp -r benchmark_logs ${log_dir}/benchmark_logs/ocr
    kill_process
}

# 创建日志目录
check_dir ${log_dir}/benchmark_excel
check_dir ${log_dir}/benchmark_logs
# 设置py版本
set_py $1
env | grep -E "PYTHONROOT|PYTHON_INCLUDE_DIR|PYTHON_LIBRARIES|PYTHON_EXECUTABLE"
# 测试
py_requirements $1 $2
pip_install_serving $1 $2

echo "--------pip list after pip: "
$py_version -m pip list
# whl包检查
if [ `$py_version -m pip list | grep -c paddlepaddle` != 1 ]; then
    py_requirements $1 $2
    if [ `$py_version -m pip list | grep -c paddlepaddle` != 1 ]; then
        echo "----------paddle install failed!----------"
        exit 2
    fi
fi

if [ `$py_version -m pip list | egrep "paddle-serving" | wc -l` -eq 3 ]; then
    echo "-----------whl_packages succ"
else
    echo "----------whl_packages failed"
    pip_install_serving $1 $2
    if [ `$py_version -m pip list | egrep "paddle-serving" | wc -l` -eq 3 ]; then
       echo "-----------whl_packages succ ---2"
    else
        echo "----------whl_packages failed ---2"
        exit 1
    fi
fi

pre_loading $2

# 性能测试
unset_proxy
pipeline_darknet53
pipeline_PaddleClas HRNet_W18_C
pipeline_PaddleClas MobileNetV1
pipeline_PaddleClas MobileNetV2
pipeline_PaddleClas MobileNetV3_large_x1_0
pipeline_PaddleClas ResNeXt101_vd_64x4d
pipeline_PaddleClas ResNet50_vd
pipeline_PaddleClas ResNet50_vd_FPGM
pipeline_PaddleClas ResNet50_vd_KL
pipeline_PaddleClas ResNet50_vd_PACT
pipeline_PaddleClas ResNet_V2_50
pipeline_PaddleClas ShuffleNetV2_x1_0
pipeline_ocr

# 生成excel
cd ${CODE_PATH}/shell_zyl
$py_version benchmark_analysis.py --log_path ${log_dir}/benchmark_logs --server_mode Pipeline
cp benchmark_excel.xlsx ${log_dir}/benchmark_excel
# 写入数据库
$py_version benchmark_backend.py --log_path=${log_dir}/benchmark_logs --post_url=${post_url} --frame_name=paddle --api=python --framework_version=ffa88c31c2da5090c6f70e8e9b523356d7cd5e7f --cuda_version=10.2 --cudnn_version=7.6.5 --trt_version=6.0.1.5 --device_name=gpu --server_mode Pipeline

generate_logs $1 $2
