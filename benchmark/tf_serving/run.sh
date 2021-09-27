#!/bin/bash

model_dir=${tf_model_dir}
log_dir=${CODE_PATH}/tf_logs/
workspace=${CODE_PATH}/tf_test
shell_dir=${CODE_PATH}/benchmark/
# 输出颜色
RED_COLOR='\E[1;31m'
GREEN_COLOR='\E[1;32m'
YELOW_COLOR='\E[1;33m'
RES='\E[0m'
# 异常词
error_words="Fail|DENIED|None"

function link_data () {
    for file in $1/*
    do
    if [ ! -h ${file##*/} ]; then
        ln -s ${file} ./${file##*/}
    fi
    done
}

function kill_process () {
  ps -ef|grep 'tensorflow_model_server'|grep -v grep|cut -c 9-15 | xargs kill -9
  echo -e "${GREEN_COLOR}process killed...${RES}"
}

function tf_serving_resnet() {
    cd ${workspace}
    dir=$PWD
    # 软链model
    link_data ${model_dir}
    # copy shell
    \cp -r ${shell_dir}/tf_serving/* ./
    # 启动服务
    export CUDA_VISIBLE_DEVICES=1
    echo -e "${GREEN_COLOR}tf_serving_ResNet_GPU server started${RES}"
    tensorflow_model_server --port=8500 --enable_batching=true --model_name="serving_default" --model_base_path="${workspace}/model/resnet_v1" --batching_parameters_file="batch_config" > server_log.txt 2>&1 &
    check_save server 15
    cat server_log.txt
    echo "============client begin==========="
    bash benchmark_tf.sh resnet
    tail -n 16 profile_log_resnet
    # 日志处理
    cp -rf ${shell_dir}/benchmark_utils.py ./
    cp -rf ${shell_dir}/parse_profile.py ./
    cp -rf ${shell_dir}/benchmark_cfg.yaml ./
    sed -i 's/runtime_device: "cpu"/runtime_device: "gpu"/g' benchmark_cfg.yaml
    sed -i "s/imagenet/ResNet/g" benchmark_cfg.yaml
    $py_version parse_profile.py --benchmark_cfg benchmark_cfg.yaml --benchmark_log profile_log_resnet > ${dir}/client_log.txt 2>&1
    tail -n 31 ${dir}/client_log.txt
    cp -r benchmark_logs ${log_dir}/benchmark_logs/tf_serving/profile_log_ocr
    kill_process
}

/usr/bin/python3 -m pip install paddle-serving-client==0.6.2 -i https://mirror.baidu.com/pypi/simple
/usr/bin/python3 -m pip install paddle-serving-app==0.6.2 -i https://mirror.baidu.com/pypi/simple
/usr/bin/python3 -m pip install psutil -i https://mirror.baidu.com/pypi/simple
/usr/bin/python3 -m pip install openpyxl -i https://mirror.baidu.com/pypi/simple
/usr/bin/python3 -m pip install pandas -i https://mirror.baidu.com/pypi/simple
/usr/bin/python3 -m pip install -r ${CODE_PATH}/Serving/python/requirements.txt -i https://mirror.baidu.com/pypi/simple
mkdir -p ${workspace}
mkdir -p ${log_dir}
mkdir -p ${log_dir}/benchmark_logs/tf_serving

tf_serving_resnet

# 生成excel
cd ${CODE_PATH}/benchmark/
#$py_version benchmark_analysis.py --log_path ${log_dir}/benchmark_logs/tf_serving --server_mode TensorFlow --output_name benchmark_excel_tf.xlsx --output_html_name benchmark_data_tf.html
#cp benchmark_excel.xlsx ${log_dir}/benchmark_excel
# 写入数据库
#$py_version benchmark_backend.py --log_path=${log_dir}/benchmark_logs/tf_serving --post_url=${post_url} --frame_name=paddle --api=python --framework_version=ffa88c31c2da5090c6f70e8e9b523356d7cd5e7f --cuda_version=10.2 --cudnn_version=7.6.5 --trt_version=6.0.1.5 --device_name=gpu --server_mode TensorFlow
