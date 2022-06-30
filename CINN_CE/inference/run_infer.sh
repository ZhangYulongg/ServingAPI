#!/bin/bash

function download_model {
    local tar_file=$1
    if [[ ! -f $tar_file ]]; then
        wget -q https://paddle-inference-dist.bj.bcebos.com/CINN/$tar_file
        tar -zxvf $tar_file
    fi
}

function main() {
    enable_gpu=$1

    log_dir=$PWD/ce_log
    rm -rf ${log_dir}
    mkdir -p ${log_dir}

    cd python/tests

    download_model ResNet18.tar.gz
    download_model MobileNetV2.tar.gz
    download_model EfficientNet.tar.gz
    download_model MobilenetV1.tar.gz
    download_model ResNet50.tar.gz
    download_model SqueezeNet.tar.gz
    download_model FaceDet.tar.gz

    sed -i "s/repeat = 10/repeat = 1000/" test_resnet18.py
    sed -i "s/repeat = 10/repeat = 1000/" test_mobilenetv2.py
    sed -i "s/repeat = 10/repeat = 1000/" test_efficientnet.py
    sed -i "s/repeat = 10/repeat = 1000/" test_mobilenetv1.py
    sed -i "s/repeat = 10/repeat = 1000/" test_resnet50.py
    sed -i "s/repeat = 10/repeat = 1000/" test_squeezenet.py
    sed -i "s/repeat = 10/repeat = 1000/" test_facedet.py

    ${py_version} test_resnet18.py $PWD/ResNet18 ${enable_gpu} 2>&1 | tee -a ${log_dir}/ResNet18.log
    ${py_version} test_mobilenetv2.py $PWD/MobileNetV2 ${enable_gpu} 2>&1 | tee -a ${log_dir}/MobileNetV2.log
    ${py_version} test_efficientnet.py $PWD/EfficientNet ${enable_gpu} 2>&1 | tee -a ${log_dir}/EfficientNet.log
    ${py_version} test_mobilenetv1.py $PWD/MobilenetV1 ${enable_gpu} 2>&1 | tee -a ${log_dir}/MobilenetV1.log
    ${py_version} test_resnet50.py $PWD/ResNet50 ${enable_gpu} 2>&1 | tee -a ${log_dir}/ResNet50.log
    ${py_version} test_squeezenet.py $PWD/SqueezeNet ${enable_gpu} 2>&1 | tee -a ${log_dir}/SqueezeNet.log
    ${py_version} test_facedet.py $PWD/FaceDet ${enable_gpu} 2>&1 | tee -a ${log_dir}/FaceDet.log
}

main $@
