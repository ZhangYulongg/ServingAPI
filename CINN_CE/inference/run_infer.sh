#!/bin/bash

function download_model {
    local tar_file=$1
    if [[ ! -f $tar_file ]]; then
        wget https://paddle-inference-dist.bj.bcebos.com/CINN/$tar_file
        tar -zxvf $tar_file
    fi
}

function main() {
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

    ${py_version} test_resnet18.py $PWD/ResNet18 ON | tee -a ${log_dir}/ResNet18.log 2>&1
    ${py_version} test_mobilenetv2.py $PWD/MobileNetV2 ON | tee -a ${log_dir}/MobileNetV2.log 2>&1
    ${py_version} test_efficientnet.py $PWD/EfficientNet ON | tee -a ${log_dir}/EfficientNet.log 2>&1
    ${py_version} test_mobilenetv1.py $PWD/MobilenetV1 ON | tee -a ${log_dir}/MobilenetV1.log 2>&1
    ${py_version} test_resnet50.py $PWD/ResNet50 ON | tee -a ${log_dir}/ResNet50.log 2>&1
    ${py_version} test_squeezenet.py $PWD/SqueezeNet ON | tee -a ${log_dir}/SqueezeNet.log 2>&1
    ${py_version} test_facedet.py $PWD/FaceDet ON | tee -a ${log_dir}/FaceDet.log 2>&1
}

main
