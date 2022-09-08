#!/bin/bash
# 需设置
# tar_path (包地址)

tar_path=${tar_path}
gcc=$1
device=$2
trt_path=$3

function get_tar() {
    cd ${code_path}/Paddle-Inference-Demo/c++/lib
    rm -rf paddle_inference*
    wget -q ${tar_path}
    tar -xf paddle_inference.tgz
}

function run_trt() {
    cd ${code_path}/Paddle-Inference-Demo/c++/gpu/resnet50
    mv /usr/bin/c++ /usr/bin/c++.bak
    mv /usr/bin/gcc /usr/bin/gcc.bak
    ln -s /usr/local/gcc-${gcc}/bin/c++ /usr/bin/
    ln -s /usr/local/gcc-${gcc}/bin/gcc /usr/bin/
    wget -q https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
    tar -xf resnet50.tgz
    sed -i "34 i TENSORRT_ROOT=${trt_path}" compile.sh
    bash -x compile.sh
    ./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --run_mode=trt_fp32
    exit_code=$?
}

function run_cpu() {
    cd ${code_path}/Paddle-Inference-Demo/c++/cpu/resnet50
    mv /usr/bin/c++ /usr/bin/c++.bak
    mv /usr/bin/gcc /usr/bin/gcc.bak
    ln -s /usr/local/gcc-${gcc}/bin/c++ /usr/bin/
    ln -s /usr/local/gcc-${gcc}/bin/gcc /usr/bin/
    wget -q https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
    tar -xf resnet50.tgz
    sed -i "s/WITH_MKL=ON/WITH_MKL=OFF/" compile.sh
    sed -i "s/WITH_ONNXRUNTIME=ON/WITH_ONNXRUNTIME=OFF/" compile.sh
    bash -x compile.sh
    ./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams
    exit_code=$?
}


export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH};
export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH};
export PYTHON_FLAGS='-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.7.0/bin/python3.7 -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.7.0/include/python3.7 -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.7.0/lib/libpython3.so';


get_tar
if [ ${device} == "cpu" ]; then
    run_cpu
elif [ ${device} == "gpu" ]; then
    run_trt
fi


exit ${exit_code}