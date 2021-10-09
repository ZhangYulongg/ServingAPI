#!/bin/bash

cuda=$1
cudnn=$2
gcc=$3
trt=$4
trt_path=$5

function get_tar() {
    rm -rf paddle_inference*
    wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.0-rc0/cxx_c/Linux/GPU/x86-64_gcc${gcc}_avx_mkl_cuda${cuda}_cudnn${cudnn}_trt${trt}/paddle_inference.tgz
#    wget https://paddle-inference-lib.bj.bcebos.com/2.1.2/linux-gpu-cuda10.1-cudnn7-mkl-gcc5.4-trt6-avx/paddle_inference.tgz
    tar -xf paddle_inference.tgz
}

function cmake() {
    rm -rf build
    mv /usr/bin/c++ /usr/bin/c++.bak
    mv /usr/bin/gcc /usr/bin/gcc.bak
    ln -s /usr/local/gcc-${gcc}/bin/c++ /usr/bin/
    ln -s /usr/local/gcc-${gcc}/bin/gcc /usr/bin/
    bash build.sh /home/zhangyulong04/continuous_integration/inference/inference_api_test/cpp_api_test/paddle_inference ON ON ON ${trt_path}
}


export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH};
export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH};
export PYTHON_FLAGS='-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.7.0/bin/python3.7 -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.7.0/include/python3.7 -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.7.0/lib/libpython3.so';


get_tar
cmake

bash bin/run-new-api-case-mini.sh > log_${cuda}_${cudnn}_${gcc}_${trt}.txt 2>&1
