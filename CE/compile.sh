#!/bin/bash
# Serving 根目录下运行，提前设置serving_dir, proxy, opencv_dir(/mnt/serving/opencv-3.4.7/opencv3)
# $1 36、37、38
# $2 cpu、101、102、110
# $3 opencv

if [ ${serving_dir} != "" ]; then
    cd ${serving_dir}
fi

if [ ${proxy} == "" ]; then
    echo "no proxy set"
    exit 1
fi

function set_proxy() {
    echo -e "---------set proxy---------${RES}"
    export https_proxy=${proxy}
    export http_proxy=${proxy}
}

function unset_proxy() {
    echo -e "---------unset proxy---------${RES}"
    unset http_proxy
    unset https_proxy
}

# 设置python版本
function set_py () {
    if [ $1 == 36 ]; then
        py_version="python3.6"
        export PYTHONROOT=/usr/local/
        export PYTHON_INCLUDE_DIR=$PYTHONROOT/include/python3.6m
        export PYTHON_LIBRARIES=$PYTHONROOT/lib/libpython3.6m.so
        export PYTHON_EXECUTABLE=$PYTHONROOT/bin/python3.6
    elif [ $1 == 37 ]; then
        py_version="python3.7"
        export PYTHONROOT=/usr/local/
        export PYTHON_INCLUDE_DIR=$PYTHONROOT/include/python3.7m
        export PYTHON_LIBRARIES=$PYTHONROOT/lib/libpython3.7m.so
        export PYTHON_EXECUTABLE=$PYTHONROOT/bin/python3.7
    elif [ $1 == 38 ]; then
        py_version="python3.8"
        export PYTHONROOT=/usr/local/
        export PYTHON_INCLUDE_DIR=$PYTHONROOT/include/python3.8
        export PYTHON_LIBRARIES=$PYTHONROOT/lib/libpython3.8.so
        export PYTHON_EXECUTABLE=$PYTHONROOT/bin/python3.8
    else
        echo -e "Error py version$1"
        exit 1
    fi
    unset_proxy
    ${py_version} -m pip install -r python/requirements.txt -i https://mirror.baidu.com/pypi/simple
    set_proxy
}

function install_go() {
    unset_proxy
    export GOPATH=$HOME/go
    export PATH=$PATH:$GOPATH/bin
    go env -w GO111MODULE=on
    go env -w GOPROXY=https://goproxy.cn,direct
    go get -u github.com/grpc-ecosystem/grpc-gateway/protoc-gen-grpc-gateway@v1.15.2
    go get -u github.com/grpc-ecosystem/grpc-gateway/protoc-gen-swagger@v1.15.2
    go get -u github.com/golang/protobuf/protoc-gen-go@v1.4.3
    go get -u google.golang.org/grpc@v1.33.0
    go env -w GO111MODULE=auto
    set_proxy
}

function compile_server_gpu() {
    export CUDA_PATH='/usr/local/cuda'
    export CUDNN_LIBRARY='/usr/local/cuda/lib64/'
    export CUDA_CUDART_LIBRARY="/usr/local/cuda/lib64/"
    if [ $1 == 101 ]; then
        export TENSORRT_LIBRARY_PATH="/usr/local/TensorRT6-cuda10.1-cudnn7/targets/x86_64-linux-gnu/"
    elif [ $1 == 102 ]; then
        export TENSORRT_LIBRARY_PATH="/usr/local/TensorRT-7.1.3.4/targets/x86_64-linux-gnu/"
    elif [ $1 == 110 ]; then
        export TENSORRT_LIBRARY_PATH="/usr/local/TensorRT-7.1.3.4/targets/x86_64-linux-gnu/"
    else
        echo -e "Error cuda version$1"
        exit 1
    fi
    mkdir server-build-gpu && cd server-build-gpu
    cmake -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
        -DPYTHON_LIBRARIES=$PYTHON_LIBRARIES \
        -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
        -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_PATH} \
        -DCUDNN_LIBRARY=${CUDNN_LIBRARY} \
        -DCUDA_CUDART_LIBRARY=${CUDA_CUDART_LIBRARY} \
        -DTENSORRT_ROOT=${TENSORRT_LIBRARY_PATH} \
        -DSERVER=ON \
        -DWITH_GPU=ON ..
    make -j10
    unset_proxy
    ${py_version} -m pip install python/dist/paddle* -i https://mirror.baidu.com/pypi/simple
    set_proxy
    cd ..
}

function compile_server_cpu() {
    mkdir server-build-cpu && cd server-build-cpu
    cmake -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR/ \
        -DPYTHON_LIBRARIES=$PYTHON_LIBRARIES \
        -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
        -DSERVER=ON ..
    make -j10
    unset_proxy
    ${py_version} -m pip install python/dist/paddle* -i https://mirror.baidu.com/pypi/simple
    set_proxy
    cd ..
}

function compile_server_gpu_withopencv() {
    export OPENCV_DIR=${opencv_dir}
    export CUDA_PATH='/usr/local/cuda'
    export CUDNN_LIBRARY='/usr/local/cuda/lib64/'
    export CUDA_CUDART_LIBRARY="/usr/local/cuda/lib64/"
    if [ $1 == 101 ]; then
        export TENSORRT_LIBRARY_PATH="/usr/local/TensorRT6-cuda10.1-cudnn7/targets/x86_64-linux-gnu/"
    elif [ $1 == 102 ]; then
        export TENSORRT_LIBRARY_PATH="/usr/local/TensorRT-7.1.3.4/targets/x86_64-linux-gnu/"
    elif [ $1 == 110 ]; then
        export TENSORRT_LIBRARY_PATH="/usr/local/TensorRT-7.1.3.4/targets/x86_64-linux-gnu/"
    else
        echo -e "Error cuda version$1"
        exit 1
    fi
    mkdir server-build-gpu-opencv && cd server-build-gpu-opencv
    cmake -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
        -DPYTHON_LIBRARIES=$PYTHON_LIBRARIES \
        -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
        -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_PATH} \
        -DCUDNN_LIBRARY=${CUDNN_LIBRARY} \
        -DCUDA_CUDART_LIBRARY=${CUDA_CUDART_LIBRARY} \
        -DTENSORRT_ROOT=${TENSORRT_LIBRARY_PATH} \
        -DOPENCV_DIR=${OPENCV_DIR} \
        -DWITH_OPENCV=ON \
        -DSERVER=ON \
        -DWITH_GPU=ON ..
    make -j10
    unset_proxy
    ${py_version} -m pip install python/dist/paddle* -i https://mirror.baidu.com/pypi/simple
    set_proxy
    cd ..
}

function compile_client() {
    mkdir client-build && cd client-build
    cmake -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
        -DPYTHON_LIBRARIES=$PYTHON_LIBRARIES \
        -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
        -DCLIENT=ON ..
    make -j10
    unset_proxy
    ${py_version} -m pip install python/dist/paddle* -i https://mirror.baidu.com/pypi/simple
    set_proxy
    cd ..
}

function compile_app() {
    mkdir app-build && cd app-build
    cmake -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
        -DPYTHON_LIBRARIES=$PYTHON_LIBRARIES \
        -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
        -DAPP=ON ..
    make -j10
    unset_proxy
    ${py_version} -m pip install python/dist/paddle* -i https://mirror.baidu.com/pypi/simple
    set_proxy
    cd ..
}


set_proxy
git submodule update --init --recursive
set_py $1
install_go
if [ $3 == "opencv" ]; then
    compile_server_gpu_withopencv $2
elif [ $2 == "cpu" ]; then
    compile_server_cpu
else
    compile_server_gpu $2
fi
compile_client
compile_app
