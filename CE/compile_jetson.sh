#!/bin/bash
# Serving 根目录下运行，提前设置serving_dir, proxy, opencv_dir(/mnt/serving/opencv-3.4.7/opencv3)
# $1 36、37、38
# $2 cpu、101、102、110、arm
# $3 opencv
export serving_dir=${CODE_PATH}/Serving

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
    export PYTHONROOT=/usr
    export PYTHON_INCLUDE_DIR=$PYTHONROOT/include/python3.6m
    export PYTHON_LIBRARIES=$PYTHONROOT/lib/libpython3.6m.so
    export PYTHON_EXECUTABLE=$PYTHONROOT/bin/python3.6
    set_proxy
    ${py_version} -m pip install --upgrade pip -i https://mirror.baidu.com/pypi/simple
    ${py_version} -m pip install -r python/requirements.txt -i https://mirror.baidu.com/pypi/simple
    ${py_version} -m pip uninstall paddle-serving-app paddle-serving-client paddle-serving-server-gpu -y
}

function install_go() {
    unset_proxy
    go env -w GO111MODULE=on
    go env -w GOPROXY=https://goproxy.cn,direct
    go install github.com/grpc-ecosystem/grpc-gateway/protoc-gen-grpc-gateway@v1.15.2
    go install github.com/grpc-ecosystem/grpc-gateway/protoc-gen-swagger@v1.15.2
    go install github.com/golang/protobuf/protoc-gen-go@v1.4.3
    go install google.golang.org/grpc@v1.33.0
    go env -w GO111MODULE=auto
    set_proxy
}

function compile_server() {
    export CUDA_PATH='/usr/local/cuda'
    export CUDNN_LIBRARY='/usr/local/cuda/lib64/'
    export CUDA_CUDART_LIBRARY="/usr/local/cuda/lib64/"
    export TENSORRT_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu"
    mkdir server-build-gpu && cd server-build-gpu
    cmake -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR/ \
      -DPYTHON_LIBRARIES=$PYTHON_LIBRARIES \
      -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
      -DCMAKE_INSTALL_PREFIX=./output \
      -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_PATH} \
      -DCUDNN_LIBRARY=${CUDNN_LIBRARY} \
      -DCUDA_CUDART_LIBRARY=${CUDA_CUDART_LIBRARY} \
      -DTENSORRT_ROOT=${TENSORRT_LIBRARY_PATH} \
      -DWITH_GPU=ON \
      -DWITH_JETSON=ON \
      -DWITH_MKL=OFF \
      -DWITH_MKLML=OFF \
      -DSERVER=ON ..
    fi
    make TARGET=ARMV8 -j3
    if [ `ls -A python/dist/ | wc -w` == 0 ]; then
        echo "--------make server failed, try again"
        make TARGET=ARMV8 -j3
    fi
    if [ ! -f "./core/general-server/serving" ]; then
        echo "--------serving bin not found, exit"
        exit 1
    fi
    unset_proxy
    ${py_version} -m pip install python/dist/paddle* -i https://mirror.baidu.com/pypi/simple
    set_proxy
    cd ..
}

function compile_client() {
    mkdir client-build && cd client-build
    cmake -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR/ \
        -DPYTHON_LIBRARIES=$PYTHON_LIBRARIES \
        -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
        -DCMAKE_INSTALL_PREFIX=./output \
        -DWITH_JETSON=ON \
        -DCLIENT=ON ..
    make TARGET=ARMV8 -j3
    if [ `ls -A python/dist/ | wc -w` == 0 ]; then
        echo "--------make client failed, try again"
        make TARGET=ARMV8 -j3
    fi
    ${py_version} -m pip install python/dist/paddle* -i https://mirror.baidu.com/pypi/simple
    set_proxy
    cd ..
}

function compile_app() {
    mkdir app-build && cd app-build
    cmake -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR/ \
        -DPYTHON_LIBRARIES=$PYTHON_LIBRARIES \
        -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
        -DCMAKE_INSTALL_PREFIX=./output \
        -DWITH_JETSON=ON \
        -DAPP=ON ..
    make TARGET=ARMV8 -j3
    if [ `ls -A python/dist/ | wc -w` == 0 ]; then
        echo "--------make app failed, try again"
        make TARGET=ARMV8 -j3
    fi
    ${py_version} -m pip install python/dist/paddle* -i https://mirror.baidu.com/pypi/simple
    set_proxy
    cd ..
}


set_proxy
git submodule update --init --recursive
set_py $1 $2
install_go
compile_server $2
fi

#compile_client
#compile_app

# check output
cp server-build-gpu/python/dist/*.whl ./
#cp client-build/python/dist/*.whl ./
#cp app-build/python/dist/*.whl ./
#cp server-build-arm-xpu/*.tar.gz ./

n_whl=`ls -l ./*.whl | wc -l`
n_tar=`ls -l ./*.tar.gz | wc -l`
if [ ${n_whl} -ne 1 ] ; then
    echo "!!!!!!!!!!!!!!! compile failed, please check the result!"
    exit 1
else
    echo " ----------The num is right!"
fi
