#!/bin/bash
# Serving 根目录下运行，提前设置serving_dir, proxy, opencv_dir(/mnt/serving/opencv-3.4.7/opencv3)
# $1 36、37、38
# $2 cpu、101、102、110
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
    ${py_version} -m pip install --upgrade pip==21.1.3 -i https://mirror.baidu.com/pypi/simple
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

function compile_server() {
    export CUDA_PATH='/usr/local/cuda'
    export CUDNN_LIBRARY='/usr/local/cuda/lib64/'
    export CUDA_CUDART_LIBRARY="/usr/local/cuda/lib64/"
    if [ $1 == 101 ]; then
        export TENSORRT_LIBRARY_PATH="/usr/local/TensorRT6-cuda10.1-cudnn7/targets/x86_64-linux-gnu/"
    elif [ $1 == 102 ]; then
        export TENSORRT_LIBRARY_PATH="/usr/local/TensorRT-7.1.3.4/targets/x86_64-linux-gnu/"
    elif [ $1 == 110 ]; then
        export TENSORRT_LIBRARY_PATH="/usr/local/TensorRT-7.1.3.4/targets/x86_64-linux-gnu/"
    elif [ $1 == "cpu" ]; then
        echo $1
    else
        echo -e "Error cuda version$1"
        exit 1
    fi
    if [ $1 == "cpu" ]; then
        mkdir server-build-cpu && cd server-build-cpu
        cmake -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR/ \
            -DPYTHON_LIBRARIES=$PYTHON_LIBRARIES \
            -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
            -DSERVER=ON ..
    else
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
    fi
    make -j10
    unset_proxy
    ${py_version} -m pip install python/dist/paddle* -i https://mirror.baidu.com/pypi/simple
    set_proxy
    cd ..
}

function compile_server_withopencv() {
    export OPENCV_DIR=${opencv_dir}
    echo "OPENCV_DIR:${OPENCV_DIR}"
    export CUDA_PATH='/usr/local/cuda'
    export CUDNN_LIBRARY='/usr/local/cuda/lib64/'
    export CUDA_CUDART_LIBRARY="/usr/local/cuda/lib64/"
    if [ $1 == 101 ]; then
        export TENSORRT_LIBRARY_PATH="/usr/local/TensorRT6-cuda10.1-cudnn7/targets/x86_64-linux-gnu/"
    elif [ $1 == 102 ]; then
        export TENSORRT_LIBRARY_PATH="/usr/local/TensorRT-7.1.3.4/targets/x86_64-linux-gnu/"
    elif [ $1 == 110 ]; then
        export TENSORRT_LIBRARY_PATH="/usr/local/TensorRT-7.1.3.4/targets/x86_64-linux-gnu/"
    elif [ $1 == "cpu" ]; then
        echo $1
    else
        echo -e "Error cuda version$1"
        exit 1
    fi
    rm -rf server-build*
    if [ $1 == "cpu" ]; then
        mkdir server-build-cpu-opencv && cd server-build-cpu-opencv
        cmake -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
            -DPYTHON_LIBRARIES=$PYTHON_LIBRARIES \
            -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
            -DOPENCV_DIR=${OPENCV_DIR} \
            -DWITH_OPENCV=ON \
            -DSERVER=ON ..
    else
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
    fi
    make -j10
    if [ `ls -A ./python/dist/ | wc -w` == 0 ]; then
        echo "--------make failed, try again"
        make -j10
    fi
    unset_proxy
    python3.6 -m pip install python/dist/paddle* -i https://mirror.baidu.com/pypi/simple
    python3.7 -m pip install python/dist/paddle* -i https://mirror.baidu.com/pypi/simple
    python3.8 -m pip install python/dist/paddle* -i https://mirror.baidu.com/pypi/simple
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

# 安装python依赖包
function py_requirements () {
    echo -e "${YELOW_COLOR}---------install python requirements---------${RES}"
    echo "---------Python Version: $py_version"
    unset_proxy
    if [ $2 == 101 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddlepaddle_gpu-2.1.0.post101-cp36-cp36m-linux_x86_64.whl
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddlepaddle_gpu-2.1.0.post101-cp37-cp37m-linux_x86_64.whl
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda10.1-cudnn7-mkl-gcc8.2/paddlepaddle_gpu-2.1.0.post101-cp38-cp38-linux_x86_64.whl
    elif [ $2 == 102 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda10.2-cudnn8-mkl-gcc8.2/paddlepaddle_gpu-2.1.0-cp36-cp36m-linux_x86_64.whl
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda10.2-cudnn8-mkl-gcc8.2/paddlepaddle_gpu-2.1.0-cp37-cp37m-linux_x86_64.whl
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda10.2-cudnn8-mkl-gcc8.2/paddlepaddle_gpu-2.1.0-cp38-cp38-linux_x86_64.whl
    elif [ $2 == 110 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda11.0-cudnn8-mkl-gcc8.2/paddlepaddle_gpu-2.1.0.post110-cp36-cp36m-linux_x86_64.whl
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda11.0-cudnn8-mkl-gcc8.2/paddlepaddle_gpu-2.1.0.post110-cp37-cp37m-linux_x86_64.whl
        wget -q https://paddle-wheel.bj.bcebos.com/with-trt/2.1.0-gpu-cuda11.0-cudnn8-mkl-gcc8.2/paddlepaddle_gpu-2.1.0.post110-cp38-cp38-linux_x86_64.whl
    elif [ $2 == "cpu" ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.1.0-cpu-avx-mkl/paddlepaddle-2.1.0-cp36-cp36m-linux_x86_64.whl
        wget -q https://paddle-wheel.bj.bcebos.com/2.1.0-cpu-avx-mkl/paddlepaddle-2.1.0-cp37-cp37m-linux_x86_64.whl
        wget -q https://paddle-wheel.bj.bcebos.com/2.1.0-cpu-avx-mkl/paddlepaddle-2.1.0-cp38-cp38-linux_x86_64.whl
    else
        echo -e "${RED_COLOR}Error cuda version$1${RES}"
        exit
    fi
    python3.6 -m pip install paddlepaddle*-cp36-* -i https://mirror.baidu.com/pypi/simple
    python3.7 -m pip install paddlepaddle*-cp37-* -i https://mirror.baidu.com/pypi/simple
    python3.8 -m pip install paddlepaddle*-cp38-* -i https://mirror.baidu.com/pypi/simple
    rm -rf paddlepaddle*
    set_proxy
    echo -e "${YELOW_COLOR}---------complete---------\n${RES}"
}

function pip_install_app_client() {
    unset_proxy
    if [ ${test_branch} == "develop" ]; then
        version=0.0.0
    else
        version=${test_branch: 1}
    fi
    # whl包数组
    whl_list=(app-${version}-py3 client-${version}-cp36 client-${version}-cp37 client-${version}-cp38)
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
    python3.6 -m pip install paddle_serving_app-${version}-py3-none-any.whl -i https://mirror.baidu.com/pypi/simple
    python3.6 -m pip install paddle_serving_client-${version}-cp36-none-any.whl -i https://mirror.baidu.com/pypi/simple
    python3.7 -m pip install paddle_serving_app-${version}-py3-none-any.whl -i https://mirror.baidu.com/pypi/simple
    python3.7 -m pip install paddle_serving_client-${version}-cp37-none-any.whl -i https://mirror.baidu.com/pypi/simple
    python3.8 -m pip install paddle_serving_app-${version}-py3-none-any.whl -i https://mirror.baidu.com/pypi/simple
    python3.8 -m pip install paddle_serving_client-${version}-cp38-none-any.whl -i https://mirror.baidu.com/pypi/simple
    set_proxy
}


set_proxy
git submodule update --init --recursive
set_py $1
install_go

compile_server_withopencv $2
pip_install_app_client
#只编server，app和client直接用流水线的包
#compile_client
#compile_app
py_requirements $1 $2
