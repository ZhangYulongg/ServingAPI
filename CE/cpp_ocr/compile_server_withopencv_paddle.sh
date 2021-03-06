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
#        export PYTHON_INCLUDE_DIR=/usr/include/python3.7m/
#        export PYTHON_LIBRARIES=/usr/lib/x86_64-linux-gnu/libpython3.7m.so
#        export PYTHON_EXECUTABLE=/usr/bin/python3.7
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
    if [ $1 == 101 ]; then
        export TENSORRT_LIBRARY_PATH="/usr/local/TensorRT6-cuda10.1-cudnn7/targets/x86_64-linux-gnu/"
    elif [ $1 == "pd_1027" ]; then
        export TENSORRT_LIBRARY_PATH="/usr/"
    elif [ $1 == 1028 ]; then
        export TENSORRT_LIBRARY_PATH="/usr/local/TensorRT-7.1.3.4/targets/x86_64-linux-gnu/"
    elif [ $1 == "pd_112" ]; then
        export TENSORRT_LIBRARY_PATH="/usr/"
    elif [ $1 == "pd_cpu" ]; then
        echo $1
    else
        echo -e "Error cuda version$1"
        exit 1
    fi
    if [ $1 == "pd_cpu" ]; then
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
    elif [ $1 == "pd_1027" ]; then
        export TENSORRT_LIBRARY_PATH="/usr/"
    elif [ $1 == 1028 ]; then
        export TENSORRT_LIBRARY_PATH="/usr/local/TensorRT-7.1.3.4/targets/x86_64-linux-gnu/"
    elif [ $1 == "pd_112" ]; then
        export TENSORRT_LIBRARY_PATH="/usr/"
    elif [ $1 == "pd_cpu" ]; then
        echo $1
    else
        echo -e "Error cuda version$1"
        exit 1
    fi
    rm -rf server-build*
    if [ $1 == "pd_cpu" ]; then
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
#    python3.6 -m pip install python/dist/paddle* -i https://mirror.baidu.com/pypi/simple
    python3.7 -m pip install python/dist/paddle* -i https://mirror.baidu.com/pypi/simple
#    python3.8 -m pip install python/dist/paddle* -i https://mirror.baidu.com/pypi/simple
    set_proxy
    cd ..
}

function compile_client() {
    cd ${serving_dir}
    mkdir client-build && cd client-build
    cmake -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
        -DPYTHON_LIBRARIES=$PYTHON_LIBRARIES \
        -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
        -DCLIENT=ON ..
    make -j10
    unset_proxy
    ${py_version} -m pip install python/dist/paddle* -i https://mirror.baidu.com/pypi/simple -y
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
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddlepaddle_gpu-2.2.0.post101-cp36-cp36m-linux_x86_64.whl
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddlepaddle_gpu-2.2.0.post101-cp37-cp37m-linux_x86_64.whl
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.1_cudnn7.6.5_trt6.0.1.5/paddlepaddle_gpu-2.2.0.post101-cp38-cp38-linux_x86_64.whl
    elif [ $2 == 1027 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.0/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.0-cp36-cp36m-linux_x86_64.whl
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.0/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.0-cp37-cp37m-linux_x86_64.whl
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.0/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.0-cp38-cp38-linux_x86_64.whl
    elif [ $2 == 1028 ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.0-cp36-cp36m-linux_x86_64.whl
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.0-cp37-cp37m-linux_x86_64.whl
        wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.0-cp38-cp38-linux_x86_64.whl
    elif [ $2 == 112 ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.0/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.0.post112-cp36-cp36m-linux_x86_64.whl
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.0/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.0.post112-cp37-cp37m-linux_x86_64.whl
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.0/linux/linux-gpu-cuda11.2-cudnn8-mkl-gcc8.2-avx/paddlepaddle_gpu-2.2.0.post112-cp38-cp38-linux_x86_64.whl
    elif [ $2 == "cpu" ]; then
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.0/linux/linux-cpu-mkl-avx/paddlepaddle-2.2.0-cp36-cp36m-linux_x86_64.whl
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.0/linux/linux-cpu-mkl-avx/paddlepaddle-2.2.0-cp37-cp37m-linux_x86_64.whl
        wget -q https://paddle-wheel.bj.bcebos.com/2.2.0/linux/linux-cpu-mkl-avx/paddlepaddle-2.2.0-cp38-cp38-linux_x86_64.whl
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
#    python3.6 -m pip install paddle_serving_app-${version}-py3-none-any.whl -i https://mirror.baidu.com/pypi/simple
#    python3.6 -m pip install paddle_serving_client-${version}-cp36-none-any.whl -i https://mirror.baidu.com/pypi/simple
    python3.7 -m pip install paddle_serving_app-${version}-py3-none-any.whl -i https://mirror.baidu.com/pypi/simple
    python3.7 -m pip install paddle_serving_client-${version}-cp37-none-any.whl -i https://mirror.baidu.com/pypi/simple
#    python3.8 -m pip install paddle_serving_app-${version}-py3-none-any.whl -i https://mirror.baidu.com/pypi/simple
#    python3.8 -m pip install paddle_serving_client-${version}-cp38-none-any.whl -i https://mirror.baidu.com/pypi/simple
    set_proxy
}


set_proxy
git submodule update --init --recursive
set_py $1
install_go

compile_server_withopencv $2
pip_install_app_client
#只编server，app和client直接用流水线的包
if [ $2 == "pd_1027" ]; then
    compile_client
fi
#compile_app
# 不用装paddle
#py_requirements $1 $2
