#!/bin/bash
# Serving 根目录下运行，提前设置serving_dir, proxy, opencv_dir(/mnt/serving/opencv-3.4.7/opencv3)
# $1 36、37、38
# $2 cpu、101、102、110、arm
# $3 opencv
export serving_dir=${CODE_PATH}/Serving
export bin_folder=serving-xpu-aarch64-${VERSION_TAG}

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
        if [ $2 == "arm" ]; then
            export PYTHONROOT=/usr/
        else
            export PYTHONROOT=/usr/local/
        fi
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
    set_proxy
    ${py_version} -m pip install --upgrade pip==21.1.3 -i https://mirror.baidu.com/pypi/simple
    ${py_version} -m pip install -r python/requirements.txt -i https://mirror.baidu.com/pypi/simple
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
    elif [ $1 == 1027 ] || [ $1 == 1028 ]; then
        export TENSORRT_LIBRARY_PATH="/usr/local/TensorRT-7.1.3.4/targets/x86_64-linux-gnu/"
    elif [ $1 == 112 ]; then
        export TENSORRT_LIBRARY_PATH="/home/TensorRT-8.0.3.4/targets/x86_64-linux-gnu/"
    elif [ $1 == "cpu" ] || [ $1 == "arm" ]; then
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
    elif [ $1 == "arm" ]; then
        mkdir server-build-arm-xpu && cd server-build-arm-xpu
        cmake -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} \
            -DPYTHON_LIBRARIES=${PYTHON_LIBRARIES} \
            -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} \
            -DWITH_PYTHON=ON \
            -DWITH_LITE=ON \
            -DWITH_XPU=ON \
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
    make -j32
    if [ `ls -A python/dist/ | wc -w` == 0 ]; then
        echo "--------make server failed, try again"
        make -j32
    fi
    if [ ! -f "./core/general-server/serving" ]; then
        echo "--------serving bin not found, exit"
        exit 1
    fi
    # 打包bin
    mkdir ${bin_folder}
    cp core/general-server/serving output/
    cp third_party/install/Paddle/third_party/install/xpu/lib/libxpuapi.so output/
    cp third_party/install/Paddle/third_party/install/xpu/lib/libxpurt.so output/
    cp third_party/install/Paddle/third_party/install/lite/cxx/lib/libpaddle_full_api_shared.so output/
    tar -zcvf ${bin_folder}.tar.gz ${bin_folder}
#    unset_proxy
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
    elif [ $1 == 1027 ] || [ $1 == 1028 ]; then
        export TENSORRT_LIBRARY_PATH="/usr/local/TensorRT-7.1.3.4/targets/x86_64-linux-gnu/"
    elif [ $1 == 112 ]; then
        export TENSORRT_LIBRARY_PATH="/home/TensorRT-8.0.3.4/targets/x86_64-linux-gnu/"
    elif [ $1 == "cpu" ]; then
        echo $1
    else
        echo -e "Error cuda version$1"
        exit 1
    fi
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
    if [ `ls -A python/dist/ | wc -w` == 0 ]; then
        echo "--------make server failed, try again"
        make -j10
    fi
#    unset_proxy
    ${py_version} -m pip install python/dist/paddle* -i https://mirror.baidu.com/pypi/simple
    set_proxy
    cd ..
}

function compile_client() {
    mkdir client-build && cd client-build
    cmake -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
        -DPYTHON_LIBRARIES=$PYTHON_LIBRARIES \
        -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
        -DWITH_PYTHON=ON \
        -DWITH_LITE=ON \
        -DWITH_XPU=ON \
        -DCLIENT=ON ..
    make -j32
    if [ `ls -A python/dist/ | wc -w` == 0 ]; then
        echo "--------make client failed, try again"
        make -j32
    fi
    ${py_version} -m pip install python/dist/paddle* -i https://mirror.baidu.com/pypi/simple
    set_proxy
    cd ..
}

function compile_app() {
    mkdir app-build && cd app-build
    cmake -DPYTHON_INCLUDE_DIR=$PYTHON_INCLUDE_DIR \
        -DPYTHON_LIBRARIES=$PYTHON_LIBRARIES \
        -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
        -DWITH_PYTHON=ON \
        -DWITH_LITE=ON \
        -DWITH_XPU=ON \
        -DAPP=ON ..
    make -j32
    if [ `ls -A python/dist/ | wc -w` == 0 ]; then
        echo "--------make app failed, try again"
        make -j32
    fi
    ${py_version} -m pip install python/dist/paddle* -i https://mirror.baidu.com/pypi/simple
    set_proxy
    cd ..
}


set_proxy
git submodule update --init --recursive
set_py $1 $2
install_go
if [ $3 == "opencv" ]; then
    compile_server_withopencv $2
else
    compile_server $2
fi

compile_client
compile_app

# check output
cp server-build-arm-xpu/python/dist/*.whl ./
cp client-build/python/dist/*.whl ./
cp app-build/python/dist/*.whl ./
cp server-build-arm-xpu/*.tar.gz ./

n_whl=`ls -l ./*.whl | wc -l`
n_tar=`ls -l ./*.tar.gz | wc -l`
if [ ${n_whl} -ne 3 ] || [ ${n_tar} -ne 1 ]; then
    echo "!!!!!!!!!!!!!!! compile failed, please check the result!"
    exit 1
else
    echo " ----------The num is right!"
fi
