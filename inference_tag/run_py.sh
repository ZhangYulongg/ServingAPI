#!/bin/bash
# 需设置
# tag_path (2.2.0-rc0)

tag_path=${tag_path}
cuda=$1
cudnn=$2
gcc=$3
trt=$4
py=$5

function set_py() {
    if [ $1 == 36 ]; then
        export LD_LIBRARY_PATH=/opt/_internal/cpython-3.6.0/lib/:${LD_LIBRARY_PATH};
        export PATH=/opt/_internal/cpython-3.6.0/bin/:${PATH};
        export PYTHON_FLAGS='-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.6.0/bin/python3.6 -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.6.0/include/python3.6 -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.6.0/lib/libpython3.so';
        export flag="36m"
    elif [ $1 == 37 ]; then
        export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH};
        export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH};
        export PYTHON_FLAGS='-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.7.0/bin/python3.7 -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.7.0/include/python3.7 -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.7.0/lib/libpython3.so';
        export flag="37m"
    elif [ $1 == 38 ]; then
        export LD_LIBRARY_PATH=/opt/_internal/cpython-3.8.0/lib/:${LD_LIBRARY_PATH};
        export PATH=/opt/_internal/cpython-3.8.0/bin/:${PATH};
        export PYTHON_FLAGS='-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.8.0/bin/python3.8 -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.8.0/include/python3.8 -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.8.0/lib/libpython3.so';
        export flag="38"
    elif [ $1 == 39 ]; then
        export LD_LIBRARY_PATH=/opt/_internal/cpython-3.9.0/lib/:${LD_LIBRARY_PATH};
        export PATH=/opt/_internal/cpython-3.9.0/bin/:${PATH};
        export PYTHON_FLAGS='-DPYTHON_EXECUTABLE:FILEPATH=/opt/_internal/cpython-3.9.0/bin/python3.9 -DPYTHON_INCLUDE_DIR:PATH=/opt/_internal/cpython-3.9.0/include/python3.9 -DPYTHON_LIBRARIES:FILEPATH=/opt/_internal/cpython-3.9.0/lib/libpython3.so';
        export flag="39"
    fi
}

function get_whl() {
    if [ $cuda == "10.1" ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/${tag_path}/python/Linux/GPU/x86-64_gcc${gcc}_avx_mkl_cuda${cuda}_cudnn${cudnn}_trt${trt}/paddlepaddle_gpu-2.2.0rc0.post101-cp${py}-cp${flag}-linux_x86_64.whl
    elif [ $cuda == "10.2" ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/${tag_path}/python/Linux/GPU/x86-64_gcc${gcc}_avx_mkl_cuda${cuda}_cudnn${cudnn}_trt${trt}/paddlepaddle_gpu-2.2.0rc0-cp${py}-cp${flag}-linux_x86_64.whl
    elif [ $cuda == "11.1" ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/${tag_path}/python/Linux/GPU/x86-64_gcc${gcc}_avx_mkl_cuda${cuda}_cudnn${cudnn}_trt${trt}/paddlepaddle_gpu-2.2.0rc0.post111-cp${py}-cp${flag}-linux_x86_64.whl
    elif [ $cuda == "11.2" ]; then
        wget -q https://paddle-inference-lib.bj.bcebos.com/${tag_path}/python/Linux/GPU/x86-64_gcc${gcc}_avx_mkl_cuda${cuda}_cudnn${cudnn}_trt${trt}/paddlepaddle_gpu-2.2.0rc0.post112-cp${py}-cp${flag}-linux_x86_64.whl
    fi
}


set_py $py
python -m pip uninstall paddlepaddle paddlepaddle-gpu -y
rm -rf paddlepaddle*
get_whl

unset http_proxy
unset https_proxy

python -m pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
python -m pip install paddlepaddle* -i https://mirror.baidu.com/pypi/simple

ln -s /usr/lib64/libnvidia-ml.so.* /usr/lib64/libnvidia-ml.so.1;

bash -x run.sh | tee log_${cuda}_${cudnn}_${gcc}_${trt}_${py}.txt
