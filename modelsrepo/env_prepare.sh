#!/bin/bash

export code_path=%code_path%/%teamcity.build.default.checkoutDir%
cd ${code_path}

whl_list=(
app-0.0.0-py3
#client-0.0.0-cp36
#client-0.0.0-cp37
client-0.0.0-cp38
#server-0.0.0-py3
#server_gpu-0.0.0.post101-py3
#server_gpu-0.0.0.post1027-py3
server_gpu-0.0.0.post1028-py3
#server_gpu-0.0.0.post112-py3
)

for whl_item in ${whl_list[@]}
do
    wget -q https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_${whl_item}-none-any.whl
    if [ $? -eq 0 ]; then
        echo "--------------download ${whl_item} succ"
    else
        echo "--------------download ${whl_item} failed"
    fi
done
wget -q https://paddle-inference-lib.bj.bcebos.com/2.2.0/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.0-cp36-cp36m-linux_x86_64.whl

${py_version} -m pip install --upgrade pip
${py_version} -m pip install paddle_serving_* -i https://mirror.baidu.com/pypi/simple
${py_version} -m pip install paddlepaddle_* -i https://mirror.baidu.com/pypi/simple
${py_version} -m pip install pytest -i https://mirror.baidu.com/pypi/simple
echo "server:`tail -1 /usr/local/lib/${py_version}/site-packages/paddle_serving_server/version.py`"
echo "client:`tail -1 /usr/local/lib/${py_version}/site-packages/paddle_serving_client/version.py`"
echo "app:`tail -1 /usr/local/lib/${py_version}/site-packages/paddle_serving_app/version.py`"
${py_version} ${CODE_PATH}/ServingAPI/CE/download_bin.py > load_bin 2>&1
tail -10 load_bin
