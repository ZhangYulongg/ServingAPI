#!/bin/bash

py_version=python3.8

# kill进程
function kill_process(){
    kill -9 `ps -ef | grep serving | awk '{print $2}'` > /dev/null 2>&1
    kill -9 `ps -ef | grep python | awk '{print $2}'` > /dev/null 2>&1
    sleep 1
    echo -e "process killed..."
}

cd ${CODE_PATH}

export https_proxy=${proxy}
export http_proxy=${proxy}

git clone https://github.com/PaddlePaddle/PaddleSeg.git -b develop --depth=1
unset http_proxy && unset https_proxy

${py_version} -m pip install -r PaddleSeg/requirements.txt -i https://mirror.baidu.com/pypi/simple
cd PaddleSeg/deploy/serving
# 下载模型
wget https://paddleseg.bj.bcebos.com/dygraph/demo/bisenet_demo_model.tar.gz
tar zxvf bisenet_demo_model.tar.gz
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

${py_version} -m paddle_serving_client.convert \
    --dirname ./bisenetv2_demo_model \
    --model_filename model.pdmodel \
    --params_filename model.pdiparams

# 分割示例
${py_version} -m paddle_serving_server.serve --model serving_server --thread 10 --port 9292 --ir_optim &
sleep 15
${py_version} test_serving.py --serving_client_path serving_client --serving_ip_port 127.0.0.1:9292 --image_path cityscapes_demo.png > result.txt 2>&1
cat result.txt
grep -r "The segmentation image is saved in" result.txt
if [ $? -ne 0 ]; then
    echo "PaddleSeg seg_C++_rpc_CPU failed" >> ${CODE_PATH}/result_failed.txt
    EXIT_CODE=8
else
    echo "PaddleSeg seg_C++_rpc_CPU success" >> ${CODE_PATH}/result_success.txt
fi
kill_process
