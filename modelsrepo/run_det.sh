#!/bin/bash

py_version=python3.8
unset LANG
unset PYTHONIOENCODING

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

git clone https://github.com/PaddlePaddle/PaddleDetection.git -b develop --depth=1
unset http_proxy && unset https_proxy

cd PaddleDetection
${py_version} -m pip install Cython -i https://mirror.baidu.com/pypi/simple
${py_version} -m pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple

# 导出模型
${py_version} tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams --export_serving_model=True
# 进入到导出模型文件夹
cd output_inference/yolov3_darknet53_270e_coco/

# GPU case
${py_version} -m paddle_serving_server.serve --model serving_server --port 9393 --gpu_ids 0 &
sleep 20
${py_version} ../../deploy/serving/test_client.py ../../demo/000000014439.jpg > gpu_result.txt 2>&1
cat gpu_result.txt
grep -r "multiclass_nms3_0.tmp_0" gpu_result.txt
if [ $? -ne 0 ]; then
    echo "PaddleDetection det_C++_rpc_GPU failed" >> ${CODE_PATH}/result_failed.txt
    EXIT_CODE=8
else
    echo "PaddleDetection det_C++_rpc_GPU success" >> ${CODE_PATH}/result_success.txt
fi
kill_process

# CPU case
${py_version} -m paddle_serving_server.serve --model serving_server --port 9393 &
sleep 10
${py_version} ../../deploy/serving/test_client.py ../../demo/000000014439.jpg > cpu_result.txt 2>&1
cat cpu_result.txt
grep -r "multiclass_nms3_0.tmp_0" cpu_result.txt
if [ $? -ne 0 ]; then
    echo "PaddleDetection det_C++_rpc_CPU failed" >> ${CODE_PATH}/result_failed.txt
    EXIT_CODE=8
else
    echo "PaddleDetection det_C++_rpc_CPU success" >> ${CODE_PATH}/result_success.txt
fi
kill_process
