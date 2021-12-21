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

git clone https://github.com/PaddlePaddle/PaddleClas.git -b develop --depth=1
unset http_proxy && unset https_proxy

${py_version} -m pip install -r PaddleClas/requirements.txt -i https://mirror.baidu.com/pypi/simple
cd PaddleClas/deploy/paddleserving
# 下载并解压 ResNet50_vd 模型
wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_vd_infer.tar
tar xf ResNet50_vd_infer.tar
# 转换 ResNet50_vd 模型
${py_version} -m paddle_serving_client.convert --dirname ./ResNet50_vd_infer/ \
                                         --model_filename inference.pdmodel  \
                                         --params_filename inference.pdiparams \
                                         --serving_server ./ResNet50_vd_serving/ \
                                         --serving_client ./ResNet50_vd_client/
# 修改fetch_var
sed -i '12d' ResNet50_vd_serving/serving_server_conf.prototxt
sed -i '12 i \ \ alias_name: "prediction"' ResNet50_vd_serving/serving_server_conf.prototxt

# 分类示例
${py_version} classification_web_service.py &
sleep 20
${py_version} pipeline_http_client.py > http_result.txt
grep -r "0.9341" http_result.txt
if [ $? -ne 0 ]; then
    echo "PaddleClas clas_Pipeline_http failed" >> ${CODE_PATH}/result_failed.txt
    EXIT_CODE=8
else
    echo "PaddleClas clas_Pipeline_http success" >> ${CODE_PATH}/result_success.txt
fi
${py_version} pipeline_rpc_client.py > rpc_result.txt
grep -r "0.9341" rpc_result.txt
if [ $? -ne 0 ]; then
    echo "PaddleClas clas_Pipeline_rpc failed" >> ${CODE_PATH}/result_failed.txt
    EXIT_CODE=8
else
    echo "PaddleClas clas_Pipeline_rpc success" >> ${CODE_PATH}/result_success.txt
fi
kill_process

# 识别示例
cd ${CODE_PATH}/PaddleClas/deploy
# 下载并解压通用识别模型
wget -q -P models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/general_PPLCNet_x2_5_lite_v1.0_infer.tar
cd models
tar -xf general_PPLCNet_x2_5_lite_v1.0_infer.tar
# 下载并解压通用检测模型
wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
tar -xf picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer.tar
# 转换识别模型
${py_version} -m paddle_serving_client.convert --dirname ./general_PPLCNet_x2_5_lite_v1.0_infer/ \
                                         --model_filename inference.pdmodel  \
                                         --params_filename inference.pdiparams \
                                         --serving_server ./general_PPLCNet_x2_5_lite_v1.0_serving/ \
                                         --serving_client ./general_PPLCNet_x2_5_lite_v1.0_client/
# 修改fetch_var
sed -i '12d' general_PPLCNet_x2_5_lite_v1.0_serving/serving_server_conf.prototxt
sed -i '12 i \ \ alias_name: "features"' general_PPLCNet_x2_5_lite_v1.0_serving/serving_server_conf.prototxt
# 转换通用检测模型
${py_version} -m paddle_serving_client.convert --dirname ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_infer/ \
                                         --model_filename inference.pdmodel  \
                                         --params_filename inference.pdiparams \
                                         --serving_server ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_serving/ \
                                         --serving_client ./picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/
# 下载并解压已经构建后的检索库index
cd ../
wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/data/drink_dataset_v1.0.tar && tar -xf drink_dataset_v1.0.tar
# 部署和请求
cd ${CODE_PATH}/PaddleClas/deploy/paddleserving/recognition
${py_version} recognition_web_service.py &
sleep 20
${py_version} pipeline_http_client.py > http_result.txt 2>&1
grep -r "rec_scores" http_result.txt
cat http_result.txt
if [ $? -ne 0 ]; then
    echo "PaddleClas recognition_Pipeline_http failed" >> ${CODE_PATH}/result_failed.txt
    EXIT_CODE=8
else
    echo "PaddleClas recognition_Pipeline_http success" >> ${CODE_PATH}/result_success.txt
fi
${py_version} pipeline_rpc_client.py > rpc_result.txt 2>&1
grep -r "rec_scores" rpc_result.txt
cat rpc_result.txt
if [ $? -ne 0 ]; then
    echo "PaddleClas recognition_Pipeline_rpc failed" >> ${CODE_PATH}/result_failed.txt
    EXIT_CODE=8
else
    echo "PaddleClas recognition_Pipeline_rpc success" >> ${CODE_PATH}/result_success.txt
fi
kill_process
