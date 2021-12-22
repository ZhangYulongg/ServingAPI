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

git clone https://github.com/PaddlePaddle/PaddleOCR.git -b dygraph --depth=1
unset http_proxy && unset https_proxy

cd PaddleOCR
${py_version} -m pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple

cd deploy/pdserving
# 下载并解压 OCR 文本检测模型
wget https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar -O ch_PP-OCRv2_det_infer.tar
tar -xf ch_PP-OCRv2_det_infer.tar
# 下载并解压 OCR 文本识别模型
wget https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar -O ch_PP-OCRv2_rec_infer.tar
tar -xf ch_PP-OCRv2_rec_infer.tar

# 转换检测模型
${py_version} -m paddle_serving_client.convert --dirname ./ch_PP-OCRv2_det_infer/ \
                                         --model_filename inference.pdmodel          \
                                         --params_filename inference.pdiparams       \
                                         --serving_server ./ppocrv2_det_serving/ \
                                         --serving_client ./ppocrv2_det_client/
# 转换识别模型
${py_version} -m paddle_serving_client.convert --dirname ./ch_PP-OCRv2_rec_infer/ \
                                         --model_filename inference.pdmodel          \
                                         --params_filename inference.pdiparams       \
                                         --serving_server ./ppocrv2_rec_serving/  \
                                         --serving_client ./ppocrv2_rec_client/

# OCR示例
${py_version} web_service.py &
sleep 20
${py_version} pipeline_http_client.py > result.txt 2>&1
cat result.txt
grep -r "GB18401-2010" result.txt
if [ $? -ne 0 ]; then
    echo "PaddleOCR ocr_Pipeline_http_CPU failed" >> ${CODE_PATH}/result_failed.txt
    EXIT_CODE=8
else
    echo "PaddleOCR ocr_Pipeline_http_CPU success" >> ${CODE_PATH}/result_success.txt
fi
kill_process
