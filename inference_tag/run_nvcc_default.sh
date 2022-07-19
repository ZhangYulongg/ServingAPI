echo ">>> Download model link url ..."
wget -q --no-proxy https://paddle-inference-dist.bj.bcebos.com/inference-ce/benchmark-daily/model.txt
wget -q --no-proxy https://paddle-inference-dist.bj.bcebos.com/inference-ce/benchmark-daily/npy.tgz
tar -xf npy.tgz
dir=$(pwd)
echo ">>> Download model ..."
rm -rf Models/*
cd Models
    echo ">>> ce daily benchmark model download and decompression..."
    cat ${dir}/model.txt | while read line
    do
        wget -q $line
        tar -xf *tgz
        if [ $? -eq 0 ]; then
            rm -rf *tgz
            model_name=$(ls |grep _upload)
            mv ${model_name} ${model_name%_upload*}
            echo ${model_name%_upload*}
        else
            echo "${$line} decompression failed"
        fi
    done
cd ${dir}
echo ">>> Generate yaml configuration file ..."
bash prepare_config.sh

echo ">>> Run onnx_convert and converted model diff checker ..."
bash onnx_convert_nvcc.sh

echo ">>> Run inference_benchmark ..."
bash inference_benchmark_daily_default.sh

echo ">>> generate tipc_benchmark_excel.xlsx..."
python result2xlsx.py --docker_name $1

echo ">>> Tipc benchmark done"
