shell_path=${CODE_PATH}/CE/cpp_ocr
cd ${shell_path}
bash compile_server_withopencv.sh $1 $2

unset http_proxy && unset https_proxy

export FLAGS_call_stack_level=2
if [ $2 == "cpu" ]; then
    export SERVING_BIN=${CODE_PATH}/Serving/server-build-cpu-opencv/core/general-server/serving
else
    export SERVING_BIN=${CODE_PATH}/Serving/server-build-gpu-opencv/core/general-server/serving
fi

# 修改det feed type
rm -rf ${CODE_PATH}/Serving/python/examples/ocr/ocr_det_client
cp -r ${DATA_PATH}/ocr/ocr_det_client ${CODE_PATH}/Serving/python/examples/ocr/
sed -i "s/feed_type: 1/feed_type: 20/g" ${CODE_PATH}/Serving/python/examples/ocr/ocr_det_client/serving_client_conf.prototxt
sed -i "s/shape: 3/shape: 1/g" ${CODE_PATH}/Serving/python/examples/ocr/ocr_det_client/serving_client_conf.prototxt
sed -i '7,8d' ${CODE_PATH}/Serving/python/examples/ocr/ocr_det_client/serving_client_conf.prototxt

rm -rf result.txt
cases=`find ./ -name "test*.py" | sort`
#cases=`find ./ -maxdepth 1 -name "test*.py" | sort`
echo $cases
# 单独在cuda10.2-cudnn7环境下验证 c++ client
if [ $2 == 1027 ]; then
    ignore=""
else
    ignore="test_cpp_client.py"
fi
ignore=""
bug=0

for test_version in python3.6 python3.7 python3.8
do
    ${test_version} -m pip install -r ${CODE_PATH}/CE/requirements.txt -i https://mirror.baidu.com/pypi/simple
    export py_version=${test_version}
    echo "========= ${test_version} start ========="
    job_bt=`date '+%Y%m%d%H%M%S'`
    echo "============ failed cases =============" >> result.txt
    for file in ${cases}
    do
        echo ${file}
        if [[ ${ignore} =~ ${file##*/} ]]; then
            echo "跳过"
        else
            if [[ $2 =~ "cpu" ]]; then
                $py_version -m pytest --disable-warnings -sv ${file} -k "cpu"
            else
                $py_version -m pytest --disable-warnings -sv ${file}
            fi
            if [[ $? -ne 0 && $? -ne 5 ]]; then
                echo "${py_version}:${file}" >> result.txt
                bug=`expr ${bug} + 1`
            fi
        fi
    done
    job_et=`date '+%Y%m%d%H%M%S'`
done

echo "total bugs: "${bug} >> result.txt
if [ ${bug} != 0 ]; then
    cp result.txt ${output_dir}/result_$2.txt
fi
cat result.txt
cost=$(expr $job_et - $job_bt)
echo "$cost s"
exit ${bug}