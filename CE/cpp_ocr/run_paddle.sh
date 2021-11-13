shell_path=${CODE_PATH}/CE/cpp_ocr
cd ${shell_path}

# 运行serving补丁脚本
if [ $2 == "pd_cpu" ]; then
    bash -x ../paddle_docker/run_paddle_cpu.sh
    export PYTHON_INCLUDE_DIR=/usr/include/python3.7m/
    export PYTHON_LIBRARIES=/usr/lib/x86_64-linux-gnu/libpython3.7m.so
    export PYTHON_EXECUTABLE=/usr/bin/python3.7
elif [ $2 == "pd_1027" ]; then
    bash -x ../paddle_docker/run_paddle_1027.sh
    export PYTHON_INCLUDE_DIR=/usr/local/python3.7.0/include/python3.7m/
    export PYTHON_LIBRARIES=/usr/local/python3.7.0/lib/libpython3.7m.so
    export PYTHON_EXECUTABLE=/usr/local/python3.7.0/bin/python3.7
elif [ $2 == "pd_112" ]; then
    bash -x ../paddle_docker/run_paddle_112.sh
    export PYTHON_INCLUDE_DIR=/usr/include/python3.7m/
    export PYTHON_LIBRARIES=/usr/lib/x86_64-linux-gnu/libpython3.7m.so
    export PYTHON_EXECUTABLE=/usr/bin/python3.7
fi

cd ${shell_path}
bash -x compile_server_withopencv_paddle.sh $1 $2

unset http_proxy && unset https_proxy

export FLAGS_call_stack_level=2
if [ $2 == "pd_cpu" ]; then
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
if [ $2 == "pd_1027" ]; then
    ignore=""
else
    ignore="test_cpp_client.py"
fi
bug=0

# paddle镜像只有37
for test_version in python3.7
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
