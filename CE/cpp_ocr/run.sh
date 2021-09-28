shell_path=${CODE_PATH}/CE/cpp_ocr
cd ${shell_path}
bash compile_server_withopencv.sh $1 $2

unset http_proxy && unset https_proxy

export FLAGS_call_stack_level=2
if [ $2 == "cpu" ]; then
    export SERVING_BIN=${CODE_PATH}/Serving/server-build-cpu/core/general-server/serving
else
    export SERVING_BIN=${CODE_PATH}/Serving/server-build-gpu/core/general-server/serving
fi
rm -rf result.txt
cases=`find ./ -name "test*.py" | sort`
#cases=`find ./ -maxdepth 1 -name "test*.py" | sort`
echo $cases
ignore=""
bug=0

for py_version in python3.6 python3.7 python3.8
do
    $py_version -m pip install -r ${CODE_PATH}/CE/requirements.txt -i https://mirror.baidu.com/pypi/simple
    echo "========= ${py_version} start ========="
    job_bt=`date '+%Y%m%d%H%M%S'`
    echo "============ failed cases =============" >> result.txt
    for file in ${cases}
    do
        echo ${file}
        if [[ ${ignore} =~ ${file##*/} ]]; then
            echo "跳过"
        else
            if [[ ${ce_name} =~ "cpu" ]]; then
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