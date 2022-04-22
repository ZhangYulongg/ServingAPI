export FLAGS_call_stack_level=2

# check bin
if [ ! -f "${CODE_PATH}/Serving/server-build-gpu-opencv/core/general-server/serving" ]; then
    echo "compile failed!"
    exit 1
fi

# add libpaddle_inference.so
export LD_LIBRARY_PATH=${CODE_PATH}/Serving/server-build-gpu-opencv/third_party/Paddle/src/extern_paddle/paddle/lib/:$LD_LIBRARY_PATH
ldconfig

rm -rf result.txt
cases=`find ./ -name "test*.py" | sort`
#cases=`find ./ -maxdepth 1 -name "test*.py" | sort`
echo $cases
ignore=""
bug=0

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
            echo ${file} >> result.txt
            bug=`expr ${bug} + 1`
        fi
    fi
done
job_et=`date '+%Y%m%d%H%M%S'`

echo "total bugs: "${bug} >> result.txt
if [ ${bug} != 0 ]; then
    cp result.txt ${output_dir}/result_${py_version}.txt
fi
cat result.txt
cost=$(expr $job_et - $job_bt)
echo "$cost s"
exit ${bug}