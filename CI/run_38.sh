export FLAGS_call_stack_level=2

# check bin
if [ ! -f "${CODE_PATH}/Serving/server-build-gpu-opencv/core/general-server/serving" ]; then
    echo "compile failed!"
    exit 1
fi

rm -rf result.txt
cases="test_for_ci38.py"
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

${CODE_PATH}/Serving/python/examples/bert
python3.8 -m paddle_serving_server.serve --model bert_seq128_model/ --port 9292 --gpu_ids 0 &
sleep 30
head data-c.txt | python3.8 bert_client.py --model bert_seq128_client/serving_client_conf.prototxt
kill -9 `ps -ef | grep serving | awk '{print $2}'` > /dev/null 2>&1

exit ${bug}