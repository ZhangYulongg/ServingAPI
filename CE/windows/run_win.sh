export FLAGS_call_stack_level=2
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
        python -m pytest --disable-warnings -sv ${file}
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