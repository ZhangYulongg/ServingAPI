export FLAGS_call_stack_level=2
rm -rf result.txt
cases=`find ./ -name "test*.py" | sort`
#cases=`find ./ -maxdepth 1 -name "test*.py" | sort`
echo $cases
ignore="test_ocr_concate.py \
test_bert_xpu.py \
test_ernie_xpu.py \
test_fit_a_line_xpu.py \
test_resnet_v2_50_xpu.py \
test_vgg19_xpu.py \
test_cpp_client.py \
test_ocr_win.py \
test_pp_shitu.py" # 需要编译opencv才能跑或XPU case或需要编译拿到simple_client
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
            $py_version -m pytest --disable-warnings -sv ${file} -k "cpu" --alluredir=report
        else
            $py_version -m pytest --disable-warnings -sv ${file}  --alluredir=report
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