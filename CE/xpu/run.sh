#!/bin/bash

shell_path=${CODE_PATH}/CE/xpu
#export https_proxy=${proxy}
#export http_proxy=${proxy}
apt-get install -y libgeos-dev
apt-get install -y net-tools

export LD_LIBRARY_PATH=/usr/local/lib/${py_version}/site-packages/paddle/libs/:/usr/local/lib/${py_version}/site-packages/paddle_serving_server/serving-xpu-x86_64-0.0.0/:$LD_LIBRARY_PATH

cd ${shell_path}
if [ $2 == "arm" ]; then
    bash pip_install.sh $1 $2
elif [ $2 == "x86" ]; then
    # CE机器环境特殊
    bash pip_install_x86_xpu.sh $1 $2
fi

unset http_proxy && unset https_proxy
# 依赖和bin
$py_version -m pip install -r ../requirements.txt -i https://mirror.baidu.com/pypi/simple
$py_version ../download_bin.py > load_bin 2>&1
tail -10 load_bin

export FLAGS_call_stack_level=2
unset SERVING_BIN
env
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
        # pytest跳过returncode=5
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



