#!/bin/bash

model_name=$1
client_py=$2
gpu_id=$3
client_mode=$4
server_mode=$5

rm -rf profile* benchmark_logs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FLAGS_profile_server=1
export FLAGS_profile_client=1
export FLAGS_serving_latency=1
#save cpu and gpu utilization log
if [ -d utilization ];then
    rm -rf utilization
else
    mkdir utilization
fi

#start server

#warm up
echo -e "import psutil\ncpu_utilization=psutil.cpu_percent(1,False)\nprint('CPU_UTIL(pre):', cpu_utilization)\n" > cpu_utilization.py

${py_version} ${client_py} --thread 10 --batch_size 1 --request ${client_mode} > profile 2>&1
echo -e "import psutil\nimport time\nwhile True:\n\tcpu_res = psutil.cpu_percent()\n\twith open('cpu.txt', 'a+') as f:\n\t\tf.write(f'{cpu_res}\\\n')\n\ttime.sleep(0.1)" > cpu.py
for thread_num in 1 5 10 15 20 25 50 100
do
for batch_size in 1
do
    echo "#----${model_name} thread num: $thread_num batch size: $batch_size mode: ${client_mode} server_mode: ${server_mode} ----" >> profile_log_${model_name}
    nvidia-smi --id=$gpu_id --query-gpu=memory.used --format=csv -lms 100 > gpu_memory_use.log 2>&1 &
    nvidia-smi --id=$gpu_id --query-gpu=utilization.gpu --format=csv -lms 100 > gpu_utilization.log 2>&1 &
    rm -rf cpu.txt
    ${py_version} cpu.py &
    gpu_memory_pid=$!
    ${py_version} ${client_py} --thread $thread_num --batch_size $batch_size --request ${client_mode} > profile 2>&1
    kill `ps -ef|grep memory.used|awk '{print $2}'` > /dev/null
    kill `ps -ef|grep utilization.gpu|awk '{print $2}'` > /dev/null
    kill `ps -ef|grep cpu.py|awk '{print $2}'` > /dev/null
    echo "model_name:" ${model_name}
    echo "thread_num:" $thread_num
    echo "batch_size:" $batch_size
    echo "=================Done===================="
    echo "model_name: ${model_name}" >> profile_log_${model_name}
    echo "batch_size: $batch_size" >> profile_log_${model_name}
    ${py_version} cpu_utilization.py >> profile_log_${model_name}
    awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "CPU_UTILIZATION:", max}' cpu.txt >> profile_log_${model_name}
    grep -av '^0 %' gpu_utilization.log > gpu_utilization.log.tmp
    awk 'BEGIN {max = 0} {if(NR>1){if ($model_name > max) max=$model_name}} END {print "MAX_GPU_MEMORY:", max}' gpu_memory_use.log >> profile_log_${model_name}
    awk -F" " '{sum+=$1} END {print "GPU_UTILIZATION:", sum/NR, sum, NR }' gpu_utilization.log.tmp >> profile_log_${model_name}
    rm -rf gpu_memory_use.log gpu_utilization.log gpu_utilization.log.tmp
    ${py_version} show_profile.py profile $thread_num >> profile_log_${model_name}
    tail -n 10 profile >> profile_log_${model_name}
    echo "" >> profile_log_${model_name}
done
done

#Divided log
#awk 'BEGIN{RS="\n\n"}{i++}{print > "resnet_log_"i}' profile_log_${model_name}
#mkdir resnet_log && mv resnet_log_* resnet_log
ps -ef|grep 'serving'|grep -v grep|cut -c 9-15 | xargs kill -9
ps -ef|grep 'service'|grep -v grep|cut -c 9-15 | xargs kill -9
pkill nvidia-smi
