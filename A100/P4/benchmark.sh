rm profile_log*
rm -rf resnet_log*
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGS_profile_server=1
export FLAGS_profile_client=1
export FLAGS_serving_latency=1
gpu_id=6
#save cpu and gpu utilization log
if [ -d utilization ];then
    rm -rf utilization
else
    mkdir utilization
fi
#start server
python3.6 -m paddle_serving_server.serve --model serving_server --port 9393 --thread 16 --gpu_ids $gpu_id >  elog  2>&1 &
sleep 10

#warm up
echo -e "import psutil\ncpu_utilization=psutil.cpu_percent(1,False)\nprint('CPU_UTIL(pre):', cpu_utilization)\n" > cpu_utilization.py

python3.6 benchmark.py --thread 10 --batch_size 1 --model serving_client/serving_client_conf.prototxt --request rpc > profile 2>&1
echo -e "import psutil\nimport time\nwhile True:\n\tcpu_res = psutil.cpu_percent()\n\twith open('cpu.txt', 'a+') as f:\n\t\tf.write(f'{cpu_res}\\\n')\n\ttime.sleep(0.1)" > cpu.py
for thread_num in 1 5 10 20 30 50 70
do
for batch_size in 1
do
    nvidia-smi --id=$gpu_id --query-compute-apps=used_memory --format=csv -lms 100 > gpu_memory_use.log 2>&1 &
    nvidia-smi --id=$gpu_id --query-gpu=utilization.gpu --format=csv -lms 100 > gpu_utilization.log 2>&1 &
    rm -rf cpu.txt
    python3.6 cpu.py &
    gpu_memory_pid=$!
    python3.6 benchmark.py --thread $thread_num --batch_size $batch_size --model serving_client/serving_client_conf.prototxt --request rpc > profile 2>&1
    kill `ps -ef|grep used_memory|awk '{print $2}'` > /dev/null
    kill `ps -ef|grep utilization.gpu|awk '{print $2}'` > /dev/null
    kill `ps -ef|grep cpu.py|awk '{print $2}'` > /dev/null
    echo "model_name:" resnet_v2_50
    echo "thread_num:" $thread_num
    echo "batch_size:" $batch_size
    echo "=================Done===================="
    echo "model_name:resnet_v2_50" >> profile_log_resnet_v2_50
    echo "batch_size:$batch_size" >> profile_log_resnet_v2_50
    python3.6 cpu_utilization.py >> profile_log_resnet_v2_50
    awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "CPU_UTILIZATION:", max}' cpu.txt >> profile_log_resnet_v2_50
    grep -av '^0 %' gpu_utilization.log > gpu_utilization.log.tmp
    awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "MAX_GPU_MEMORY:", max}' gpu_memory_use.log >> profile_log_resnet_v2_50
    awk -F" " '{sum+=$1} END {print "GPU_UTILIZATION:", sum/NR, sum, NR }' gpu_utilization.log.tmp >> profile_log_resnet_v2_50
    rm -rf gpu_memory_use.log gpu_utilization.log gpu_utilization.log.tmp
    python3.6 ../util/show_profile.py profile $thread_num >> profile_log_resnet_v2_50
    tail -n 10 profile >> profile_log_resnet_v2_50
    echo "" >> profile_log_resnet_v2_50
done
done

#Divided log
awk 'BEGIN{RS="\n\n"}{i++}{print > "resnet_log_"i}' profile_log_resnet_v2_50
mkdir resnet_log && mv resnet_log_* resnet_log
ps -ef|grep 'serving'|grep -v grep|cut -c 9-15 | xargs kill -9
