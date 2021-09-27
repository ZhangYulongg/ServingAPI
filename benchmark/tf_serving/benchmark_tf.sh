rm -rf profile_log* tf_serving.log $1_log
export CUDA_VISIBLE_DEVICES=1
export FLAGS_profile_server=1
export FLAGS_profile_client=1
export FLAGS_serving_latency=1
cur_dir=$(cd `dirname $0`; pwd)
model_name=ResNet
echo ${cur_dir}
tensorflow_model_server --port=8500 --enable_batching=true --model_name="serving_default" --model_base_path="/mnt/serving/zyl/tf_serving/tf_test/model/resnet_v1" --batching_parameters_file="batch_config" > tf_serving.log &
sleep 10

#save cpu and gpu utilization log
if [ -d utilization ];then
    rm -rf utilization
else
    mkdir utilization
fi

#warm up
/usr/bin/python3 benchmark_tf.py --thread 4 --batch_size 1 --request rpc > profile 2>&1
sleep 3
echo -e "import psutil\ncpu_utilization=psutil.cpu_percent(1,False)\nprint('CPU_UTILIZATION:', cpu_utilization)\n" > cpu_utilization.py

for thread_num in 1 2 4 6 8 12 16
do
for batch_size in 1
do
    echo "#----ResNet thread num: $thread_num batch size: $batch_size mode:grpc ----" >> profile_log_$1
    nvidia-smi --id=1 --query-compute-apps=used_memory --format=csv -lms 100 > gpu_use.log 2>&1 &
    nvidia-smi --id=1 --query-gpu=utilization.gpu --format=csv -lms 100 > gpu_utilization.log 2>&1 &
    /usr/bin/python3 cpu_utilization.py >> profile_log_$1
    gpu_memory_pid=$!
    /usr/bin/python3 benchmark_tf.py --thread $thread_num --batch_size $batch_size --request rpc > profile 2>&1
    kill ${gpu_memory_pid}
    kill `ps -ef|grep used_memory|awk '{print $2}'`
    echo "model name :" $1
    echo "thread_num: " $thread_num
    echo "batch_size: " $batch_size
    echo "thread_num: " $thread_num >> profile_log_$1
    echo "batch_size: " $batch_size >> profile_log_$1
    echo "=================Done===================="
    echo "model name: $1" >> profile_log
    echo "batch size: $batch_size" >> profile_log
    awk 'BEGIN {max = 0} {if(NR>1){if ($model_name > max) max=$model_name}} END {print "MAX_GPU_MEMORY:", max}' gpu_use.log >> profile_log_$1
    grep -av '^0 %' gpu_utilization.log > gpu_utilization.log.tmp
    awk -F" " '{sum+=$1} END {print "GPU_UTILIZATION:", sum/NR, sum, NR }' gpu_utilization.log.tmp >> profile_log_$1
    rm -rf gpu_memory_use.log gpu_utilization.log gpu_utilization.log.tmp
    tail -n 9 profile >> profile_log_$1
    echo "" >> profile_log_$1
    sleep 3
done
done

#Divided log
awk 'BEGIN{RS="\n\n"}{i++}{print > "ResNet_log_"i}' profile_log_$1
mkdir $1_log && mv ResNet_log_* $1_log
ps -ef|grep 'tensorflow_model_server'|grep -v grep|cut -c 9-15 | xargs kill -9