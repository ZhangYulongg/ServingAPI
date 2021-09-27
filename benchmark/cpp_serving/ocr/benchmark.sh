rm profile_log*
rm -rf resnet_log*
export CUDA_VISIBLE_DEVICES=0,1
export FLAGS_profile_server=1
export FLAGS_profile_client=1
export FLAGS_serving_latency=1
gpu_id=1
#save cpu and gpu utilization log
if [ -d utilization ];then
    rm -rf utilization
else
    mkdir utilization
fi
model_name="OCR"
#start server
#${py_version} -m paddle_serving_server.serve --model ocr_det_model ocr_rec_model --port 9293 --gpu_ids 1 > ${dir}server_log.txt 2>&1 &
#sleep 15

#warm up
echo -e "import psutil\ncpu_utilization=psutil.cpu_percent(1,False)\nprint('CPU_UTILIZATION:', cpu_utilization)\n" > cpu_utilization.py

${py_version} benchmark.py --thread 10 --batch_size 1 --model ocr_det_client/serving_client_conf.prototxt,ocr_rec_client/serving_client_conf.prototxt --request rpc > profile 2>&1
echo -e "import psutil\nimport time\nwhile True:\n\tcpu_res = psutil.cpu_percent()\n\twith open('cpu.txt', 'a+') as f:\n\t\tf.write(f'{cpu_res}\\\n')\n\ttime.sleep(0.1)" > cpu.py
for thread_num in 1 2 4 6 8 12 16
do
for batch_size in 1
do
    echo "#----OCR thread num: $thread_num batch size: $batch_size mode:brpc ----" >> profile_log_ocr
    nvidia-smi --id=$gpu_id --query-gpu=memory.used --format=csv -lms 100 > gpu_memory_use.log 2>&1 &
    nvidia-smi --id=$gpu_id --query-gpu=utilization.gpu --format=csv -lms 100 > gpu_utilization.log 2>&1 &
    rm -rf cpu.txt
    ${py_version} cpu.py &
    gpu_memory_pid=$!
    ${py_version} benchmark.py --thread $thread_num --batch_size $batch_size --model ocr_det_client/serving_client_conf.prototxt,ocr_rec_client/serving_client_conf.prototxt --request rpc > profile 2>&1
    kill `ps -ef|grep memory.used|awk '{print $2}'` > /dev/null
    kill `ps -ef|grep utilization.gpu|awk '{print $2}'` > /dev/null
    kill `ps -ef|grep cpu.py|awk '{print $2}'` > /dev/null
    echo "model_name:" OCR
    echo "thread_num:" $thread_num
    echo "batch_size:" $batch_size
    echo "=================Done===================="
    echo "model_name: OCR" >> profile_log_ocr
    echo "batch_size: $batch_size" >> profile_log_ocr
    ${py_version} cpu_utilization.py >> profile_log_ocr
    awk 'BEGIN {max = 0} {if(NR>1){if ($1 > max) max=$1}} END {print "CPU_UTILIZATION_MAX:", max}' cpu.txt >> profile_log_ocr
    grep -av '^0 %' gpu_utilization.log > gpu_utilization.log.tmp
    awk 'BEGIN {max = 0} {if(NR>1){if ($model_name > max) max=$model_name}} END {print "MAX_GPU_MEMORY:", max}' gpu_memory_use.log >> profile_log_ocr
    awk -F" " '{sum+=$1} END {print "GPU_UTILIZATION:", sum/NR, sum, NR }' gpu_utilization.log.tmp >> profile_log_ocr
    rm -rf gpu_memory_use.log gpu_utilization.log gpu_utilization.log.tmp
    ${py_version} ../util/show_profile.py profile $thread_num >> profile_log_ocr
    tail -n 10 profile >> profile_log_ocr
    echo "" >> profile_log_ocr
done
done

#Divided log
awk 'BEGIN{RS="\n\n"}{i++}{print > "resnet_log_"i}' profile_log_ocr
mkdir resnet_log && mv resnet_log_* resnet_log
ps -ef|grep 'serving'|grep -v grep|cut -c 9-15 | xargs kill -9
pkill nvidia-smi