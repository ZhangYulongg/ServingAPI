#!/bin/bash

# 日志目录
log_dir=${CODE_PATH}/logs/
demo_dir=${CODE_PATH}/Serving/python/examples/
dir=`pwd`
# 数据目录
data=/mnt/serving/dataset/data/
# 输出颜色
RED_COLOR='\E[1;31m'  #红
GREEN_COLOR='\E[1;32m' #绿
YELOW_COLOR='\E[1;33m' #黄
RES='\E[0m'
# 异常词
error_words="Fail|DENIED|None"
# cuda版本
cuda_version=`cat /usr/local/cuda/version.txt`
if [ $? -ne 0 ]; then
  cuda_version=11.0
fi

# 获取运行参数
if [ $# == 2 ]
then
  echo -e "${YELOW_COLOR}Selected python version is py$1${RES}"
else
  echo "Usage: `basename $0` first secend"
  echo "first(py_version): 36 37 38"
  echo "second(cuda_version): cpu 101 102 110"
  echo "You provided $# parameters,but 2 are required."
  exit
fi

# 设置python版本
function set_py () {
  if [ $1 == 36 ]; then
    export PYTHONROOT=/usr/local/
    export PYTHON_INCLUDE_DIR=$PYTHONROOT/include/python3.6m
    export PYTHON_LIBRARIES=$PYTHONROOT/lib/libpython3.6m.so
    export PYTHON_EXECUTABLE=$PYTHONROOT/bin/python3.6
    py_version="python3.6"
  elif [ $1 == 37 ]; then
    export PYTHONROOT=/usr/local/
    export PYTHON_INCLUDE_DIR=$PYTHONROOT/include/python3.7m
    export PYTHON_LIBRARIES=$PYTHONROOT/lib/libpython3.7m.so
    export PYTHON_EXECUTABLE=$PYTHONROOT/bin/python3.7
    py_version="python3.7"
  elif [ $1 == 38 ]; then
    export PYTHONROOT=/usr/local/
    export PYTHON_INCLUDE_DIR=$PYTHONROOT/include/python3.8
    export PYTHON_LIBRARIES=$PYTHONROOT/lib/libpython3.8.so
    export PYTHON_EXECUTABLE=$PYTHONROOT/bin/python3.8
    py_version="python3.8"
  fi
}

# kill进程
function kill_process () {
  kill `ps -ef | grep serving | awk '{print $2}'` > /dev/null 2>&1
  kill `ps -ef | grep python | awk '{print $2}'` > /dev/null 2>&1
  echo -e "${GREEN_COLOR}process killed...${RES}"
}

function prepare () {
  # 准备
  dir=${log_dir}$2/$1/
  enter_dir $1/
  check_dir ${dir}
}

# 进入目录
function enter_dir () {
  cd ${demo_dir}$1
  pwd
}

# check命令并保存日志
function check_save () {
  if [ $? == 0 ]
  then
  	echo -e "${GREEN_COLOR}$1 execute normally${RES}"
  	# server端
  	if [ $1 == "server" ]
  	then
      sleep $2
      cat ${dir}server_log.txt | tee -a ${log_dir}server_total.txt
    fi
    # client端
    if [ $1 == "client" ]
    then
      cat ${dir}client_log.txt | tee -a ${log_dir}client_total.txt
      # 检查日志异常词
      grep -E "${error_words}" ${dir}client_log.txt > /dev/null
      if [ $? == 0 ]; then
        # 包含关键词，预测报错
        echo -e "${RED_COLOR}$1 error command${RES}\n" | tee -a ${log_dir}server_total.txt ${log_dir}client_total.txt
        # 记录模型、部署方式、环境信息
        error_log $2
      else
        # 不含关键词，正常预测
        echo -e "${GREEN_COLOR}$2${RES}\n" | tee -a ${log_dir}server_total.txt ${log_dir}client_total.txt
      fi
    fi
  else
    echo -e "${RED_COLOR}$1 error command${RES}\n" | tee -a ${log_dir}server_total.txt ${log_dir}client_total.txt
    cat ${dir}client_log.txt | tee -a ${log_dir}client_total.txt
    # 记录模型、部署方式、环境信息
    error_log $2
  fi
}

# 提取错误日志
function error_log () {
  # 记录模型、部署方式、环境信息
  arg=${1//\//_} # 将所有/替换为_
  echo "-----------------------------" | tee -a ${log_dir}error_models.txt
  arg=${arg%% *}
  arr=(${arg//_/ })
  if [ ${arr[@]: -1} == 1 -o ${arr[@]: -1} == 2 ]; then
    model=${arr[*]:0:${#arr[*]}-3}
    deployment=${arr[*]: -3}
  else
    model=${arr[*]:0:${#arr[*]}-2}
    deployment=${arr[*]: -2}
  fi
  echo "model: ${model// /_}" | tee -a ${log_dir}error_models.txt
  echo "deployment: ${deployment// /_}" | tee -a ${log_dir}error_models.txt
  echo "py_version: ${py_version}" | tee -a ${log_dir}error_models.txt
  echo "cuda_version: ${cuda_version}" | tee -a ${log_dir}error_models.txt
  echo "status: Failed" | tee -a ${log_dir}error_models.txt
  echo -e "-----------------------------\n\n" | tee -a ${log_dir}error_models.txt
  # 日志汇总
  prefix=${arg//\//_}
  for file in ${dir}*
  do
  	cp ${file} ${log_dir}error/${prefix}_${file##*/}
  done
}

# 判断目录是否存在
function check_dir () {
  if [ ! -d "$1" ]
  then
  	mkdir -p $1
  fi
}

# 链接模型数据
function link_data () {
  for file in $1*
  do
  	if [ ! -h ${file##*/} ]
  	then
  	  ln -s ${file} ./${file##*/}
  	fi
  done
}

function prepare_bert_model () {
  # 准备
  prepare bert cpu_rpc
  # 下载模型
  $py_version prepare_model.py 128
  if [ $? == 0 ]
  then
    echo -e "${GREEN_COLOR}bert model download successfully${RES}" | tee -a ${log_dir}server_total.txt
  else
    echo -e "${RED_COLOR}bert model download failed${RES}\n" | tee -a ${log_dir}server_total.txt
  fi
}

function bert_cpu_rpc () {
  # 准备
  prepare bert cpu_rpc
  # 链接模型数据
  data_dir=${data}bert/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}bert_CPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model bert_seq128_model/ --port 9292 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}bert_CPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  head data-c.txt | $py_version bert_client.py --model bert_seq128_client/serving_client_conf.prototxt > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "bert_CPU_RPC server test completed"
  # kill进程
  kill_process
}

function blazeface_cpu_rpc () {
	# 准备
  prepare blazeface cpu_rpc
  # 链接模型数据
  data_dir=${data}blazeface/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}blazeface_CPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model serving_server --port 9494 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 12
  # 预测
  # 重置预测结果
  rm -rf output/
  echo -e "${GREEN_COLOR}blazeface_CPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_client.py serving_client/serving_client_conf.prototxt test.jpg > ${dir}client_log.txt 2>&1
  # 命令检查
  if [ $? != 0 -o ! -d "output/" ]; then
    echo -e "${RED_COLOR}$1 error command${RES}\n" | tee -a ${log_dir}server_total.txt ${log_dir}client_total.txt
    error_log blazeface_CPU_RPC
  fi
  tail ${dir}client_log.txt | tee -a ${log_dir}client_total.txt
  echo -e "${GREEN_COLOR}blazeface_CPU_RPC server test completed${RES}\n" | tee -a ${log_dir}server_total.txt ${log_dir}client_total.txt
  # kill进程
  kill_process
}

function criteo_ctr_cpu_rpc () {
	# 准备
  prepare criteo_ctr cpu_rpc
  # 链接模型数据
  data_dir=${data}criteo_ctr/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}criteo_ctr_CPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model ctr_serving_model/ --port 9292 > ${dir}server_log.txt 2>&1 &
  check_save server 8
  # 预测
  echo -e "${GREEN_COLOR}criteo_ctr_CPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_client.py ctr_client_conf/serving_client_conf.prototxt raw_data/part-0 > ${dir}client_log.txt 2>&1
  check_save client "criteo_ctr_CPU_RPC server test completed"
  # kill进程
  kill_process
}

function encryption_cpu_rpc () {
	# 准备
	prepare encryption cpu_rpc
	# 链接模型数据
  data_dir=${data}encryption/
  link_data ${data_dir}
  # 模型加密
  $py_version encrypt.py
	# 启动服务
  echo -e "${GREEN_COLOR}encryption_CPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model encrypt_server/ --port 9300 --use_encryption_model > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 6
  # 预测
  echo -e "${GREEN_COLOR}encryption_CPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_client.py uci_housing_client/serving_client_conf.prototxt > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "encryption_CPU_RPC server test completed"
  # kill进程
  kill_process
}

function fit_a_line_cpu_rpc () {
	# 准备
	prepare fit_a_line cpu_rpc
	# 链接模型数据
  data_dir=${data}fit_a_line/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}fit_a_line_CPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model uci_housing_model --thread 10 --port 9393 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 6
  # 预测
  echo -e "${GREEN_COLOR}fit_a_line_CPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_client.py uci_housing_client/serving_client_conf.prototxt > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "fit_a_line_CPU_RPC server test completed"
  # kill进程
  kill_process
}

function imagenet_cpu_rpc () {
	# 准备
	prepare imagenet cpu_rpc
	# 链接模型数据
  data_dir=${data}imagenet/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}imagenet_CPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model ResNet50_vd_model --port 9696 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}imagenet_CPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version resnet50_rpc_client.py ResNet50_vd_client_config/serving_client_conf.prototxt > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "imagenet_CPU_RPC server test completed"
  # kill进程
  kill_process
}

function imdb_cpu_rpc () {
	# 准备
	prepare imdb cpu_rpc
	# 链接模型数据
  data_dir=${data}imdb/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}imdb_CPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model imdb_cnn_model/ --port 9292 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}imdb_CPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  head test_data/part-0 | $py_version test_client.py imdb_cnn_client_conf/serving_client_conf.prototxt imdb.vocab > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "imdb_CPU_RPC server test completed"
  # kill进程
  kill_process
}

function lac_cpu_rpc () {
	# 准备
	prepare lac cpu_rpc
	# 链接模型数据
  data_dir=${data}lac/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}lac_CPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model lac_model/ --port 9292 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}lac_CPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  echo "我爱北京天安门" | $py_version lac_client.py lac_client/serving_client_conf.prototxt > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "lac_CPU_RPC server test completed"
  # kill进程
  kill_process
}

function ocr_cpu_rpc () {
	# 准备
	dir=${log_dir}cpu_rpc/$1/
  enter_dir ocr/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}ocr/
  link_data ${data_dir}
  # 启动服务
  if [ $1 == "ocr" ]
  then
    echo -e "${GREEN_COLOR}$1_CPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
    $py_version -m paddle_serving_server.serve --model ocr_det_model --port 9293 > ${dir}server_log.txt 2>&1 &
    check_save server 5
    $py_version ocr_web_server.py cpu >> ${dir}server_log.txt 2>&1 &
    sleep 5
  elif [ $1 == "ocr/local" ]
  then
  	echo -e "${GREEN_COLOR}$1_CPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
    $py_version ocr_debugger_server.py cpu > ${dir}server_log.txt 2>&1 &
  elif [ $1 == "ocr/only_det" ]
  then
  	echo -e "${GREEN_COLOR}$1_CPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
    $py_version det_web_server.py cpu > ${dir}server_log.txt 2>&1 &
  elif [ $1 == "ocr/only_det_local" ]
  then
  	echo -e "${GREEN_COLOR}$1_CPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
    $py_version det_debugger_server.py cpu > ${dir}server_log.txt 2>&1 &
  elif [ $1 == "ocr/only_rec" ]
  then
  	echo -e "${GREEN_COLOR}$1_CPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
    $py_version rec_web_server.py cpu > ${dir}server_log.txt 2>&1 &
  elif [ $1 == "ocr/only_rec_local" ]
  then
  	echo -e "${GREEN_COLOR}$1_CPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
    $py_version rec_debugger_server.py cpu > ${dir}server_log.txt 2>&1 &
  fi
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}$1_CPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  if [ $2 == 1 ]
  then
  	$py_version ocr_web_client.py > ${dir}client_log.txt 2>&1
  elif [ $2 == 2 ]
  then
  	$py_version rec_web_client.py > ${dir}client_log.txt 2>&1
  fi
  # 命令检查
  check_save client "$1_CPU_RPC server test completed"
  # kill进程
  kill_process
}

function bert_cpu_http () {
	# 准备
  prepare bert cpu_http
  # 链接模型数据
  data_dir=${data}bert/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}bert_CPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version bert_web_service.py bert_seq128_model/ 9292 > ${dir}server_log.txt 2>&1 &
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}bert_CPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  curl -H "Content-Type:application/json" -X POST -d '{"feed":[{"words": "hello"}], "fetch":["pooled_output"]}' http://127.0.0.1:9292/bert/prediction > ${dir}client_log.txt 2>&1
  check_save client "bert_CPU_HTTP server test completed"
  # kill进程
  kill_process
}

function fit_a_line_cpu_http () {
	# 准备
	prepare fit_a_line cpu_http
	# 链接模型数据
  data_dir=${data}fit_a_line/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}fit_a_line_CPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model uci_housing_model --thread 10 --port 9393 --name uci > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 6
  # 预测
  echo -e "${GREEN_COLOR}fit_a_line_CPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  curl -H "Content-Type:application/json" -X POST -d '{"feed":[{"x": [0.0137, -0.1136, 0.2553, -0.0692, 0.0582, -0.0727, -0.1583, -0.0584, 0.6283, 0.4919, 0.1856, 0.0795, -0.0332]}], "fetch":["price"]}' http://127.0.0.1:9393/uci/prediction > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "fit_a_line_CPU_HTTP server test completed"
  # kill进程
  kill_process
}

function imagenet_cpu_http () {
	# 准备
	prepare imagenet cpu_http
	# 链接模型数据
  data_dir=${data}imagenet/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}imagenet_CPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version resnet50_web_service.py ResNet50_vd_model cpu 9696 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}imagenet_CPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  curl -H "Content-Type:application/json" -X POST -d '{"feed":[{"image": "https://paddle-serving.bj.bcebos.com/imagenet-example/daisy.jpg"}], "fetch": ["score"]}' http://127.0.0.1:9696/image/prediction > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "imagenet_CPU_HTTP server test completed"
  # kill进程
  kill_process
}

function imdb_cpu_http () {
	# 准备
	prepare imdb cpu_http
	# 链接模型数据
  data_dir=${data}imdb/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}imdb_CPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version text_classify_service.py imdb_cnn_model/ workdir/ 9292 imdb.vocab > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}imdb_CPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  curl -H "Content-Type:application/json" -X POST -d '{"feed":[{"words": "i am very sad | 0"}], "fetch":["prediction"]}' http://127.0.0.1:9292/imdb/prediction > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "imdb_CPU_HTTP server test completed"
  # kill进程
  kill_process
}

function lac_cpu_http () {
	# 准备
	prepare lac cpu_http
	# 链接模型数据
  data_dir=${data}lac/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}lac_CPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version lac_web_service.py lac_model/ lac_workdir 9292 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}lac_CPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  curl -H "Content-Type:application/json" -X POST -d '{"feed":[{"words": "我爱北京天安门"}], "fetch":["word_seg"]}' http://127.0.0.1:9292/lac/prediction > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "lac_CPU_HTTP server test completed"
  # kill进程
  kill_process
}

function senta_cpu_http () {
	# 准备
  dir=${log_dir}cpu_http/senta/
  enter_dir senta/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}senta/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}senta_CPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model lac_model --port 9300 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  echo -e "\n\n" >> ${dir}server_log.txt
  $py_version senta_web_service.py >> ${dir}server_log.txt 2>&1 &
  check_save server 8
  # 预测
  echo -e "${GREEN_COLOR}senta_CPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  curl -H "Content-Type:application/json" -X POST -d '{"feed":[{"words": "天气不错"}], "fetch":["class_probs"]}' http://127.0.0.1:9393/senta/prediction > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "senta_CPU_HTTP server test completed"
  # kill进程
  kill_process
}

function bert_gpu_rpc () {
	# 准备
  prepare bert gpu_rpc
  # 链接模型数据
  data_dir=${data}bert/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}bert_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model bert_seq128_model/ --port 9292 --gpu_ids 0 > ${dir}server_log.txt 2>&1 &
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}bert_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  head data-c.txt | $py_version bert_client.py --model bert_seq128_client/serving_client_conf.prototxt > ${dir}client_log.txt 2>&1
  check_save client "bert_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function cascade_rcnn_gpu_rpc () {
	# 准备
  prepare cascade_rcnn gpu_rpc
  # 链接模型数据
  data_dir=${data}cascade_rcnn/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}cascade_rcnn_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model serving_server --port 9292 --gpu_id 0 > ${dir}server_log.txt 2>&1 &
  check_save server 9
  # 预测
  echo -e "${GREEN_COLOR}cascade_rcnn_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_client.py > ${dir}client_log.txt 2>&1
  check_save client "cascade_rcnn_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function criteo_ctr_gpu_rpc () {
	# 准备
  prepare criteo_ctr gpu_rpc
  # 链接模型数据
  data_dir=${data}criteo_ctr/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}criteo_ctr_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model ctr_serving_model/ --port 9292 --gpu_ids 0 > ${dir}server_log.txt 2>&1 &
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}criteo_ctr_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_client.py ctr_client_conf/serving_client_conf.prototxt raw_data/part-0 > ${dir}client_log.txt 2>&1
  check_save client "criteo_ctr_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function deeplabv3_gpu_rpc () {
	# 准备
	prepare deeplabv3 gpu_rpc
	# 链接模型数据
  data_dir=${data}deeplabv3/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}deeplabv3_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model deeplabv3_server --gpu_ids 0 --port 9494 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  # 清除结果
  rm -f *mask.png *result.png
  echo -e "${GREEN_COLOR}deeplabv3_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version deeplabv3_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  if [ $? != 0 -o ! -e "N0060_jpg_mask.png" -o ! -e "N0060_jpg_result.png" ]; then
    echo -e "${RED_COLOR}$1 error command${RES}\n" | tee -a ${log_dir}server_total.txt ${log_dir}client_total.txt
    error_log deeplabv3_GPU_RPC
  fi
  tail ${dir}client_log.txt | tee -a ${log_dir}client_total.txt
  echo -e "${GREEN_COLOR}deeplabv3_GPU_RPC server test completed${RES}\n" | tee -a ${log_dir}server_total.txt ${log_dir}client_total.txt
  # kill进程
  kill_process
}

function faster_rcnn_hrnetv2p_w18_1x_gpu_rpc () {
	# 准备
	prepare detection/faster_rcnn_hrnetv2p_w18_1x gpu_rpc
	# 链接模型数据
  data_dir=${data}detection/faster_rcnn_hrnetv2p_w18_1x/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}faster_rcnn_hrnetv2p_w18_1x_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model serving_server --port 9494 --gpu_ids 0 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 22
  # 预测
  echo -e "${GREEN_COLOR}faster_rcnn_hrnetv2p_w18_1x_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_client.py 000000570688.jpg > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "faster_rcnn_hrnetv2p_w18_1x_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function faster_rcnn_r50_fpn_1x_coco_gpu_rpc () {
	# 准备
	prepare detection/faster_rcnn_r50_fpn_1x_coco gpu_rpc
	# 链接模型数据
  data_dir=${data}detection/faster_rcnn_r50_fpn_1x_coco/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}faster_rcnn_r50_fpn_1x_coco_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model serving_server --port 9494 --gpu_ids 0 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 6
  # 预测
  echo -e "${GREEN_COLOR}faster_rcnn_r50_fpn_1x_coco_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_client.py 000000570688.jpg > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "faster_rcnn_r50_fpn_1x_coco_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function fcos_dcn_r50_fpn_1x_coco_gpu_rpc () {
	# 准备
	prepare detection/fcos_dcn_r50_fpn_1x_coco gpu_rpc
	# 链接模型数据
  data_dir=${data}detection/fcos_dcn_r50_fpn_1x_coco/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}fcos_dcn_r50_fpn_1x_coco_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model serving_server --port 9494 --gpu_ids 0 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 6
  # 预测
  echo -e "${GREEN_COLOR}fcos_dcn_r50_fpn_1x_coco_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_client.py 000000570688.jpg > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "fcos_dcn_r50_fpn_1x_coco_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function ppyolo_r50vd_dcn_1x_coco_gpu_rpc () {
	# 准备
	prepare detection/ppyolo_r50vd_dcn_1x_coco gpu_rpc
	# 链接模型数据
  data_dir=${data}detection/ppyolo_r50vd_dcn_1x_coco/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}ppyolo_r50vd_dcn_1x_coco_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model serving_server --port 9494 --use_trt --gpu_ids 0 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 60
  # 预测
  echo -e "${GREEN_COLOR}ppyolo_r50vd_dcn_1x_coco_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_client.py 000000570688.jpg > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "ppyolo_r50vd_dcn_1x_coco_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function ssd_vgg16_300_240e_voc_gpu_rpc () {
	# 准备
	prepare detection/ssd_vgg16_300_240e_voc gpu_rpc
	# 链接模型数据
  data_dir=${data}detection/ssd_vgg16_300_240e_voc/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}ssd_vgg16_300_240e_voc_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model serving_server --port 9494 --gpu_ids 0 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}ssd_vgg16_300_240e_voc_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_client.py 000000570688.jpg > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "ssd_vgg16_300_240e_voc_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function ttfnet_darknet53_1x_coco_gpu_rpc () {
	# 准备
	prepare detection/ttfnet_darknet53_1x_coco gpu_rpc
	# 链接模型数据
  data_dir=${data}detection/ttfnet_darknet53_1x_coco/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}ttfnet_darknet53_1x_coco_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model serving_server --port 9494 --gpu_ids 0 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}ttfnet_darknet53_1x_coco_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_client.py 000000570688.jpg > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "ttfnet_darknet53_1x_coco_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function yolov3_darknet53_270e_coco_gpu_rpc () {
	# 准备
	prepare detection/yolov3_darknet53_270e_coco gpu_rpc
	# 链接模型数据
  data_dir=${data}detection/yolov3_darknet53_270e_coco/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}yolov3_darknet53_270e_coco_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model serving_server --port 9494 --use_trt --gpu_ids 0 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 60
  # 预测
  echo -e "${GREEN_COLOR}yolov3_darknet53_270e_coco_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_client.py 000000570688.jpg > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "yolov3_darknet53_270e_coco_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function encryption_gpu_rpc () {
	# 准备
	prepare encryption gpu_rpc
	# 链接模型数据
  data_dir=${data}encryption/
  link_data ${data_dir}
  # 模型加密
  $py_version encrypt.py
	# 启动服务
  echo -e "${GREEN_COLOR}encryption_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model encrypt_server/ --port 9300 --use_encryption_model --gpu_ids 0 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 6
  # 预测
  echo -e "${GREEN_COLOR}encryption_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_client.py uci_housing_client/serving_client_conf.prototxt > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "encryption_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function imagenet_gpu_rpc () {
	# 准备
	prepare imagenet gpu_rpc
	# 链接模型数据
  data_dir=${data}imagenet/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}imagenet_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model ResNet50_vd_model --port 9696 --gpu_ids 0 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}imagenet_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version resnet50_rpc_client.py ResNet50_vd_client_config/serving_client_conf.prototxt > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "imagenet_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function mobilenet_gpu_rpc () {
	# 准备
	prepare mobilenet gpu_rpc
	# 链接模型数据
  data_dir=${data}mobilenet/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}mobilenet_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model mobilenet_v2_imagenet_model --gpu_ids 0 --port 9393 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}mobilenet_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version mobilenet_tutorial.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "mobilenet_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function ocr_gpu_rpc () {
	# 准备
	dir=${log_dir}gpu_rpc/$1/
  enter_dir ocr/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}ocr/
  link_data ${data_dir}
  # 启动服务
  if [ $1 == "ocr" ]
  then
    echo -e "${GREEN_COLOR}$1_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
    $py_version -m paddle_serving_server.serve --model ocr_det_model --port 9293 --gpu_id 0 > ${dir}server_log.txt 2>&1 &
    check_save server 5
    $py_version ocr_web_server.py gpu >> ${dir}server_log.txt 2>&1 &
    sleep 5
  elif [ $1 == "ocr/local" ]
  then
  	echo -e "${GREEN_COLOR}$1_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
    $py_version ocr_debugger_server.py gpu > ${dir}server_log.txt 2>&1 &
    sleep 5
  elif [ $1 == "ocr/only_det" ]
  then
  	echo -e "${GREEN_COLOR}$1_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
    $py_version det_web_server.py gpu > ${dir}server_log.txt 2>&1 &
  elif [ $1 == "ocr/only_det_local" ]
  then
  	echo -e "${GREEN_COLOR}$1_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
    $py_version det_debugger_server.py gpu > ${dir}server_log.txt 2>&1 &
  elif [ $1 == "ocr/only_rec" ]
  then
  	echo -e "${GREEN_COLOR}$1_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
    $py_version rec_web_server.py gpu > ${dir}server_log.txt 2>&1 &
  elif [ $1 == "ocr/only_rec_local" ]
  then
  	echo -e "${GREEN_COLOR}$1_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
    $py_version rec_debugger_server.py gpu > ${dir}server_log.txt 2>&1 &
  fi
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}$1_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  if [ $2 == 1 ]
  then
  	$py_version ocr_web_client.py > ${dir}client_log.txt 2>&1
  elif [ $2 == 2 ]
  then
  	$py_version rec_web_client.py > ${dir}client_log.txt 2>&1
  fi
  # 命令检查
  check_save client "$1_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function resnet_v2_50_gpu_rpc () {
	# 准备
  prepare resnet_v2_50 gpu_rpc
  # 链接模型数据
  data_dir=${data}resnet_v2_50/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}resnet_v2_50_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model resnet_v2_50_imagenet_model --gpu_ids 0 --port 9393 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}resnet_v2_50_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version resnet50_v2_tutorial.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "resnet_v2_50_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function unet_for_image_seg_gpu_rpc () {
	# 准备
  prepare unet_for_image_seg gpu_rpc
  # 链接模型数据
  data_dir=${data}unet_for_image_seg/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}unet_for_image_seg_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model unet_model --gpu_ids 0 --port 9494 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  rm -f *mask.png *result.png
  echo -e "${GREEN_COLOR}unet_for_image_seg_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version seg_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "unet_for_image_seg_GPU_RPC server test completed"
  if [ ! -e "N0060_jpg_mask.png" -o ! -e "N0060_jpg_result.png" ]; then
    echo -e "${RED_COLOR}$1 error command${RES}\n" | tee -a ${log_dir}server_total.txt ${log_dir}client_total.txt
    error_log unet-for-image-seg_GPU_RPC
  fi
  # kill进程
  kill_process
}

function yolov4_gpu_rpc () {
	# 准备
  prepare yolov4 gpu_rpc
  # 链接模型数据
  data_dir=${data}yolov4/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}yolov4_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model yolov4_model --port 9393 --gpu_ids 0 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  # 清除结果
  rm -rf output/
  echo -e "${GREEN_COLOR}yolov4_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_client.py 000000570688.jpg > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "yolov4_GPU_RPC server test completed"
  if [ ! -d "output/" ]; then
    echo -e "${RED_COLOR}client error command${RES}\n" | tee -a ${log_dir}server_total.txt ${log_dir}client_total.txt
    error_log yolov4_GPU_RPC
  fi
  # kill进程
  kill_process
}

function bert_gpu_http () {
	# 准备
  prepare bert gpu_http
  # 链接模型数据
  data_dir=${data}bert/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}bert_GPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version bert_web_service_gpu.py bert_seq128_model/ 9292 > ${dir}server_log.txt 2>&1 &
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}bert_GPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  curl -H "Content-Type:application/json" -X POST -d '{"feed":[{"words": "hello"}], "fetch":["pooled_output"]}' http://127.0.0.1:9292/bert/prediction > ${dir}client_log.txt 2>&1
  check_save client "bert_GPU_HTTP server test completed"
  # kill进程
  kill_process
}

function imagenet_gpu_http () {
	# 准备
	prepare imagenet gpu_http
	# 链接模型数据
  data_dir=${data}imagenet/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}imagenet_GPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version resnet50_web_service.py ResNet50_vd_model gpu 9696 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}imagenet_GPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  curl -H "Content-Type:application/json" -X POST -d '{"feed":[{"image": "https://paddle-serving.bj.bcebos.com/imagenet-example/daisy.jpg"}], "fetch": ["score"]}' http://127.0.0.1:9696/image/prediction > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "imagenet_GPU_HTTP server test completed"
  # kill进程
  kill_process
}

function grpc_impl_example_fit_a_line_cpu_grpc_1 () {
	# 准备
	dir=${log_dir}grpc/fit_a_line/cpu_1/
  enter_dir grpc_impl_example/fit_a_line/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}fit_a_line/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}grpc_impl_example_fit_a_line_CPU_gRPC_1 server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version test_server.py uci_housing_model/ > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}grpc_impl_example_fit_a_line_CPU_gRPC_1 client started${RES}" | tee -a ${log_dir}client_total.txt
  # 同步预测
  echo -e "sync predict:" > ${dir}client_log.txt 2>&1
  $py_version test_sync_client.py >> ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "grpc_impl_example_fit_a_line_CPU_gRPC_1 server sync test completed"
  # 异步预测
  echo -e "asyn predict:" >> ${dir}client_log.txt 2>&1
  $py_version test_asyn_client.py >> ${dir}client_log.txt 2>&1
  check_save client "grpc_impl_example_fit_a_line_CPU_gRPC_1 server asyn test completed"
  # Batch预测
  echo -e "batch predict:" >> ${dir}client_log.txt 2>&1
  $py_version test_batch_client.py >> ${dir}client_log.txt 2>&1
  check_save client "grpc_impl_example_fit_a_line_CPU_gRPC_1 server batch test completed"
  # 预测超时
  echo -e "timeout predict:" >> ${dir}client_log.txt 2>&1
  $py_version test_timeout_client.py >> ${dir}client_log.txt 2>&1
  check_save client "grpc_impl_example_fit_a_line_CPU_gRPC_1 server timeout test completed"
  # kill进程
  kill_process
}

function grpc_impl_example_fit_a_line_cpu_grpc_2 () {
	# 准备
	dir=${log_dir}grpc/fit_a_line/cpu_2/
  enter_dir grpc_impl_example/fit_a_line/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}fit_a_line/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}grpc_impl_example_fit_a_line_CPU_gRPC_2 server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model uci_housing_model --thread 10 --port 9393 --use_multilang > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}grpc_impl_example_fit_a_line_CPU_gRPC_2 client started${RES}" | tee -a ${log_dir}client_total.txt
  # 同步预测
  echo -e "sync predict:" > ${dir}client_log.txt 2>&1
  $py_version test_sync_client.py >> ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "grpc_impl_example_fit_a_line_CPU_gRPC_2 server sync test completed"
  # 异步预测
  echo -e "asyn predict:" >> ${dir}client_log.txt 2>&1
  $py_version test_asyn_client.py >> ${dir}client_log.txt 2>&1
  check_save client "grpc_impl_example_fit_a_line_CPU_gRPC_2 server asyn test completed"
  # Batch预测
  echo -e "batch predict:" >> ${dir}client_log.txt 2>&1
  $py_version test_batch_client.py >> ${dir}client_log.txt 2>&1
  check_save client "grpc_impl_example_fit_a_line_CPU_gRPC_2 server batch test completed"
  # 预测超时
  echo -e "timeout predict:" >> ${dir}client_log.txt 2>&1
  $py_version test_timeout_client.py >> ${dir}client_log.txt 2>&1
  check_save client "grpc_impl_example_fit_a_line_CPU_gRPC_2 server timeout test completed"
  # kill进程
  kill_process
}

function grpc_impl_example_fit_a_line_gpu_grpc_1 () {
	# 准备
	dir=${log_dir}grpc/fit_a_line/gpu_1/
  enter_dir grpc_impl_example/fit_a_line/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}fit_a_line/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}grpc_impl_example_fit_a_line_GPU_gRPC_1 server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version test_server_gpu.py uci_housing_model/ > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}grpc_impl_example_fit_a_line_GPU_gRPC_1 client started${RES}" | tee -a ${log_dir}client_total.txt
  # 同步预测
  echo -e "sync predict:" > ${dir}client_log.txt 2>&1
  $py_version test_sync_client.py >> ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "grpc_impl_example_fit_a_line_GPU_gRPC_1 server sync test completed"
  # 异步预测
  echo -e "asyn predict:" >> ${dir}client_log.txt 2>&1
  $py_version test_asyn_client.py >> ${dir}client_log.txt 2>&1
  check_save client "grpc_impl_example_fit_a_line_GPU_gRPC_1 server asyn test completed"
  # Batch预测
  echo -e "batch predict:" >> ${dir}client_log.txt 2>&1
  $py_version test_batch_client.py >> ${dir}client_log.txt 2>&1
  check_save client "grpc_impl_example_fit_a_line_GPU_gRPC_1 server batch test completed"
  # 预测超时
  echo -e "timeout predict:" >> ${dir}client_log.txt 2>&1
  $py_version test_timeout_client.py >> ${dir}client_log.txt 2>&1
  check_save client "grpc_impl_example_fit_a_line_GPU_gRPC_1 server timeout test completed"
  # kill进程
  kill_process
}

function grpc_impl_example_fit_a_line_gpu_grpc_2 () {
	# 准备
	dir=${log_dir}grpc/fit_a_line/gpu_2/
  enter_dir grpc_impl_example/fit_a_line/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}fit_a_line/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}grpc_impl_example_fit_a_line_GPU_gRPC_2 server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model uci_housing_model --thread 10 --gpu_ids 0 --port 9393 --use_multilang > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}grpc_impl_example_fit_a_line_GPU_gRPC_2 client started${RES}" | tee -a ${log_dir}client_total.txt
  # 同步预测
  echo -e "sync predict:" > ${dir}client_log.txt 2>&1
  $py_version test_sync_client.py >> ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "grpc_impl_example_fit_a_line_GPU_gRPC_2 server sync test completed"
  # 异步预测
  echo -e "asyn predict:" >> ${dir}client_log.txt 2>&1
  $py_version test_asyn_client.py >> ${dir}client_log.txt 2>&1
  check_save client "grpc_impl_example_fit_a_line_GPU_gRPC_2 server asyn test completed"
  # Batch预测
  echo -e "batch predict:" >> ${dir}client_log.txt 2>&1
  $py_version test_batch_client.py >> ${dir}client_log.txt 2>&1
  check_save client "grpc_impl_example_fit_a_line_GPU_gRPC_2 server batch test completed"
  # 预测超时
  echo -e "timeout predict:" >> ${dir}client_log.txt 2>&1
  $py_version test_timeout_client.py >> ${dir}client_log.txt 2>&1
  check_save client "grpc_impl_example_fit_a_line_GPU_gRPC_2 server timeout test completed"
  # kill进程
  kill_process
}

function grpc_impl_example_imdb_cpu_grpc () {
	# 准备
	dir=${log_dir}grpc/imdb/cpu/
  enter_dir grpc_impl_example/imdb/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}imdb/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}grpc_impl_example_imdb_CPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model imdb_cnn_model/ --thread 10 --port 9393 --use_multilang > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}grpc_impl_example_imdb_CPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  head test_data/part-0 | $py_version test_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "grpc_impl_example_imdb_CPU_RPC server test completed"
  # kill进程
  kill_process
}

function grpc_impl_example_yolov4_gpu_grpc () {
	# 准备
	dir=${log_dir}grpc/yolov4/gpu/
  enter_dir grpc_impl_example/yolov4/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}yolov4/
  link_data ${data_dir}
	# 启动服务
  echo -e "${GREEN_COLOR}grpc_impl_example_yolov4_GPU_gRPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model yolov4_model --port 9393 --gpu_ids 0 --use_multilang > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}grpc_impl_example_yolov4_GPU_gRPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_client.py 000000570688.jpg > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "grpc_impl_example_yolov4_GPU_gRPC server test completed"
  # kill进程
  kill_process
}

function pipeline_imagenet_gpu_rpc () {
  # 准备
	dir=${log_dir}pipeline/imagenet/gpu_rpc/
  enter_dir pipeline/imagenet/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}imagenet/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}pipeline_imagenet_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version resnet50_web_service.py > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}pipeline_imagenet_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version pipeline_rpc_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "pipeline_imagenet_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function pipeline_imdb_model_ensemble_cpu_rpc () {
	# 准备
  dir=${log_dir}pipeline/imdb_model_ensemble/cpu_rpc/
  enter_dir pipeline/imdb_model_ensemble/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}imdb/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}pipeline_imdb_model_ensemble_CPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model imdb_cnn_model --port 9292 > ${dir}cnn_log.txt 2>&1 &
  # 单独检查
  if [ $? == 0 ]
  then
  	echo -e "${GREEN_COLOR}pipeline_imdb_model_ensemble_CNN_CPU_RPC execute normally${RES}"
  	sleep 5
    tail ${dir}cnn_log.txt | tee -a ${log_dir}server_total.txt
  fi
  $py_version -m paddle_serving_server.serve --model imdb_bow_model --port 9393 > ${dir}bow_log.txt 2>&1 &
  if [ $? == 0 ]
  then
  	echo -e "${GREEN_COLOR}pipeline_imdb_model_ensemble_BOW_CPU_RPC execute normally${RES}"
  	sleep 5
    tail ${dir}bow_log.txt | tee -a ${log_dir}server_total.txt
  fi
  $py_version test_pipeline_server.py > ${dir}server_log.txt 2>&1 &
  if [ $? == 0 ]
  then
  	echo -e "${GREEN_COLOR}pipeline_imdb_model_ensemble_PIPELINE_SERVER execute normally${RES}"
  	sleep 5
    tail ${dir}bow_log.txt | tee -a ${log_dir}server_total.txt
  fi
  # 预测
  echo -e "${GREEN_COLOR}pipeline_imdb_model_ensemble_CPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_pipeline_client.py > ${dir}client_log.txt 2>&1
  check_save client "pipeline_imdb_model_ensemble_CPU_RPC server test completed"
  # kill进程
  kill_process
}

function pipeline_ocr_gpu_http () {
  # 准备
  dir=${log_dir}pipeline/ocr/gpu_http/
  enter_dir pipeline/ocr/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}ocr/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}pipeline_ocr_GPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version web_service.py > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 9
  # 预测
  echo -e "${GREEN_COLOR}pipeline_ocr_GPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version pipeline_http_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "pipeline_ocr_GPU_HTTP server test completed"
  # kill进程
  kill_process
}

function pipeline_simple_web_service_cpu_http () {
	# 准备
  dir=${log_dir}pipeline/simple_web_service/cpu_http/
  enter_dir pipeline/simple_web_service/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}fit_a_line/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}pipeline_simple_web_service_CPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version web_service.py > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 5
  # 预测
  echo -e "${GREEN_COLOR}pipeline_simple_web_service_CPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  curl -X POST -k http://localhost:18082/uci/prediction -d '{"key": ["x"], "value": ["0.0137, -0.1136, 0.2553, -0.0692, 0.0582, -0.0727, -0.1583, -0.0584, 0.6283, 0.4919, 0.1856, 0.0795, -0.0332"]}' > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "pipeline_simple_web_service_CPU_HTTP server test completed"
  # kill进程
  kill_process
}

function pre_loading() {
  cd ${demo_dir}/bert
  # 链接模型数据
  data_dir=${data}bert/
  link_data ${data_dir}
  if [ $1 == "cpu" ]; then
    $py_version -m paddle_serving_server.serve --model bert_seq128_model/ --port 9292 > ${log_dir}/cpu_pre_loading.txt 2>&1 &
    n=1
    while [ ${n} -le 12 ]; do
      sleep 10
      tail -2 ${log_dir}/cpu_pre_loading.txt
      # 检查日志
      echo "-----check for next step n=${n}-----"
      tail ${log_dir}/cpu_pre_loading.txt | grep -E "ir_graph_to_program_pass" ${log_dir}/cpu_pre_loading.txt > /dev/null
      if [ $? == 0 ]; then
        # 预加载成功
        echo "----------gpu_server loaded----------"
        head ${log_dir}/cpu_pre_loading.txt
        break
      elif [ $n == 12 ]; then
        tail ${log_dir}/cpu_pre_loading.txt | grep -E "failed|Error" ${log_dir}/cpu_pre_loading.txt > /dev/null
        if [ $? == 0 ]; then
          # 下载出错
          exit 1
        fi
        n=10
      fi
      n=$((n + 1))
    done
    tail ${log_dir}/cpu_pre_loading.txt
    echo "----------cpu_server loaded----------"
    kill_process
  else
    $py_version -m paddle_serving_server.serve --model bert_seq128_model/ --port 9292 --gpu_ids 0 > ${log_dir}/gpu_pre_loading.txt 2>&1 &
    n=1
    while [ ${n} -le 50 ]; do
      sleep 10
      tail -2 ${log_dir}/gpu_pre_loading.txt
      # 检查日志
      echo "-----check for next step n=${n}-----"
  #    tail ${log_dir}/gpu_pre_loading.txt | grep -E "Succ load model" ${log_dir}/gpu_pre_loading.txt > /dev/null
      tail ${log_dir}/gpu_pre_loading.txt | grep -E "ir_graph_to_program_pass" ${log_dir}/gpu_pre_loading.txt > /dev/null
      if [ $? == 0 ]; then
        # 预加载成功
        echo "----------gpu_server loaded----------"
        head ${log_dir}/gpu_pre_loading.txt
        break
      elif [ $n == 50 ]; then
        tail ${log_dir}/gpu_pre_loading.txt | grep -E "failed|Error|serving: not found" ${log_dir}/gpu_pre_loading.txt > /dev/null
        if [ $? == 0 ]; then
          # 下载出错
          exit 1
        fi
        n=45
      fi
      n=$((n + 1))
    done
    tail ${log_dir}/gpu_pre_loading.txt
    echo "----------gpu_server loaded----------"
    kill_process
  fi
}

function faster_rcnn_hrnetv2p_w18_1x_encrypt () {
  # 准备
  dir=${log_dir}trt_encrypt/faster_rcnn_hrnetv2p_w18_1x/
  enter_dir detection/faster_rcnn_hrnetv2p_w18_1x
  check_dir ${dir}
	# 链接模型数据
  data_dir=${data}detection/faster_rcnn_hrnetv2p_w18_1x/
  link_data ${data_dir}
  # 模型加密
  $py_version encrypt.py
	# 启动服务
  echo -e "${GREEN_COLOR}faster_rcnn_hrnetv2p_w18_1x_ENCRYPTION_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model encrypt_server/ --port 9494 --gpu_ids 0 --use_encryption_model > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 3
  # 预测
  echo -e "${GREEN_COLOR}faster_rcnn_hrnetv2p_w18_1x_ENCRYPTION_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_encryption.py 000000570688.jpg > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "faster_rcnn_hrnetv2p_w18_1x_ENCRYPTION_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function faster_rcnn_r50_fpn_1x_coco_encrypt () {
  # 准备
  dir=${log_dir}trt_encrypt/faster_rcnn_r50_fpn_1x_coco/
  enter_dir detection/faster_rcnn_r50_fpn_1x_coco
  check_dir ${dir}
	# 链接模型数据
  data_dir=${data}detection/faster_rcnn_r50_fpn_1x_coco/
  link_data ${data_dir}
  # 模型加密
  $py_version encrypt.py
	# 启动服务
  echo -e "${GREEN_COLOR}faster_rcnn_r50_fpn_1x_coco_ENCRYPTION_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model encrypt_server/ --port 9494 --gpu_ids 0 --use_encryption_model > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 3
  # 预测
  echo -e "${GREEN_COLOR}faster_rcnn_r50_fpn_1x_coco_ENCRYPTION_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_encryption.py 000000570688.jpg > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "faster_rcnn_r50_fpn_1x_coco_ENCRYPTION_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function fcos_dcn_r50_fpn_1x_coco_encrypt () {
  # 准备
  dir=${log_dir}trt_encrypt/fcos_dcn_r50_fpn_1x_coco/
  enter_dir detection/fcos_dcn_r50_fpn_1x_coco
  check_dir ${dir}
	# 链接模型数据
  data_dir=${data}detection/fcos_dcn_r50_fpn_1x_coco/
  link_data ${data_dir}
  # 模型加密
  $py_version encrypt.py
	# 启动服务
  echo -e "${GREEN_COLOR}fcos_dcn_r50_fpn_1x_coco_ENCRYPTION_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model encrypt_server/ --port 9494 --gpu_ids 0 --use_encryption_model > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 3
  # 预测
  echo -e "${GREEN_COLOR}fcos_dcn_r50_fpn_1x_coco_ENCRYPTION_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_encryption.py 000000570688.jpg > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "fcos_dcn_r50_fpn_1x_coco_ENCRYPTION_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function ppyolo_r50vd_dcn_1x_coco_encrypt () {
  # 准备
  dir=${log_dir}trt_encrypt/ppyolo_r50vd_dcn_1x_coco/
  enter_dir detection/ppyolo_r50vd_dcn_1x_coco
  check_dir ${dir}
	# 链接模型数据
  data_dir=${data}detection/ppyolo_r50vd_dcn_1x_coco/
  link_data ${data_dir}
  # 模型加密
  $py_version encrypt.py
	# 启动服务
  echo -e "${GREEN_COLOR}ppyolo_r50vd_dcn_1x_coco_ENCRYPTION_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model encrypt_server/ --port 9494 --use_trt --gpu_ids 0 --use_encryption_model > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 3
  # 预测
  echo -e "${GREEN_COLOR}ppyolo_r50vd_dcn_1x_coco_ENCRYPTION_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_encryption.py 000000570688.jpg > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "ppyolo_r50vd_dcn_1x_coco_ENCRYPTION_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function ssd_vgg16_300_240e_voc_encrypt () {
  # 准备
  dir=${log_dir}trt_encrypt/ssd_vgg16_300_240e_voc/
  enter_dir detection/ssd_vgg16_300_240e_voc
  check_dir ${dir}
	# 链接模型数据
  data_dir=${data}detection/ssd_vgg16_300_240e_voc/
  link_data ${data_dir}
  # 模型加密
  $py_version encrypt.py
	# 启动服务
  echo -e "${GREEN_COLOR}ssd_vgg16_300_240e_voc_ENCRYPTION_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model encrypt_server/ --port 9494 --gpu_ids 0 --use_encryption_model > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 3
  # 预测
  echo -e "${GREEN_COLOR}ssd_vgg16_300_240e_voc_ENCRYPTION_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_encryption.py 000000570688.jpg > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "ssd_vgg16_300_240e_voc_ENCRYPTION_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function ttfnet_darknet53_1x_coco_encrypt () {
  # 准备
  dir=${log_dir}trt_encrypt/ttfnet_darknet53_1x_coco/
  enter_dir detection/ttfnet_darknet53_1x_coco
  check_dir ${dir}
	# 链接模型数据
  data_dir=${data}detection/ttfnet_darknet53_1x_coco/
  link_data ${data_dir}
  # 模型加密
  $py_version encrypt.py
	# 启动服务
  echo -e "${GREEN_COLOR}ttfnet_darknet53_1x_coco_ENCRYPTION_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model encrypt_server/ --port 9494 --gpu_ids 0 --use_encryption_model > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 3
  # 预测
  echo -e "${GREEN_COLOR}ttfnet_darknet53_1x_coco_ENCRYPTION_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_encryption.py 000000570688.jpg > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "ttfnet_darknet53_1x_coco_ENCRYPTION_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function yolov3_darknet53_270e_coco_encrypt () {
  # 准备
  dir=${log_dir}trt_encrypt/yolov3_darknet53_270e_coco/
  enter_dir detection/yolov3_darknet53_270e_coco
  check_dir ${dir}
	# 链接模型数据
  data_dir=${data}detection/yolov3_darknet53_270e_coco/
  link_data ${data_dir}
  # 模型加密
  $py_version encrypt.py
	# 启动服务
  echo -e "${GREEN_COLOR}yolov3_darknet53_270e_coco_ENCRYPTION_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model encrypt_server/ --port 9494 --use_trt --gpu_ids 0 --use_encryption_model > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 3
  # 预测
  echo -e "${GREEN_COLOR}yolov3_darknet53_270e_coco_ENCRYPTION_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version test_encryption.py 000000570688.jpg > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "yolov3_darknet53_270e_coco_ENCRYPTION_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function low_precision_resnet50_int8 () {
  # 准备
  dir=${log_dir}low_precision/resnet50/
  enter_dir low_precision/resnet50/
  check_dir ${dir}
  wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ResNet50_quant.tar.gz
  tar zxvf ResNet50_quant.tar.gz
  # 模型转换
  $py_version -m paddle_serving_client.convert --dirname ResNet50_quant
	# 启动服务
  echo -e "${GREEN_COLOR}low_precision_resnet50_int8_GPU_RPC server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version -m paddle_serving_server.serve --model serving_server --port 9393 --gpu_ids 0 --use_trt --precision int8 > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 15
  # 预测
  echo -e "${GREEN_COLOR}low_precision_resnet50_int8_GPU_RPC client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version resnet50_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "low_precision_resnet50_int8_GPU_RPC server test completed"
  # kill进程
  kill_process
}

function pipeline_det_faster_rcnn_gpu_http () {
  # 准备
  dir=${log_dir}pipeline/PaddleDetection/faster_rcnn/
  enter_dir pipeline/PaddleDetection/faster_rcnn/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}detection/faster_rcnn_r50_fpn_1x_coco/
  link_data ${data_dir}
  # 启动服务
  sed -i "s/devices: '2'/devices: '0'/g" config.yml # 修改gpuid
  echo -e "${GREEN_COLOR}pipeline_det_faster_rcnn_GPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version web_service.py > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 9
  # 预测
  echo -e "${GREEN_COLOR}pipeline_det_faster_rcnn_GPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version pipeline_http_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "pipeline_det_faster_rcnn_GPU_HTTP server test completed"
  # kill进程
  kill_process
}

function pipeline_det_ppyolo_mbv3_gpu_http () {
  # 准备
  dir=${log_dir}pipeline/PaddleDetection/ppyolo_mbv3/
  enter_dir pipeline/PaddleDetection/ppyolo_mbv3/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}detection/ppyolo_mbv3/
  link_data ${data_dir}
  # 启动服务
  sed -i "s/devices: '2'/devices: '0'/g" config.yml # 修改gpuid
  echo -e "${GREEN_COLOR}pipeline_det_ppyolo_mbv3_GPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version web_service.py > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 9
  # 预测
  echo -e "${GREEN_COLOR}pipeline_det_ppyolo_mbv3_GPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version pipeline_http_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "pipeline_det_ppyolo_mbv3_GPU_HTTP server test completed"
  # kill进程
  kill_process
}

function pipeline_det_yolov3_gpu_http () {
  # 准备
  dir=${log_dir}pipeline/PaddleDetection/yolov3/
  enter_dir pipeline/PaddleDetection/yolov3/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}detection/yolov3_darknet53_270e_coco/
  link_data ${data_dir}
  # 启动服务
  sed -i "s/devices: '2'/devices: '0'/g" config.yml # 修改gpuid
  echo -e "${GREEN_COLOR}pipeline_det_yolov3_GPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version web_service.py > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 9
  # 预测
  echo -e "${GREEN_COLOR}pipeline_det_yolov3_GPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version pipeline_http_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "pipeline_det_yolov3_GPU_HTTP server test completed"
  # kill进程
  kill_process
}

function pipeline_clas_darknet53_gpu_http () {
  # 准备
  dir=${log_dir}pipeline/PaddleClas/DarkNet53/
  enter_dir pipeline/PaddleClas/DarkNet53/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}pipeline/DarkNet53/
  link_data ${data_dir}
  data_dir=${data}pipeline/images/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}pipeline_clas_darknet53_GPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version resnet50_web_service.py > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 8
  # 预测
  echo -e "${GREEN_COLOR}pipeline_clas_darknet53_GPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version pipeline_rpc_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "pipeline_clas_darknet53_GPU_HTTP server test completed"
  # kill进程
  kill_process
}

function pipeline_clas_hrnet_w18_C_gpu_http () {
  # 准备
  dir=${log_dir}pipeline/PaddleClas/HRNet_W18_C/
  enter_dir pipeline/PaddleClas/HRNet_W18_C/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}pipeline/HRNet_W18_C/
  link_data ${data_dir}
  data_dir=${data}pipeline/images/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}pipeline_clas_HRNet_W18_C_GPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version resnet50_web_service.py > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 8
  # 预测
  echo -e "${GREEN_COLOR}pipeline_clas_HRNet_W18_C_GPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version pipeline_rpc_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "pipeline_clas_HRNet_W18_C_GPU_HTTP server test completed"
  # kill进程
  kill_process
}

function pipeline_clas_MobileNetV1_gpu_http () {
  # 准备
  dir=${log_dir}pipeline/PaddleClas/MobileNetV1/
  enter_dir pipeline/PaddleClas/MobileNetV1/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}pipeline/MobileNetV1/
  link_data ${data_dir}
  data_dir=${data}pipeline/images/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}pipeline_clas_MobileNetV1_GPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version resnet50_web_service.py > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 8
  # 预测
  echo -e "${GREEN_COLOR}pipeline_clas_MobileNetV1_GPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version pipeline_rpc_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "pipeline_clas_MobileNetV1_GPU_HTTP server test completed"
  # kill进程
  kill_process
}

function pipeline_clas_MobileNetV2_gpu_http () {
  # 准备
  dir=${log_dir}pipeline/PaddleClas/MobileNetV2/
  enter_dir pipeline/PaddleClas/MobileNetV2/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}pipeline/MobileNetV2/
  link_data ${data_dir}
  data_dir=${data}pipeline/images/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}pipeline_clas_MobileNetV2_GPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version resnet50_web_service.py > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 8
  # 预测
  echo -e "${GREEN_COLOR}pipeline_clas_MobileNetV2_GPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version pipeline_rpc_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "pipeline_clas_MobileNetV2_GPU_HTTP server test completed"
  # kill进程
  kill_process
}

function pipeline_clas_MobileNetV3_large_x1_0_gpu_http () {
  # 准备
  dir=${log_dir}pipeline/PaddleClas/MobileNetV3_large_x1_0/
  enter_dir pipeline/PaddleClas/MobileNetV3_large_x1_0/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}pipeline/MobileNetV3_large_x1_0/
  link_data ${data_dir}
  data_dir=${data}pipeline/images/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}pipeline_clas_MobileNetV3_large_x1_0_GPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version resnet50_web_service.py > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 8
  # 预测
  echo -e "${GREEN_COLOR}pipeline_clas_MobileNetV3_large_x1_0_GPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version pipeline_rpc_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "pipeline_clas_MobileNetV3_large_x1_0_GPU_HTTP server test completed"
  # kill进程
  kill_process
}

function pipeline_clas_ResNet50_vd_gpu_http () {
  # 准备
  dir=${log_dir}pipeline/PaddleClas/ResNet50_vd/
  enter_dir pipeline/PaddleClas/ResNet50_vd/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}pipeline/ResNet50_vd/
  link_data ${data_dir}
  data_dir=${data}pipeline/images/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}pipeline_clas_ResNet50_vd_GPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version resnet50_web_service.py > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 8
  # 预测
  echo -e "${GREEN_COLOR}pipeline_clas_ResNet50_vd_GPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version pipeline_rpc_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "pipeline_clas_ResNet50_vd_GPU_HTTP server test completed"
  # kill进程
  kill_process
}

function pipeline_clas_ResNet50_vd_FPGM_gpu_http () {
  # 准备
  dir=${log_dir}pipeline/PaddleClas/ResNet50_vd_FPGM/
  enter_dir pipeline/PaddleClas/ResNet50_vd_FPGM/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}pipeline/ResNet50_vd_FPGM/
  link_data ${data_dir}
  data_dir=${data}pipeline/images/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}pipeline_clas_ResNet50_vd_FPGM_GPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version resnet50_web_service.py > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 8
  # 预测
  echo -e "${GREEN_COLOR}pipeline_clas_ResNet50_vd_FPGM_GPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version pipeline_rpc_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "pipeline_clas_ResNet50_vd_FPGM_GPU_HTTP server test completed"
  # kill进程
  kill_process
}

function pipeline_clas_ResNet_V2_50_gpu_http () {
  # 准备
  dir=${log_dir}pipeline/PaddleClas/ResNet_V2_50/
  enter_dir pipeline/PaddleClas/ResNet_V2_50/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}pipeline/ResNet_V2_50/
  link_data ${data_dir}
  data_dir=${data}pipeline/images/
  link_data ${data_dir}
  sed -i "s/127.0.0.1:18000/127.0.0.1:18080/g" pipeline_http_client.py
  # 启动服务
  echo -e "${GREEN_COLOR}pipeline_clas_ResNet_V2_50_GPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version resnet50_web_service.py > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 8
  # 预测
  echo -e "${GREEN_COLOR}pipeline_clas_ResNet_V2_50_GPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version pipeline_http_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "pipeline_clas_ResNet_V2_50_GPU_HTTP server test completed"
  # kill进程
  kill_process
}

function pipeline_clas_ResNet50_vd_KL_gpu_http () {
  # 准备
  dir=${log_dir}pipeline/PaddleClas/ResNet50_vd_KL/
  enter_dir pipeline/PaddleClas/ResNet50_vd_KL/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}pipeline/ResNet50_vd_KL/
  link_data ${data_dir}
  data_dir=${data}pipeline/images/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}pipeline_clas_ResNet50_vd_KL_GPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version resnet50_web_service.py > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 8
  # 预测
  echo -e "${GREEN_COLOR}pipeline_clas_ResNet50_vd_KL_GPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version pipeline_rpc_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "pipeline_clas_ResNet50_vd_KL_GPU_HTTP server test completed"
  # kill进程
  kill_process
}

function pipeline_clas_ResNet50_vd_PACT_gpu_http () {
  # 准备
  dir=${log_dir}pipeline/PaddleClas/ResNet50_vd_PACT/
  enter_dir pipeline/PaddleClas/ResNet50_vd_PACT/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}pipeline/ResNet50_vd_PACT/
  link_data ${data_dir}
  data_dir=${data}pipeline/images/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}pipeline_clas_ResNet50_vd_PACT_GPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version resnet50_web_service.py > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 8
  # 预测
  echo -e "${GREEN_COLOR}pipeline_clas_ResNet50_vd_PACT_GPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version pipeline_rpc_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "pipeline_clas_ResNet50_vd_PACT_GPU_HTTP server test completed"
  # kill进程
  kill_process
}

function pipeline_clas_ResNeXt101_vd_64x4d_gpu_http () {
  # 准备
  dir=${log_dir}pipeline/PaddleClas/ResNeXt101_vd_64x4d/
  enter_dir pipeline/PaddleClas/ResNeXt101_vd_64x4d/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}pipeline/ResNeXt101_vd_64x4d/
  link_data ${data_dir}
  data_dir=${data}pipeline/images/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}pipeline_clas_ResNeXt101_vd_64x4d_GPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version resnet50_web_service.py > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 8
  # 预测
  echo -e "${GREEN_COLOR}pipeline_clas_ResNeXt101_vd_64x4d_GPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version pipeline_rpc_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "pipeline_clas_ResNeXt101_vd_64x4d_GPU_HTTP server test completed"
  # kill进程
  kill_process
}

function pipeline_clas_ShuffleNetV2_x1_0_gpu_http () {
  # 准备
  dir=${log_dir}pipeline/PaddleClas/ShuffleNetV2_x1_0/
  enter_dir pipeline/PaddleClas/ShuffleNetV2_x1_0/
  check_dir ${dir}
  # 链接模型数据
  data_dir=${data}pipeline/ShuffleNetV2_x1_0/
  link_data ${data_dir}
  data_dir=${data}pipeline/images/
  link_data ${data_dir}
  # 启动服务
  echo -e "${GREEN_COLOR}pipeline_clas_ShuffleNetV2_x1_0_GPU_HTTP server started${RES}" | tee -a ${log_dir}server_total.txt
  $py_version resnet50_web_service.py > ${dir}server_log.txt 2>&1 &
  # 命令检查
  check_save server 8
  # 预测
  echo -e "${GREEN_COLOR}pipeline_clas_ShuffleNetV2_x1_0_GPU_HTTP client started${RES}" | tee -a ${log_dir}client_total.txt
  $py_version pipeline_http_client.py > ${dir}client_log.txt 2>&1
  # 命令检查
  check_save client "pipeline_clas_ShuffleNetV2_x1_0_GPU_HTTP server test completed"
  # kill进程
  kill_process
}

function generate_logs() {
  # 获取commit_id
  cd ${CODE_PATH}/Serving/
  commit_id=`git log | head -1`
  commit_id=${commit_id// /_} # 将空格替换为_
  commit_id=${commit_id:0:14} # 截取
  echo "-------commit_id: ${commit_id}"
  export TZ='Asia/Shanghai'
  today=`date +%m%d`
  echo "-------today: ${today}"
  # 生成日志
  cd ${CODE_PATH}
  mkdir -p ../logs_output/daily_logs/${ce_name}
  if [ -d "../logs_output/${today}_${commit_id}/${ce_name}/logs_$1_$2" ]; then
    rm -rf ../logs_output/${today}_${commit_id}/${ce_name}/logs_$1_$2
  fi
  mkdir -p ../logs_output/${today}_${commit_id}/${ce_name}
  # 汇总信息
  rm -rf ../logs_output/daily_logs/${ce_name}/error_models_$1_$2.txt
  cp logs/error_models.txt ../logs_output/daily_logs/${ce_name}/error_models_$1_$2.txt
  # 详细日志
  mv logs ../logs_output/${today}_${commit_id}/${ce_name}/logs_$1_$2
}

# 创建日志目录
check_dir ${log_dir}cpu_rpc/
check_dir ${log_dir}cpu_http/
check_dir ${log_dir}gpu_rpc/
check_dir ${log_dir}gpu_http/
check_dir ${log_dir}grpc/
check_dir ${log_dir}pipeline/
check_dir ${log_dir}trt_encrypt/
check_dir ${log_dir}low_precision/
check_dir ${log_dir}error/
if [ -f "${log_dir}server_total.txt" ]
then
  rm -f ${log_dir}server_total.txt
  rm -f ${log_dir}client_total.txt
fi
touch ${log_dir}server_total.txt
touch ${log_dir}client_total.txt
# 清除错误模型信息
rm -f ${log_dir}error_models.txt
rm -rf ${log_dir}error/*

# 设置py版本
set_py $1
env | grep -E "PYTHONROOT|PYTHON_INCLUDE_DIR|PYTHON_LIBRARIES|PYTHON_EXECUTABLE"

unset http_proxy
unset https_proxy
pre_loading $2
#prepare_bert_model
cpu_rpc=(
bert_cpu_rpc
blazeface_cpu_rpc
criteo_ctr_cpu_rpc
encryption_cpu_rpc
fit_a_line_cpu_rpc
imagenet_cpu_rpc
imdb_cpu_rpc
lac_cpu_rpc
ocr_cpu_rpc
)

cpu_http=(
bert_cpu_http
fit_a_line_cpu_http
imagenet_cpu_http
imdb_cpu_http
lac_cpu_http
senta_cpu_http
)

gpu_rpc=(
bert_gpu_rpc
cascade_rcnn_gpu_rpc
criteo_ctr_gpu_rpc
deeplabv3_gpu_rpc
faster_rcnn_hrnetv2p_w18_1x_gpu_rpc
faster_rcnn_r50_fpn_1x_coco_gpu_rpc
fcos_dcn_r50_fpn_1x_coco_gpu_rpc
ppyolo_r50vd_dcn_1x_coco_gpu_rpc
ssd_vgg16_300_240e_voc_gpu_rpc
ttfnet_darknet53_1x_coco_gpu_rpc
yolov3_darknet53_270e_coco_gpu_rpc
encryption_gpu_rpc
imagenet_gpu_rpc
mobilenet_gpu_rpc
ocr_gpu_rpc
resnet_v2_50_gpu_rpc
unet_for_image_seg_gpu_rpc
yolov4_gpu_rpc
)

gpu_http=(
bert_gpu_http
imagenet_gpu_http
)

cpu_grpc=(
grpc_impl_example_fit_a_line_cpu_grpc_1
grpc_impl_example_fit_a_line_cpu_grpc_2
grpc_impl_example_imdb_cpu_grpc
)
gpu_grpc=(
grpc_impl_example_fit_a_line_gpu_grpc_1
grpc_impl_example_fit_a_line_gpu_grpc_2
grpc_impl_example_yolov4_gpu_grpc
)

# pipeline
pipeline_cpu_rpc=(
pipeline_imdb_model_ensemble_cpu_rpc
)
pipeline_cpu_http=(
# Unicode解码错误
pipeline_simple_web_service_cpu_http
)
pipeline_gpu_rpc=(
pipeline_imagenet_gpu_rpc
)
pipeline_gpu_http=(
pipeline_ocr_gpu_http
pipeline_det_faster_rcnn_gpu_http
pipeline_det_ppyolo_mbv3_gpu_http
pipeline_det_yolov3_gpu_http
pipeline_clas_darknet53_gpu_http
pipeline_clas_hrnet_w18_C_gpu_http
pipeline_clas_MobileNetV1_gpu_http
pipeline_clas_MobileNetV2_gpu_http
pipeline_clas_MobileNetV3_large_x1_0_gpu_http
pipeline_clas_ResNet50_vd_gpu_http
pipeline_clas_ResNet50_vd_FPGM_gpu_http
pipeline_clas_ResNet50_vd_KL_gpu_http
pipeline_clas_ResNet_V2_50_gpu_http
pipeline_clas_ResNet50_vd_PACT_gpu_http
pipeline_clas_ResNeXt101_vd_64x4d_gpu_http
pipeline_clas_ShuffleNetV2_x1_0_gpu_http
)

# trt加密部署
trt_encryption=(
faster_rcnn_hrnetv2p_w18_1x_encrypt
faster_rcnn_r50_fpn_1x_coco_encrypt
fcos_dcn_r50_fpn_1x_coco_encrypt
ppyolo_r50vd_dcn_1x_coco_encrypt
ssd_vgg16_300_240e_voc_encrypt
ttfnet_darknet53_1x_coco_encrypt
yolov3_darknet53_270e_coco_encrypt
)

# 低精度部署
low_precision=(
low_precision_resnet50_int8
)

echo -e "${YELOW_COLOR}-----------------------CPU_RPC TEST START-----------------------${RES}" | tee -a ${log_dir}server_total.txt -a ${log_dir}client_total.txt
for test in ${cpu_rpc[*]}
do
  if [ ${test} == "ocr_cpu_rpc" ]
	then
		ocr_cpu_rpc ocr 1
		ocr_cpu_rpc ocr/local 1
		ocr_cpu_rpc ocr/only_det 1
		ocr_cpu_rpc ocr/only_det_local 1
		ocr_cpu_rpc ocr/only_rec 2
		ocr_cpu_rpc ocr/only_rec_local 2
		continue
	fi
	${test}
done
echo -e "${YELOW_COLOR}-----------------------CPU_RPC TEST COMPLETED-----------------------${RES}\n\n" | tee -a ${log_dir}server_total.txt -a ${log_dir}client_total.txt

echo -e "${YELOW_COLOR}-----------------------CPU_HTTP TEST START-----------------------${RES}" | tee -a ${log_dir}server_total.txt -a ${log_dir}client_total.txt
for test in ${cpu_http[*]}
do
  ${test}
done
echo -e "${YELOW_COLOR}-----------------------CPU_HTTP TEST COMPLETED-----------------------${RES}\n\n" | tee -a ${log_dir}server_total.txt -a ${log_dir}client_total.txt

echo -e "${YELOW_COLOR}-----------------------GPU_RPC TEST START-----------------------${RES}" | tee -a ${log_dir}server_total.txt -a ${log_dir}client_total.txt
if [ $2 != "cpu" ]; then
  for test in ${gpu_rpc[*]}
  do
    if [ ${test} == "ocr_gpu_rpc" ]
    then
      ocr_gpu_rpc ocr 1
      ocr_gpu_rpc ocr/local 1
      ocr_gpu_rpc ocr/only_det 1
      ocr_gpu_rpc ocr/only_det_local 1
      ocr_gpu_rpc ocr/only_rec 2
      ocr_gpu_rpc ocr/only_rec_local 2
      continue
    fi
    ${test}
  done
else
  echo -e "${YELOW_COLOR}----------skip GPU_RPC-----------${RES}"
fi

echo -e "${YELOW_COLOR}-----------------------GPU_RPC TEST COMPLETED-----------------------${RES}\n\n" | tee -a ${log_dir}server_total.txt -a ${log_dir}client_total.txt

echo -e "${YELOW_COLOR}-----------------------GPU_HTTP TEST START-----------------------${RES}" | tee -a ${log_dir}server_total.txt -a ${log_dir}client_total.txt
if [ $2 != "cpu" ]; then
  for test in ${gpu_http[*]}
  do
    ${test}
  done
else
  echo -e "${YELOW_COLOR}----------skip GPU_HTTP-----------${RES}"
fi
echo -e "${YELOW_COLOR}-----------------------GPU_HTTP TEST COMPLETED-----------------------${RES}\n\n" | tee -a ${log_dir}server_total.txt -a ${log_dir}client_total.txt

echo -e "${YELOW_COLOR}-----------------------GRPC TEST START-----------------------${RES}" | tee -a ${log_dir}server_total.txt -a ${log_dir}client_total.txt
for test in ${cpu_grpc[*]}
do
  ${test}
done

if [ $2 != "cpu" ]; then
  for test in ${gpu_grpc[*]}
  do
    ${test}
  done
else
  echo -e "${YELOW_COLOR}----------skip GPU_GRPC-----------${RES}"
fi
echo -e "${YELOW_COLOR}-----------------------GRPC TEST COMPLETED-----------------------${RES}\n\n" | tee -a ${log_dir}server_total.txt -a ${log_dir}client_total.txt

echo -e "${YELOW_COLOR}-----------------------PIPELINE TEST START-----------------------${RES}" | tee -a ${log_dir}server_total.txt -a ${log_dir}client_total.txt
for test in ${pipeline_cpu_rpc[*]}
do
  ${test}
done

for test in ${pipeline_cpu_http[*]}
do
  ${test}
done

if [ $2 != "cpu" ]; then
  for test in ${pipeline_gpu_rpc[*]}
  do
    ${test}
  done

  for test in ${pipeline_gpu_http[*]}
  do
    ${test}
  done
else
  echo -e "${YELOW_COLOR}----------skip GPU_PIPELINE-----------${RES}"
fi
echo -e "${YELOW_COLOR}-----------------------PIPELINE TEST COMPLETED-----------------------${RES}\n\n" | tee -a ${log_dir}server_total.txt -a ${log_dir}client_total.txt

echo -e "${YELOW_COLOR}-----------------------TRT_ENCRYPT TEST START-----------------------${RES}" | tee -a ${log_dir}server_total.txt -a ${log_dir}client_total.txt
if [ $2 != "cpu" ]; then
  for test in ${trt_encryption[*]}
  do
    ${test}
  done
else
  echo -e "${YELOW_COLOR}----------skip TRT_ENCRYPT-----------${RES}"
fi
echo -e "${YELOW_COLOR}-----------------------TRT_ENCRYPT TEST COMPLETED-----------------------${RES}\n\n" | tee -a ${log_dir}server_total.txt -a ${log_dir}client_total.txt

echo -e "${YELOW_COLOR}-----------------------LOW_PRECISION TEST START-----------------------${RES}" | tee -a ${log_dir}server_total.txt -a ${log_dir}client_total.txt
if [ $2 != "cpu" ]; then
  for test in ${low_precision[*]}
  do
    ${test}
  done
else
  echo -e "${YELOW_COLOR}----------skip LOW_PRECISION-----------${RES}"
fi
echo -e "${YELOW_COLOR}-----------------------LOW_PRECISION TEST COMPLETED-----------------------${RES}\n\n" | tee -a ${log_dir}server_total.txt -a ${log_dir}client_total.txt

if [ -f ${log_dir}error_models.txt ]; then
  cat ${log_dir}error_models.txt
fi

generate_logs $1 $2
