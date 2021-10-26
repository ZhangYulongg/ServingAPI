#!/bin/bash

# 进入serving目录
serving_dir=${CODE_PATH}/Serving/
# 日志目录
log_dir=${CODE_PATH}/logs/
# 输出颜色
RED_COLOR='\E[1;31m'  #红
GREEN_COLOR='\E[1;32m' #绿
YELOW_COLOR='\E[1;33m' #黄
RES='\E[0m'
# 异常词
error_words="Fail|DENIED|None|empty|failed|not match"
cuda_version=$2

function set_proxy() {
  echo -e "${YELOW_COLOR}---------set proxy---------${RES}"
  export https_proxy=${proxy}
  export http_proxy=${proxy}
}

function unset_proxy() {
  echo -e "${YELOW_COLOR}---------unset proxy---------${RES}"
  unset http_proxy
  unset https_proxy
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
      tail ${dir}server_log.txt | tee -a ${log_dir}server_total.txt
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
        echo "$2 failed" >> ${log_dir}/result.txt
      else
        # 不含关键词，正常预测
        echo -e "${GREEN_COLOR}$2${RES}\n" | tee -a ${log_dir}server_total.txt ${log_dir}client_total.txt
      fi
    fi
  else
    echo -e "${RED_COLOR}$1 error command${RES}\n" | tee -a ${log_dir}server_total.txt ${log_dir}client_total.txt
    cat ${dir}client_log.txt | tee -a ${log_dir}client_total.txt
    # 记录模型、部署方式、环境信息
    echo "$2 failed" >> ${log_dir}/result.txt
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
  if [ -d "../logs_output/${today}_${commit_id}/${ce_name}/logs_java_$1_$2" ]; then
    rm -rf ../logs_output/${today}_${commit_id}/${ce_name}/logs_java_$1_$2
  fi
  mkdir -p ../logs_output/${today}_${commit_id}/${ce_name}
  # 汇总信息
  rm -rf ../logs_output/daily_logs/${ce_name}/error_models_java_$1_$2.txt
  cp logs/error_models.txt ../logs_output/daily_logs/${ce_name}/error_models_java_$1_$2.txt
  # 详细日志
  mv logs ../logs_output/${today}_${commit_id}/${ce_name}/logs_java_$1_$2
}

function java_sync_client() {
    dir=${log_dir}java/sync_client/
    check_dir ${dir}
    cd /Serving/java/examples/target
    java -cp paddle-serving-sdk-java-examples-0.0.1-jar-with-dependencies.jar PipelineClientExample string_imdb_predict > ${dir}client_log.txt 2>&1
    check_save client "pipeline_imdb_model_ensemble_sync_java_client_CPU_PIPELINE"
}

function java_asyn_client() {
    dir=${log_dir}java/asyn_client/
    check_dir ${dir}
    cd /Serving/java/examples/target
    java -cp paddle-serving-sdk-java-examples-0.0.1-jar-with-dependencies.jar PipelineClientExample asyn_predict > ${dir}client_log.txt 2>&1
    check_save client "pipeline_imdb_model_ensemble_asyn_java_client_CPU_PIPELINE"
}

function java_indarray_sync_client() {
    dir=${log_dir}java/indarray_sync_client/
    check_dir ${dir}
    cd /Serving/java/examples/target
    java -cp paddle-serving-sdk-java-examples-0.0.1-jar-with-dependencies.jar PipelineClientExample indarray_predict > ${dir}client_log.txt 2>&1
    check_save client "pipeline_simple_web_service_indarray_sync_java_client_CPU_PIPELINE"
}

function java_http_proto_client() {
    dir=${log_dir}java/http_proto/
    check_dir ${dir}
    cd /Serving/java/examples/target
    # 拷贝模型配置文件
    cp -r /mnt/serving/dataset/data/fit_a_line/uci_housing_client/serving_client_conf.prototxt  ./fit_a_line.prototxt
    java -cp paddle-serving-sdk-java-examples-0.0.1-jar-with-dependencies.jar PaddleServingClientExample http_proto fit_a_line.prototxt > ${dir}client_log.txt 2>&1
    check_save client "fit_a_line_java_http_proto_GPU_BRPC"
}

function java_http_json_client() {
    dir=${log_dir}java/http_json/
    check_dir ${dir}
    cd /Serving/java/examples/target
    java -cp paddle-serving-sdk-java-examples-0.0.1-jar-with-dependencies.jar PaddleServingClientExample http_json fit_a_line.prototxt > ${dir}client_log.txt 2>&1
    check_save client "fit_a_line_java_http_json_GPU_BRPC"
}

function java_grpc_client() {
    dir=${log_dir}java/grpc/
    check_dir ${dir}
    cd /Serving/java/examples/target
    java -cp paddle-serving-sdk-java-examples-0.0.1-jar-with-dependencies.jar PaddleServingClientExample grpc fit_a_line.prototxt > ${dir}client_log.txt 2>&1
    check_save client "fit_a_line_java_grpc_GPU_BRPC"
}

function java_compress_client() {
    dir=${log_dir}java/grpc/
    check_dir ${dir}
    cd /Serving/java/examples/target
    java -cp paddle-serving-sdk-java-examples-0.0.1-jar-with-dependencies.jar PaddleServingClientExample compress fit_a_line.prototxt > ${dir}client_log.txt 2>&1
    check_save client "fit_a_line_java_compress_GPU_BRPC"
}

# 安装java client 依赖
echo "---------------install java client begin---------------"
cd /Serving
set_proxy
git checkout ${test_branch}
git pull
git branch
git log | head -10
cd /Serving/java
# 修改示例设置端口
sed -i "s/9393/9898/g" examples/src/main/java/PaddleServingClientExample.java
mvn compile
mvn install
cd examples
mvn compile
mvn install

unset_proxy
rm -rf ${log_dir}/result.txt
java_sync_client
java_asyn_client
java_indarray_sync_client
java_http_proto_client
java_http_json_client
java_grpc_client
java_compress_client

if [ -f ${log_dir}result.txt ]; then
  cat ${log_dir}result.txt
  exit 1
fi

