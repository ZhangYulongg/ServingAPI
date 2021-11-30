shell_path=`pwd`
report_path="${shell_path}/report"
mkdir -p ${report_path}

bash -x pip_install.sh 38 1027

# 暂时适配cuda11.2镜像
#if [ $2 == 112 ]; then
#    mv /home/TensorRT-8.0.3.4 /usr/local/
#    cp -rf /usr/local/TensorRT-8.0.3.4/include/* /usr/include/ && cp -rf /usr/local/TensorRT-8.0.3.4/lib/* /usr/lib/
#fi
unset http_proxy && unset https_proxy
$py_version -m pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
$py_version download_bin.py > load_bin 2>&1
tail -10 load_bin
bash -x run.sh
