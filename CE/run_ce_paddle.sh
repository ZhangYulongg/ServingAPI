shell_path=${CODE_PATH}/CE
cd ${shell_path}

# 运行serving补丁脚本
if [ $2 == "pd_cpu" ]; then
    bash -x paddle_docker/run_paddle_cpu.sh
    rm -rf util.py
    mv util_paddle.py util.py
    cuda="cpu"
elif [ $2 == "pd_1027" ]; then
    bash -x paddle_docker/run_paddle_1027.sh
    cuda=1027
elif [ $2 == "pd_112" ]; then
    bash -x paddle_docker/run_paddle_112.sh
    cuda=112
fi

cd ${shell_path}
bash -x pip_install_paddle.sh $1 ${cuda}

# paddle镜像上直接重打到/usr/local #适配cuda11.2镜像
#if [ $2 == 112 ]; then
#    mv /home/TensorRT-8.0.3.4 /usr/local/
#    cp -rf /usr/local/TensorRT-8.0.3.4/include/* /usr/include/ && cp -rf /usr/local/TensorRT-8.0.3.4/lib/* /usr/lib/
#fi
unset http_proxy && unset https_proxy
$py_version -m pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
$py_version download_bin.py > load_bin 2>&1
tail -10 load_bin
ldconfig
bash -x run.sh
