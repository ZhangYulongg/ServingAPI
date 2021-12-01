shell_path=${CODE_PATH}/continuous_evaluation/src/task
cd ${shell_path}
bash -x pip_install.sh ${py_flag} ${cuda_version}

unset http_proxy && unset https_proxy
$py_version -m pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
$py_version download_bin.py > load_bin 2>&1
tail -10 load_bin
bash -x run.sh
