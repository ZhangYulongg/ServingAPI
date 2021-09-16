shell_path=${CODE_PATH}/CE
cd ${shell_path}
bash pip_install.sh $1 $2

unset http_proxy && unset https_proxy
$py_version -m pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
$py_version download_bin.py > load_bin 2>&1
tail -10 load_bin
bash run.sh
