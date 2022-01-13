shell_path=${CODE_PATH}/benchmark
cd ${shell_path}
bash compile.sh $1 $2 $3

unset http_proxy && unset https_proxy
$py_version -m pip install psutil -i https://mirror.baidu.com/pypi/simple
$py_version -m pip install openpyxl -i https://mirror.baidu.com/pypi/simple
$py_version -m pip install pandas -i https://mirror.baidu.com/pypi/simple
tail -10 load_bin
bash run_benchmark.sh $1 $2