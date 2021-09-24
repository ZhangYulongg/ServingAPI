shell_path=${CODE_PATH}/benchmark
cd ${shell_path}
bash compile.sh $1 $2 $3

unset http_proxy && unset https_proxy
$py_version download_bin.py > load_bin 2>&1
tail -10 load_bin
bash run_benchmark.sh $1 $2