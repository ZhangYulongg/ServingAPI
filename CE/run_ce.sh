shell_path=${CODE_PATH}/CE
cd ${shell_path}
bash pip_install.sh $1 $2
python3.6 download_bin.py
bash run.sh
