#!/bin/bash
set -xe
### 传入参数
device_id=${1:-"0"}
use_cinn=${2:-"False"}
### 指定单卡
export CUDA_VISIBLE_DEVICES=${device_id}
### cinn 相关参数
if [[ ${use_cinn} == "True" ]]; then
  export FLAGS_use_cinn="True"
  export FLAGS_cinn_use_cuda_vectorize="True"
  export FLAGS_cinn_use_new_fusion_pass="True"
  export FLAGS_allow_cinn_ops="batch_norm;batch_norm_grad;conv2d;conv2d_grad;elementwise_add;elementwise_add_grad;relu;relu_grad;sum"
fi
### 启动训练

log_dir=$PWD/ce_log
rm -rf ${log_dir}
mkdir -p ${log_dir}

python3.7 -u ppcls/static/train.py \
            -c ppcls/configs/ImageNet/ResNet/ResNet50.yaml \
            -o use_gpu=True \
            -o print_interval=10 \
            -o is_distributed=False \
            -o Global.epochs=120 \
            -o DataLoader.Train.sampler.batch_size=64 \
            -o DataLoader.Train.dataset.image_root=${Data_path}/ILSVRC2012_w \
            -o DataLoader.Train.dataset.cls_label_path=${Data_path}/ILSVRC2012_w/train_list_20_percent.txt \
            -o DataLoader.Train.loader.num_workers=8 \
            -o Global.save_interval=10000 \
            -o Global.eval_interval=10000 \
            -o Global.eval_during_train=False | tee -a ${log_dir}/ResNet50_bs64.log 2>&1
