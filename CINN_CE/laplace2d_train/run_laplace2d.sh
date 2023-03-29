export CUDA_VISIBLE_DEVICES=0
export NVIDIA_TF32_OVERRIDE=0
export PYTHONPATH=$PWD:$PYTHONPATH
export FLAGS_use_cinn=1
export FLAGS_allow_cinn_ops="fill_any_like;fill_constant_p;broadcast_p;add_p;sub_p;div_p;mul_p;sqrt_p;tanh_p;matmul_p;reduce_p;concat_p;reshape_p;transpose_p;slice_select_p;slice_assign_p;split_p;gather_p;scatter_add_p;reduce_sum_p"
#export FLAGS_cinn_use_new_fusion_pass=1
export FLAGS_enable_pe_launch_cinn=0
#export FLAGS_cinn_use_fill_constant_folding=1

log_dir=$PWD/ce_log
rm -rf ${log_dir}
mkdir -p ${log_dir}

sed -e "s/epochs: 20000/epochs: 10000/" \
    -e "s/num_layers: 5/num_layers: 5/" \
    -e "s/hidden_size: 20/hidden_size: 20/" \
    -e "s/npoints: 512/npoints: 512/" tests/config/laplace2d.yaml > test_laplace2d_1w_config1.yaml

python3.7 examples/laplace/laplace2d.py -c test_laplace2d_1w_config1.yaml | tee -a ${log_dir}/laplace2d_config1.log 2>&1
#sed -i "s/num_epoch = 2010/num_epoch = 10010/" examples/laplace2d/laplace2d_static_new_ad.py
#python3.7 examples/laplace2d/laplace2d_static_new_ad.py | tee -a ${log_dir}/laplace2d_10000_epoch.log 2>&1
