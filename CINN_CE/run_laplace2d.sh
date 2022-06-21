export CUDA_VISIBLE_DEVICES=0
export NVIDIA_TF32_OVERRIDE=0
export PYTHONPATH=$PWD:$PYTHONPATH
export FLAGS_use_cinn=1
export FLAGS_allow_cinn_ops="fill_any_like;fill_constant_p;broadcast_p;add_p;sub_p;div_p;mul_p;sqrt_p;tanh_p;matmul_p;reduce_p;concat_p;reshape_p;transpose_p;slice_select_p;slice_assign_p;split_p;gather_p;scatter_add_p"
export FLAGS_cinn_use_new_fusion_pass=1
export FLAGS_enable_pe_launch_cinn=0
export FLAGS_cinn_use_fill_constant_folding=1

python3.7 examples/laplace2d/laplace2d_static_new_ad.py
