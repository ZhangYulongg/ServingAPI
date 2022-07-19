export FLAGS_cudnn_exhaustive_search=1
export FLAGS_allocator_strategy=auto_growth

backend_type_list=(onnxruntime paddle)
enable_trt_list=(true false)
enable_gpu_list=(true)
enable_mkldnn_list=(true false)
gpu_id=0
batch_size_list=(1 8 16)
precision_list=(fp32 fp16)
subgraph_size=3
config_file=config.yaml
export BASEPATH=$(cd `dirname $0`; pwd)
export MODELPATH="$BASEPATH/Models"

run_benchmark(){
  for backend_type in ${backend_type_list[@]};do
    if [ "$backend_type" = "onnxruntime" ] || [ "$backend_type" = "openvino" ] ;then
      if [ ! -f "$model_dir/model.onnx" ];then
         echo "cont find ONNX model file. "
        continue
      fi
    fi
    model_file=""
    params_file=""
    for file in $(ls $model_dir)
      do
        if [ "${file##*.}"x = "pdmodel"x ];then
          model_file=$file
          echo "find model file: $model_file"
        fi

        if [ "${file##*.}"x = "pdiparams"x ];then
          params_file=$file
          echo "find param file: $params_file"
        fi
    done

    if [ "$dir_name" = "fairmot_dla34_30e_1088x608" -o "$dir_name" = "mask_rcnn_r50_fpn_1x_coco" ]; then
      batch_size_var=(1)
    else
      batch_size_var=${batch_size_list[@]}
    fi

    if [ "$dir_name" = "mask_rcnn_r50_fpn_1x_coco" ]; then
      subgraph_size_var=8
    else
      subgraph_size_var=${subgraph_size}
    fi

    if [ "$dir_name" = "picodet_l_640_coco_lcnet" -o "$dir_name" = "SwinTransformer_tiny_patch4_window7_224" ]; then
      continue
    fi

    for batch_size in ${batch_size_var[@]};do
      for enable_gpu in ${enable_gpu_list[@]};do
        if [ ${enable_gpu} = "true" ]; then
          for enable_trt in ${enable_trt_list[@]};do
            if [ ${enable_trt} = "true" ]; then
                for precision in ${precision_list[@]};do
                    # tune
                    python benchmark.py --model_dir=${model_dir} --config_file ${config_file} --precision ${precision} --enable_gpu=${enable_gpu} --gpu_id=${gpu_id} --enable_trt=${enable_trt} --backend_type=${backend_type} --batch_size=${batch_size} --paddle_model_file "$model_file" --paddle_params_file "$params_file" --enable_tune=true --return_result=true

                    python benchmark.py --model_dir=${model_dir} --config_file ${config_file} --precision ${precision} --enable_gpu=${enable_gpu} --gpu_id=${gpu_id} --enable_trt=${enable_trt} --backend_type=${backend_type} --batch_size=${batch_size} --subgraph_size=${subgraph_size_var} --paddle_model_file "$model_file" --paddle_params_file "$params_file"
                done
            elif [ ${enable_trt} = "false" ]; then
                python benchmark.py --model_dir=${model_dir} --config_file ${config_file} --precision fp32 --enable_gpu=${enable_gpu} --gpu_id=${gpu_id} --enable_trt=${enable_trt} --backend_type=${backend_type} --batch_size=${batch_size} --paddle_model_file "$model_file" --paddle_params_file "$params_file"
            fi
          done
        elif [ ${enable_gpu} = "false" ]; then
          for enable_mkldnn in ${enable_mkldnn_list[@]};do
            python benchmark.py --model_dir=${model_dir} --config_file ${config_file} --enable_mkldnn=${enable_mkldnn} --enable_gpu=false --gpu_id=0 --enable_trt=false --backend_type=${backend_type} --batch_size=${batch_size} --paddle_model_file "$model_file" --paddle_params_file "$params_file"
          done
        fi
      done
    done
  done
}

echo "============ Benchmark result =============" >> result.txt

for dir in $(ls $MODELPATH);do
  CONVERTPATH=$MODELPATH/$dir
  echo " >>>> Model path: $CONVERTPATH"
  export model_dir=$CONVERTPATH
  export dir_name=$dir
  run_benchmark
done
