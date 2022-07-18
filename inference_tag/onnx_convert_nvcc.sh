export PYTHON_CMD=python
export BASEPATH=$(cd `dirname $0`; pwd)
export MODELPATH="$BASEPATH/Models"

rm -r result.txt
rm -r *.pdmodel
rm -r *.dot
for dir in $(ls $MODELPATH)
do
  CONVERTPATH=$MODELPATH/$dir
  cd $CONVERTPATH
  rm -r model.onnx
  cd $BASEPATH
done

echo "============ covert and diff check result =============" >> result.txt
for dir in $(ls $MODELPATH)
do
  CONVERTPATH=$MODELPATH/$dir
  echo " Model path: $CONVERTPATH"
  model_file=""
  params_file=""
  for file in $(ls $CONVERTPATH)
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
  if [ "$model_file" = "__model__" ];then
      cd $CONVERTPATH
      echo "$model_file"
      paddle2onnx --model_dir ./  --save_file model.onnx --opset_version 13 --enable_onnx_checker True
      if [[ "$?" != 0 ]]; then
          echo $CONVERTPATH": convert failed(paddle2onnx failed!)" >> result.txt
          continue
      fi
      cd $BASEPATH
  fi
  if [ "${model_file##*.}"x = "pdmodel"x ];then
    cd $CONVERTPATH
    if [ "$params_file" = "" ];then
      paddle2onnx --model_dir ./  --model_filename "$model_file" --save_file model.onnx --opset_version 13 --enable_onnx_checker True
    else
      paddle2onnx --model_dir ./  --model_filename "$model_file" --params_filename "$params_file" --save_file model.onnx --opset_version 13 --enable_onnx_checker True
    fi
    if [[ "$?" != 0 ]]; then
        echo $CONVERTPATH": convert failed(paddle2onnx failed!)" >> result.txt
        continue
    fi
    cd $BASEPATH
    $PYTHON_CMD model_check.py --config_file config.yaml --model_dir "$CONVERTPATH" --paddle_model_file "$model_file" --paddle_params_file "$params_file"
  fi
done
