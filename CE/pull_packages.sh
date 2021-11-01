whl_list=(
app-0.0.0-py3
client-0.0.0-cp36
#client-0.0.0-cp37
#client-0.0.0-cp38
#server-0.0.0-py3
#server_gpu-0.0.0.post101-py3
#server_gpu-0.0.0.post1027-py3
#server_gpu-0.0.0.post1028-py3
server_gpu-0.0.0.post112-py3
)

for whl_item in ${whl_list[@]}
do
    wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_${whl_item}-none-any.whl
    if [ $? -eq 0 ]; then
        echo "--------------download ${whl_item} succ"
    else
        echo "--------------download ${whl_item} failed"
    fi
done