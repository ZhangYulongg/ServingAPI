#!/bin/bash
export http_proxy=${proxy}
export https_proxy=${proxy}

#wget https://paddle-ci.gz.bcebos.com/TRT/TensorRT6-cuda10.2-cudnn7.tar.gz --no-check-certificate
#tar -zxf TensorRT6-cuda10.2-cudnn7.tar.gz -C /usr/local
#cp -rf /usr/local/TensorRT-6.0.1.8/include/*  /usr/include/ && cp -rf /usr/local/TensorRT-6.0.1.8/lib/* /usr/lib/
#rm -rf TensorRT6-cuda10.2-cudnn7.tar.gz

wget https://paddle-ci.gz.bcebos.com/TRT/TensorRT7-cuda10.2-cudnn8.tar.gz --no-check-certificate
tar -zxf TensorRT7-cuda10.2-cudnn8.tar.gz -C /usr/local
cp -rf /usr/local/TensorRT-7.1.3.4/include/*  /usr/include/ && cp -rf /usr/local/TensorRT-7.1.3.4/lib/* /usr/lib/
rm -rf TensorRT7-cuda10.2-cudnn8.tar.gz

apt install -y libcurl4-openssl-dev libbz2-dev
wget https://paddle-serving.bj.bcebos.com/others/centos_ssl.tar && tar xf centos_ssl.tar && rm -rf centos_ssl.tar && mv libcrypto.so.1.0.2k /usr/lib/libcrypto.so.1.0.2k && mv libssl.so.1.0.2k /usr/lib/libssl.so.1.0.2k && ln -sf /usr/lib/libcrypto.so.1.0.2k /usr/lib/libcrypto.so.10 && ln -sf /usr/lib/libssl.so.1.0.2k /usr/lib/libssl.so.10 && ln -sf /usr/lib/libcrypto.so.10 /usr/lib/libcrypto.so && ln -sf /usr/lib/libssl.so.10 /usr/lib/libssl.so

export PYTHON_INCLUDE_DIR=/usr/local/python3.7.0/include/python3.7m/
export PYTHON_LIBRARIES=/usr/local/python3.7.0/lib/libpython3.7m.so
export PYTHON_EXECUTABLE=/usr/local/python3.7.0/bin/python3.7
