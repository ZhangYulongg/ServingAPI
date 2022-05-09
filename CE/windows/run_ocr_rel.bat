@echo off

goto start

:check_result
if %errorlevel% == 0 (
   findstr 200 ..\logs\%1%_client.log
   if %errorlevel% == 0 (
       echo "----------successfully----------"
       echo "successfully" >> D:\zhangyulong04\688ed254c8576433\test_ocr\success.txt
   ) else (
       echo "----------inference failed-----------"
       echo "inference failed" >> D:\zhangyulong04\688ed254c8576433\test_ocr\failed.txt
   )
) else (
   echo "----------command failed-----------"
   echo "inference failed" >> D:\zhangyulong04\688ed254c8576433\test_ocr\failed.txt
)
goto:eof

:start
::set env
set GNU_HOME=C:\Program Files (x86)\GnuWin32
set Path=D:\zhangyulong04\Python38;%Path%;%GNU_HOME%\bin;
set PATH=d:\v10.2\bin;d:\v10.2\libnvvp;%PATH%
set PATH=D:\zhangyulong04\nvidia\TensorRT-7.0.0.11\lib;%PATH%
d:
cd D:\zhangyulong04\daily_output
rmdir %date:~5,2%%date:~8,2%_%time:~0,2% /q /s
mkdir %date:~5,2%%date:~8,2%_%time:~0,2%

cd D:\zhangyulong04\688ed254c8576433
::copy case file
::/s /e 复制子目录和空目录
xcopy ..\test_ocr\*.* .\test_ocr\ /s /e
::install serving
mkdir whl_packages
mkdir logs
cd whl_packages
wget --no-check-certificate https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_app-0.9.0-py3-none-any.whl >nul 2>nul
wget --no-check-certificate https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_client-0.9.0-cp38-none-any.whl >nul 2>nul
wget --no-check-certificate https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_server_gpu-0.9.0.post1028-py3-none-any.whl >nul 2>nul
curl -O https://paddle-inference-lib.bj.bcebos.com/2.3.0/python/Windows/GPU/x86-64_vs2017_avx_mkl_cuda10.2_cudnn7.6.5_trt7.0.0.11/paddlepaddle_gpu-2.3.0-cp38-cp38-win_amd64.whl >nul 2>nul

python -m pip uninstall paddle_serving_app -y
python -m pip uninstall paddle_serving_client -y
python -m pip uninstall paddle_serving_server_gpu -y
python -m pip uninstall paddlepaddle-gpu -y
::python -m pip uninstall paddlepaddle -y

python -m pip install paddle_serving_app-0.9.0-py3-none-any.whl
python -m pip install paddle_serving_client-0.9.0-cp38-none-any.whl
python -m pip install paddle_serving_server_gpu-0.9.0.post1028-py3-none-any.whl
python -m pip install paddlepaddle_gpu-2.3.0-cp38-cp38-win_amd64.whl
::python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple

cd D:\zhangyulong04\688ed254c8576433\test_ocr
::delete logs
del /f /q /s D:\zhangyulong04\daily_output\daily\*

::start server
start cmd /C "python ocr_debugger_server.py cpu > ..\logs\cpu_server.log 2>&1"
::sleep
@ping 127.0.0.1 -n 15 >nul
::inference
python ocr_web_client.py > ..\logs\cpu_client.log 2>&1
call:check_result cpu
@echo ===========cpu_server.log============
type ..\logs\cpu_server.log
@echo =====================================
@echo ===========cpu_client.log============
type ..\logs\cpu_client.log
@echo =====================================
::kill process
taskkill /f /t /im python.exe

::start server
start cmd /C "python ocr_debugger_server.py gpu > ..\logs\gpu_server.log 2>&1"
::sleep
@ping 127.0.0.1 -n 15 >nul
::inference
python ocr_web_client.py > ..\logs\gpu_client.log 2>&1
call:check_result gpu
@echo ===========gpu_server.log============
type ..\logs\gpu_server.log
@echo =====================================
@echo ===========gpu_client.log============
type ..\logs\gpu_client.log
@echo =====================================
::kill process
taskkill /f /t /im python.exe

xcopy ..\logs\*.* D:\zhangyulong04\daily_output\daily\ /s /e
cd D:\zhangyulong04\daily_output
rmdir %date:~5,2%%date:~8,2%_%time:~0,2% /q /s
mkdir %date:~5,2%%date:~8,2%_%time:~0,2%
xcopy .\daily\*.* D:\zhangyulong04\daily_output\%date:~5,2%%date:~8,2%_%time:~0,2%\ /s /e

::check final result
type D:\zhangyulong04\688ed254c8576433\test_ocr\failed.txt
if %errorlevel% == 0 (
   echo "---------error occur!---------"
   echo "error" > D:\zhangyulong04\daily_output\daily\error.log
   exit 1
) else (
   echo "----------final successfully!-----------"
   echo "successfully" > D:\zhangyulong04\daily_output\daily\success.log
)
