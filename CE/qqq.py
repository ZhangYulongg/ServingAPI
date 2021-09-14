import os
import subprocess
import numpy as np
import copy
import cv2
import requests
import json

from paddle_serving_client import Client, HttpClient
from paddle_serving_app.reader import OCRReader
from paddle_serving_app.reader import Sequential, ResizeByFactor
from paddle_serving_app.reader import Div, Normalize, Transpose, File2Image
from paddle_serving_app.reader import DBPostProcess, FilterBoxes, GetRotateCropImage, SortedBoxes
import paddle.inference as paddle_infer

from util import *

seq = Sequential([
            File2Image(), ResizeByFactor(32, 960), Div(255),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), Transpose(
                (2, 0, 1))
        ])
det_client = Client()
det_client.load_client_config("ocr_det_model")
det_client.connect(["127.0.0.1:9293"])


filename = "imgs/1.jpg"
im = seq(filename)

det_out = det_client.predict(feed={"image": im}, fetch=["concat_1.tmp_0"], batch=False)
print(det_out)

