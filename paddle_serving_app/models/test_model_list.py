import pytest
from paddle_serving_app.models.model_list import ServingModels
from collections import OrderedDict
import os


class TestServingModels(object):
    def test_get_model_list(self):
        """
        """
        test_dict = OrderedDict()
        test_dict["SentimentAnalysis"] = ["senta_bilstm", "senta_bow", "senta_cnn"]
        test_dict["SemanticRepresentation"] = ["ernie"]
        test_dict["ChineseWordSegmentation"] = ["lac"]
        test_dict["ObjectDetection"] = ["faster_rcnn", "yolov4", "blazeface"]
        test_dict["ImageSegmentation"] = ["unet", "deeplabv3", "deeplabv3+cityscapes"]
        test_dict["ImageClassification"] = ["resnet_v2_50_imagenet", "mobilenet_v2_imagenet"]
        test_dict["TextDetection"] = ["ocr_det"]
        test_dict["OCR"] = ["ocr_rec"]
        sm = ServingModels().get_model_list()
        assert sm == test_dict

    def test_download(self):
        ServingModels().download("ocr_det")
        tar_name = "ocr_det.tar.gz"
        file_list = os.listdir("./")
        assert tar_name in file_list
        os.remove('./ocr_det.tar.gz')


if __name__ == '__main__':
    print(ServingModels().get_model_list())
    print(ServingModels().url_dict)
    TestServingModels().test_download()
    # os.path.abspath()
