import numpy as np
import base64
import cv2

from paddle_serving_app.reader import Sequential, Resize, CenterCrop, RGB2BGR, Transpose, Div, Normalize
from paddle_serving_server.web_service import WebService, Op


class ImagenetOp(Op):
    def init_op(self):
        self.seq = Sequential([
            Resize(256), CenterCrop(224), RGB2BGR(), Transpose((2, 0, 1)),
            Div(255), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                                True)
        ])
        self.label_dict = {}
        label_idx = 0
        with open("imagenet.label") as fin:
            for line in fin:
                self.label_dict[label_idx] = line.strip()
                label_idx += 1

    def preprocess(self, input_dicts, data_id, log_id):
        (_, input_dict), = input_dicts.items()
        batch_size = len(input_dict.keys())
        imgs = []
        for key in input_dict.keys():
            data = base64.b64decode(input_dict[key].encode('utf8'))
            data = np.fromstring(data, np.uint8)
            im = cv2.imdecode(data, cv2.IMREAD_COLOR)
            img = self.seq(im)
            imgs.append(img[np.newaxis, :].copy())
        input_imgs = np.concatenate(imgs, axis=0)
        return {"image": input_imgs}, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, log_id):
        score_list = fetch_dict["score"]
        result = {"label": [], "prob": []}
        for score in score_list:
            score = score.tolist()
            max_score = max(score)
            result["label"].append(self.label_dict[score.index(max_score)]
                                   .strip().replace(",", ""))
            result["prob"].append(max_score)
        result["label"] = str(result["label"])
        result["prob"] = str(result["prob"])
        return result, None, ""


class ImageService(WebService):
    def get_pipeline_response(self, read_op):
        image_op = ImagenetOp(name="imagenet", input_ops=[read_op])
        return image_op
