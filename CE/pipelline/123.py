import numpy as np
import copy

from paddle_serving_app.reader import RCNNPostprocess


postprocess = RCNNPostprocess("label_list.txt", "output")
result = np.load("fetch_dict.npy", allow_pickle=True).item()
print(result)

# draw bboxes
dict_ = copy.deepcopy(result)
dict_["image"] = "000000570688.jpg"
postprocess(dict_)
