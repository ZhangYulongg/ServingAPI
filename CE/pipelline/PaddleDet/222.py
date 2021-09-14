
import numpy as np


aaa = "array([[0.40309864, 0.59690136]], dtype=float32)"
bbb = aaa[7:-17]
print(np.array(eval(bbb), dtype=np.float32))
# bbb = "np." + aaa
# ccc = eval(bbb)
# print(ccc)
print(np.fromstring(aaa, dtype=np.float32))


