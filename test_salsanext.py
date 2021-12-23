import sys
import numpy as np
sys.path.append("SalsaNext/")
sys.path.append("SalsaNext/train")
sys.path.append("SalsaNext/salsanext_16")
from infer_test import load
import time


model_dir = "SalsaNext/salsanext_16"
filename = "SalsaNext/0000012.bin"
scan = np.fromfile(filename, dtype=np.float32)
scan = scan.reshape((-1, 4))


user_runtime = load(model_dir)
preds = user_runtime.infer(scan)




print("Preds: ", preds)

print("##"*10)

t0 = time.time()
preds = user_runtime.infer(scan)
print("Inference time: ", time.time() - t0)