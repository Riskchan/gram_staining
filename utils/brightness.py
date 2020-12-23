import os
import sys
import glob
import cv2
import statistics
import numpy as np
from matplotlib import pyplot as plt

path = sys.argv[1:][0]
path = os.path.join(path, "*/aerobic/*.JPG")
files = glob.glob(path)

for filepath in files:
    img = cv2.imread(filepath)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

#    mode = statistics.mode(v.ravel())
#    print(mode)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    basename = os.path.splitext(os.path.basename(filepath))[0]
    ax.hist(v.ravel(),256,[0,256])
    fig.savefig(basename + "_hist.png")


"""

plt.hist(v.ravel(),256,[0,256])
plt.show()

#ヒストグラム平坦化
#result = cv2.equalizeHist(v)

#適応的ヒストグラム平坦化
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
#result = clahe.apply(v)

#輝度値の平均と標準偏差を指定
#v = (v-np.mean(v)) / np.std(v) * 80 + 100
#result = np.array(v, dtype=np.uint8)
result = v - 35
plt.hist(result.ravel(),256,[0,256]);plt.show()

hsv = cv2.merge((h,s,result))
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite("result.jpg", rgb)
"""