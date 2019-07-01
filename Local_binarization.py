import cv2
import numpy as np
import skimage, skimage.morphology
import skimage, skimage.transform
from skimage import transform
from mpmath import im
from pylab import *
import scipy.ndimage as ndi
from skimage import measure,color
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('D:\\picture processing\\samples\\27\\1908259132\\H-3-2.jpg')

# 对图像进行灰度化处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) -> dst
# src：图像
# maxValue：设置的最大值
# adaptiveMethod：指定的阈值计算方法，有cv2.ADAPTIVE_THRESH_MEAN_C：阈值取自相邻区域的平均值；cv2.ADAPTIVE_THRESH_GAUSSIAN_C：阈值取自相邻区域的加权和，权重为一个高斯窗口。
# thresholdType：跟threshold()方法中一样，有cv2.THRESH_BINARY、cv2.THRESH_BINARY_INV等值
# blockSize：邻域大小
# C：一个常数，阈值等于平均值或加权平均值减去该常数
# 函数返回一个二值图像。
th1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,5)

#先把要转换的二值图转换为布尔类型
array=th1.astype(bool)

#去除小面积连通区域
remove = skimage.morphology.remove_small_objects(array, min_size=400, connectivity=1,in_place=False)

#再由布尔值转换回来
remove2=remove.astype(uint8)*255

#创建矩形结构单元
g=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

#膨胀
dilate1=cv2.dilate(remove2,g)

#孔洞填充的代码
# Copy the thresholded image.
im_floodfill = dilate1.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = dilate1.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0, 0), 255);

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
filling = dilate1 | im_floodfill_inv

labels=measure.label(filling,connectivity=2)  #8连通区域标记
# dst=color.label2rgb(labels)  #根据不同的标记显示不同的颜色
print('regions number:',labels.max()+1)  #显示连通区域块数(从0开始标记)
charactors=skimage.measure.regionprops(labels)
print("面积是:", charactors[0].area)

n=1

for i in range(labels.max()):
    aleft=(charactors[i].bbox)[1]
    atop = (charactors[i].bbox)[0]
    aright = (charactors[i].bbox)[3]
    abottom = (charactors[i].bbox)[2]

    cut=img[atop:abottom,aleft:aright]
    # cut1 = np.array(cut)
    cv2.imwrite('d:/picture processing/pic0/'+np.str(n)+'.jpg', cut)
    n=n+1

imgs = np.hstack([gray,filling])
cv2.namedWindow('input_image', cv2.WINDOW_NORMAL)
cv2.imshow('input_image', imgs)


cv2.waitKey(0)
cv2.destroyAllWindows()
