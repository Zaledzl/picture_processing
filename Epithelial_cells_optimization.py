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
img = cv2.imread('D:\\picture processing\\samples\\27\\1908259132\\H-14-2.jpg')
# im = Image.open('D:\\picture processing\\samples\\27\\1908259132\\H-14-2.jpg')

# 对图像进行灰度化处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 先进行高斯滤波降噪。
# GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)
gauss = cv2.GaussianBlur(gray, (3,3), 0)

#Canny(image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None)
# 现在需要确定哪些边界是真正的边界，需要两个阈值，
# minVal和maxVal。图像灰度梯度	高于maxVal被认为是真正的边界，低于minVal的舍弃。
# 两者之间的值要判断是否与真正的边界相连，相连就保留，不相连舍弃。
canny = cv2.Canny(gray, 0, 30)
# canny2 = cv2.Canny(gauss, 0, 30)

#创建矩形结构单元
g=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

#膨胀
dilate1=cv2.dilate(canny,g)


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

#再次膨胀
dilate2=cv2.dilate(filling,g)

#先把要转换的二值图转换为布尔类型
array=dilate2.astype(bool)

#去除小面积连通区域
remove = skimage.morphology.remove_small_objects(array, min_size=20000, connectivity=1,in_place=False)

#再由布尔值转换回来
remove2=remove.astype(uint8)*255

labels=measure.label(remove2,connectivity=2)  #8连通区域标记
# dst=color.label2rgb(labels)  #根据不同的标记显示不同的颜色
print('regions number:',labels.max()+1)  #显示连通区域块数(从0开始标记)
charactors=skimage.measure.regionprops(labels)
print("面积是:", charactors[0].area)

# for i in range(labels.max()):
#     cut=im.crop((charactors[i].bbox))
#     cut1 = np.array(cut)
#     cut2 = cv2.resize(cut1, (1500, 1500))
#     cv2.imwrite('d:/picture processing/pic2/'+np.str(i)+'.jpg', cut2)
n=1

for i in range(labels.max()):
    aleft=(charactors[i].bbox)[1]
    atop = (charactors[i].bbox)[0]
    aright = (charactors[i].bbox)[3]
    abottom = (charactors[i].bbox)[2]

    cut=img[atop:abottom,aleft:aright]
    # cut1 = np.array(cut)
    cut2 = cv2.resize(cut, (1500, 1500))
    cv2.imwrite('d:/picture processing/pic4/'+np.str(n)+'.jpg', cut2)
    n=n+1

# labels2=labels.astype(uint8)

# imgs = np.hstack([im,remove2,labels2])
# cv2.namedWindow('input_image', cv2.WINDOW_NORMAL)
cv2.imshow('input_image', canny)
# cv2.imwrite('lables.bmp',labels2)



cv2.waitKey(0)
cv2.destroyAllWindows()
