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
img = cv2.imread('D:\\PF3\\Photos&Documents3\\ducuments3\\samples\\27.1\\1908259135\\H-6-2.jpg')
im = Image.open('D:\\PF3\\Photos&Documents3\\ducuments3\\samples\\27.1\\1908259135\\H-6-2.jpg')

# 对图像进行灰度化处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 去噪声


# 对图像进行二值化处理
ret,im_th = cv2.threshold(gray,118,255,cv2.THRESH_BINARY)

im_th=255-im_th


#孔洞填充的代码
# Copy the thresholded image.
im_floodfill = im_th.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0, 0), 255);

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
im_out = im_th | im_floodfill_inv



#创建矩形结构单元
g=cv2.getStructuringElement(cv2.MORPH_RECT,(5,7))

#膨胀
im_out_dilate=cv2.dilate(im_th,g)
im_out_dilate1=cv2.dilate(im_out_dilate,g)
im_out_dilate2=cv2.dilate(im_out_dilate1,g)


print(array)
array=im_out_dilate2.astype(bool)



# ar: 上边的获取的标记好连通域的数组
# connectivity: 邻接模式，1表示4邻接，2表示8邻接
# in_place: bool型值，如果为True,表示直接在输入图像中删除小块区域，否则进行复制后再删除。默认为False.
filling = skimage.morphology.remove_small_objects(array, min_size=5000, connectivity=1,in_place=False)


filling=filling.astype(uint8)*255


#再次进行孔洞填充
im_floodfill = filling.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = filling.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0, 0), 255);

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
filling_out = filling | im_floodfill_inv


labels=measure.label(filling_out,connectivity=2)  #8连通区域标记
# dst=color.label2rgb(labels)  #根据不同的标记显示不同的颜色
print('regions number:',labels.max()+1)  #显示连通区域块数(从0开始标记)
charactors=skimage.measure.regionprops(labels)
# print("面积是:", charactors[0].area)
# label_att = measure.regionprops(labels)
# print("这是属性",label_att)
print("bbox", charactors[0].bbox)

for i in range(labels.max()):
    cut=im.crop((charactors[i].bbox))
    cut1 = np.array(cut)
    cut2 = cv2.resize(cut1, (1500, 1500))
    cv2.imwrite('D:\projects\pic/'+np.str(i)+'.jpg', cut2)

# # cut = img.crop(charactors[0].bbox)
# cut = im.crop((charactors[4].bbox))
# cut1 = np.array(cut)
# # cv2的保存图片的方法
# cv2.imwrite('cut.jpg', cut1)

# imgs = np.hstack([filling ,filling_out])
# cv2.namedWindow('input_image', cv2.WINDOW_NORMAL)
# cv2.imshow('input_image', filling_out)


cv2.waitKey(0)
cv2.destroyAllWindows()