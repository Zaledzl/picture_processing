import cv2
import numpy as np

# 读取图像
img = cv2.imread('D:\\picture processing\\samples\\27\\1908259132\\H-14-2.jpg')

# 对图像进行灰度化处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 去噪声
# blurred = cv2.GaussianBlur(gray, (9, 9),0)

# ret, dst = cv2.threshold(src, thresh, maxval, type)
# src： 输入图，只能输入单通道图像，通常来说为灰度图
# dst： 输出图
# thresh： 阈值
# maxval： 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
# type：二值化操作的类型，包含以下5种类型： cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV； cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV

# 对图像进行二值化处理

ret,binary = cv2.threshold(gray,118,255,cv2.THRESH_BINARY)

binary=255-binary

#WINDOW_NORMAL设置了这个值，用户便可以改变窗口的大小（没有限制)
#WINDOW_AUTOSIZE如果设置了这个值，窗口大小会自动调整以适应所显示的图像，并且不能手动改变窗口大小.
# 展示图像
#cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
imgs = np.hstack([gray, binary])
cv2.namedWindow('input_image', cv2.WINDOW_NORMAL)
cv2.imshow('input_image', imgs)


# 参数为毫秒 如果参数为0  则表示程序会无限制的等待用户的按键事件
cv2.waitKey(0)
cv2.destroyAllWindows()