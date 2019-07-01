import cv2
from matplotlib import pyplot as plt

def image_hist(image): #画三通道图像的直方图
   color = ("blue", "green", "red")#画笔颜色的值可以为大写或小写或只写首字母或大小写混合
   for i, color in enumerate(color):
       hist = cv2.calcHist([image], [i], None, [256], [0, 256])
       plt.plot(hist, color=color)
       plt.xlim([0, 256])
   plt.show()

image = cv2.imread('D:\\picture processing\\pic1\\5.jpg')
cv2.imshow('souce image', image)
image_hist(image)
cv2.waitKey(0)
cv2.destroyAllWindows()
