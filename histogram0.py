import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def histogram_demo(image):
    plt.hist(image.ravel(), 256, [0, 256])#ravel函数功能是将多维数组降为一维数组
    plt.show()

image = cv.imread('D:\\picture processing\\pic1\\5.jpg')
# image1 = cv.imread('D:\\picture processing\\pic1\\7.jpg')

# cv.imshow('souce image', image)
histogram_demo(image)
# histogram_demo(image1)
cv.waitKey(0)
cv.destroyAllWindows()
