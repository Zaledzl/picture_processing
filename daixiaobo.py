import cv2
import numpy as np
import os
pic_no = 0
#加载人脸检测分类器,路径找寻一下自己的目录中的.xml文件，
# face=cv2.CascadeClassifier(r'C:\ProgramData\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')

face=cv2.CascadeClassifier(r'D:\haarcascade_frontalface_alt2.xml')
cap=cv2.VideoCapture(0)#opencv视频流捕获
cap.set(3,480)#设置捕获图像每一帧的宽度
cap.set(4,480)#设置捕获图像每一帧的高度
while True:
    ret,frame=cap.read()#捕获一帧图片，返回两个值，第一个是返回ret(true/false)，frame---图片矩阵
    frame=cv2.flip(frame,1)##图像翻转，-1垂直水平翻转，0垂直翻转，1水平翻转
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#图片的BGR转换成灰度图(opencv读取的图片的色彩通道默认是BGR)
    #人脸检测函数
        #第一个参数是灰度图
        #第二个参数是scale(1.0-1.5最佳)
        #第三个参数5最佳
        #返回图片中每一个人脸由左上角参数
        #宽高四个参数组成的一个元组，一个人脸一个元组
    faces=face.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:#遍历每一张人脸
        cropped=frame[y:y+h,x:x+w]#扣取人脸部分，并存入一个新的矩阵
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2,cv2.LINE_AA)#画框
        pic_no = pic_no + 1
        # img = np.array(cropped)
        cv2.imwrite('D:\projects\dzl_face',str(pic_no)+'.jpg',cropped)#把扣取的矩阵以.jpg的格式存入当前目录
    cv2.imshow('frame',frame)
    if(cv2.waitKey(1) & 0XFF==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()