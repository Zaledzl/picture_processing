import os
import time
import cv2

def alter(path,object):
    result = []
    s = os.listdir(path)
    count = 1
    for i in s:
        # 获取文件路径、文件名、后缀名
        (filepath, tempfilename) = os.path.split(i);
        (shotname, extension) = os.path.splitext(tempfilename);
        # print(shotname,extension)
        # 判断这个文件是不是jpg文件，然后处理
        if (extension == '.jpg'):
          document = os.path.join(path,i)
          img = cv2.imread(document)
          img = cv2.resize(img, (500,500))
          listStr = [str(int(time.time())), str(count)]
          fileName = 'nihao'.join(listStr)
          cv2.imwrite(object+os.sep+'%s.jpg' % fileName, img)
          count = count + 1

alter('D:\\picture processing\\samples\\27\\1908259132','d:\\picture processing\\pic6')


# # 打开文件
# path = "d:\\picture processing"
# dirs = os.listdir(path)
#
# # 输出所有文件和文件夹
# for file in dirs:
#     print(file)

# os.path.isfile
# os.path.isdir
# os.walk
# Filelist.append