import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gc

gc.disable();

def alter(path):
    result = []
    s = os.listdir(path)
    count = 1
    j=0

    img = np.array(
        Image.open('D:\\picture processing\\The positive and negative samples\\impurity1\\1.jpg').convert('L'))
    arr = img.flatten()
    n, bins, patches = plt.hist(arr, bins=256, density=1, facecolor='green', alpha=0.75)
    p = n.reshape(1, 256)

    dirname = os.path.abspath(path)
    filenames = list()
    for root, dirs, files in os.walk(dirname, topdown=False):  # 扫描一层目录
        for name in files:
            filenames.append(root + os.path.sep + name)  # 每一个文件的绝对路径放入列表
            # print(root + os.path.sep + name)
    for i in filenames:

        # 获取文件路径、文件名、后缀名
        (filepath, tempfilename) = os.path.split(filenames[j]);
        (shotname, extension) = os.path.splitext(tempfilename);
        # print(extension)
        if (extension == '.jpg'):
           document = os.path.join(path,i)
           img = cv2.imread(document)
           # img = cv2.resize(img, (500,500))
           # listStr = [str(int(time.time())), str(count)]
           # fileName = 'nihao'.join(listStr)
           # cv2.imwrite(object+os.sep+'%s.jpg' % fileName, img)
           arr = img.flatten()
           n, bins, patches = plt.hist(arr, bins=256, density=1, facecolor='green', alpha=0.75)
           m = n.reshape(1, 256)
           p = np.vstack((p, m))
           count=count+1
           j=j+1
           print("第",count,"个图片处理完成")

    data_df = pd.DataFrame(p)
    writer = pd.ExcelWriter('Excel_White_Blood_Cell.xlsx')
    data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
    writer.save()

    # np.savetxt('new.csv', m, delimiter=',')



alter('D:\\picture processing\\The positive and negative samples\\white_blood_cell2')

gc.enable();