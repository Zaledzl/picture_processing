import os
import time
import cv2

def alter(path,object):
    result = []
    s = os.listdir(path)
    count = 1

# os.walk(top[, topdown=True[, onerror=None[, followlinks=False]]])
# top - - 是你所要遍历的目录的地址.topdown - - 可选,为True, 则优先遍历top目录,
# 否则优先遍历top的子目录(默认为开启)。onerror - - 可选，需要一个callable对象, 当
# walk需要异常时，会调用。followlinks - - 可选, 如果为True，则会遍历目录下的快捷方式，默认开启return None
# 该函数没有返回值会使用yield关键字抛出一个存放当前该层目录(root, dirs, files)的三元组，最终将所有目录层的的结果变为一个生成器root
# 所指的是当前正在遍历的这个文件夹的本身的地址dirs是一个list ，内容是该文件夹中所有的目录的名字(不包括子目录)
# files同样是list, 内容是该文件夹中所有的文件(不包括子目录)

    dirname = os.path.abspath(path)
    filenames = list()
    for root, dirs, files in os.walk(dirname, topdown=False):  # 扫描一层目录
        for name in files:
            filenames.append(root + os.path.sep + name)  # 每一个文件的绝对路径放入列表
            # print(root + os.path.sep + name)
    for i in filenames:
        j=0
        # 获取文件路径、文件名、后缀名
        (filepath, tempfilename) = os.path.split(filenames[j]);
        (shotname, extension) = os.path.splitext(tempfilename);
        # print(extension)
        if (extension == '.jpg'):
           document = os.path.join(path,i)
           #如果这个文件不是JPEG，读取就会出问题，应该在这里加一个判断
           img = cv2.imread(document)
           # img = cv2.resize(img, (500,500))
           listStr = [str(int(time.time())), str(count)]
           fileName = 'nihao'.join(listStr)
           cv2.imwrite(object+os.sep+'%s.jpg' % fileName, img)
           count = count + 1
           j=j+1


alter('D:\\picture processing\\samples\\27','d:\\picture processing\\pic1')