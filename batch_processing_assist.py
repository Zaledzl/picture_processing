import os
def SearchAbsPath(dirname):
    dirname = os.path.abspath(dirname)
    filenames = list()
    for root,dirs,files in os.walk(dirname, topdown=False): #扫描一层目录
        for name in files:
            filenames.append(root+os.path.sep+name) #每一个文件的绝对路径放入列表
            print(root+os.path.sep+name)
    return filenames
SearchAbsPath('D:\\picture processing\\two pic file')
