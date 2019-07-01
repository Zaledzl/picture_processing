import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df = pd.read_csv('D:\\picture processing\\training_data\\Excel_Mould2.csv')
# df = pd.read_csv('D:\\picture processing\\training_data\\All_csv_clean.csv')
#
# # 查看前5条数据
# print(df.head(5))


birth_data = []
with open('D:\\picture processing\\training_data\\All_csv_clean.csv') as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    birth_header = next(csv_reader)  # 读取第一行每一列的标题
    for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
        birth_data.append(row)


birth_data = [[float(x) for x in row] for row in birth_data]  # 将数据从string形式转换为float形式

#取第1列所有数据

birth_data = np.array(birth_data)  # 将list数组转化成array数组便于查看数据结构
birth_header = np.array(birth_header)

f = birth_data[:,256]
g = birth_data[:,0:255]

f=f.reshape(1,458)

print(birth_data.shape)  # 利用.shape查看结构。
print(f.shape)
print(g.shape)
print(f)
# print(birth_data)
