from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets  # 导入方法类
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split #将数据分为测试集和训练集
import numpy as np
import csv
from sklearn import datasets#引入数据集,sklearn包含众多数据集
from sklearn.model_selection import train_test_split#将数据分为测试集和训练集
from sklearn.neighbors import KNeighborsClassifier#利用邻近点方式训练数据


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

target = birth_data[:,256]
feature = birth_data[:,0:255]

#利用train_test_split进行将训练集和测试集进行分开，test_size占30%
# X_train,X_test,y_train,y_test=train_test_split(feature,target,test_size=0.3)
feature_train, feature_test, target_train, target_test = train_test_split(feature,target, test_size=0.33, random_state=42)
# print(y_train)#我们看到训练数据的特征值分为3类


###训练数据###
knn=KNeighborsClassifier()#引入训练方法
knn.fit(feature_train,target_train)#进行填充测试数据进行训练
predict_results = knn.predict(feature_test) # 使用模型对测试集进行预测
# ###预测数据###
# print(knn.predict(X_test))#预测特征值
print(accuracy_score(predict_results, target_test))