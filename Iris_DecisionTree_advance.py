from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets  # 导入方法类
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split #将数据分为测试集和训练集
import numpy as np
import csv


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
# target = target.reshape(1,458)




'''feature_train, feature_test, target_train, target_test 分别代表训练集特征、测试集特征、训练集目标值、验证集特征。
test_size 参数代表划分到测试集数据占全部数据的百分比，你也可以用 train_size 来指定训练集所占全部数据的百分比。
一般情况下，我们会将整个训练集划分为 70% 训练集和 30% 测试集。最后的 random_state 参数表示乱序程度。'''
feature_train, feature_test, target_train, target_test = train_test_split(feature,target, test_size=0.33, random_state=42)

#可以直接通过 print iris_target 查看一下花的分类数据。这里，scikit-learn 已经将花的原名称进行了转换，
# 其中 0, 1, 2 分别代表 Iris Setosa, Iris Versicolour 和 Iris Virginica。
#print(target_train)

'''DecisionTreeClassifier() 模型方法中也包含非常多的参数值。例如：
criterion = gini/entropy 可以用来选择用基尼指数或者熵来做损失函数。
splitter = best/random 用来确定每个节点的分裂策略。支持“最佳”或者“随机”。
max_depth = int 用来控制决策树的最大深度，防止模型出现过拟合。
min_samples_leaf = int 用来设置叶节点上的最少样本数量，用于对树进行修剪。'''

dt_model = DecisionTreeClassifier() # 所有参数均置为默认状态
dt_model.fit(feature_train,target_train) # 使用训练集训练模型
predict_results = dt_model.predict(feature_test) # 使用模型对测试集进行预测

print(accuracy_score(predict_results, target_test))

#其实，在 scikit-learn 中的分类决策树模型就带有 score 方法，只是传入的参数和 accuracy_score() 不太一致。
scores = dt_model.score(feature_test, target_test)
print(scores)

'''两种准确度方法输入参数的区别。一般情况下，模型预测的准确度会和多方面因素相关。
首先是数据集质量，本实验中，我们使用的数据集非常规范，几乎不包含噪声，所以预测准确度非常高。
其次，模型的参数也会对预测结果的准确度造成影响。'''