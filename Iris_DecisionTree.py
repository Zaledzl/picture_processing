'''数据集名称的准确名称为 Iris Data Set，总共包含 150 行数据。每一行数据由 4 个特征值及一个目标值组成。
其中 4 个特征值分别为：萼片长度、萼片宽度、花瓣长度、花瓣宽度。而目标值及为三种不同类别的鸢尾花，
分别为：Iris Setosa，Iris Versicolour，Iris Virginica'''

from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets  # 导入方法类
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split #将数据分为测试集和训练集
import numpy as np

iris = datasets.load_iris()  # 加载 iris 数据集
iris_feature = iris.data  # 特征数据
iris_target = iris.target  # 分类数据

'''feature_train, feature_test, target_train, target_test 分别代表训练集特征、测试集特征、训练集目标值、验证集特征。
test_size 参数代表划分到测试集数据占全部数据的百分比，你也可以用 train_size 来指定训练集所占全部数据的百分比。
一般情况下，我们会将整个训练集划分为 70% 训练集和 30% 测试集。最后的 random_state 参数表示乱序程度。'''
feature_train, feature_test, target_train, target_test = train_test_split(iris_feature, iris_target, test_size=0.33, random_state=42)

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