import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# 读入数据  这个数据集的列 是有名称的
df = pd.read_csv('D:\\picture processing\\Iris data set\\iris.csv')
# "Sepal.Length" "Sepal.Width" "Petal.Length" "Petal.Width" "Species"
# '''
# 数据时以逗号为分隔符的，
# 但是这个数据没有列的名字，
# 所以先给每个列取个名字，
# 直接使用数据说明中的描述
# '''
# df.columns = ['sepal_len', 'sepal_width', 'petal_len', 'petal_width', 'class']

# 查看前5条数据
print(df.head(5))

def scatter_plot_by_category(feat, x, y):
    alpha = 0.5
    gs = df.groupby(feat)
    cs = cm.rainbow(np.linspace(0, 1, len(gs)))
    for g, c in zip(gs, cs):
        plt.scatter(g[1][x], g[1][y], color=c, alpha=alpha)

plt.figure(figsize=(20,5))

plt.subplot(131)
scatter_plot_by_category('Species', 'Sepal.Length', 'Petal.Length')
plt.xlabel('sepal_len')
plt.ylabel('petal_len')
plt.title('class')

