import pandas as pd
import  numpy as np

beer = pd.read_csv('data.txt', sep=' ')
# print(beer)
X = beer[["calories","sodium","alcohol","cost"]]

from sklearn.cluster import DBSCAN
# eps半径，min_samples表示最小的密度，即每一个圈内最小的样本数
# eps的大小和数据的分布有关，如果 做了归一化则eps值偏小一些，如果没有做，则设置的偏大一些
db = DBSCAN(eps=16, min_samples=2).fit(X)
labels = db.labels_
beer['cluster_db'] = labels
beer.sort_values('cluster_db')

colors = np.array(['red', 'green', 'blue', 'yellow'])
beer.groupby('cluster_db').mean()
pd.scatter_matrix(X, c=colors[beer.cluster_db], figsize=(10,10), s=100)

import matplotlib.pyplot as plt
from sklearn import metrics

score = metrics.silhouette_score(X,beer.cluster_db)
print( score)
# 选择聚类的k值
# 绘图 的时候，前面影响后面，需要新建一个画布
scores = []
for k in range(10,20):
    labels = DBSCAN(eps=k).fit(X).labels_
    score = metrics.silhouette_score(X, labels)
    scores.append(score)
plt.figure()
plt.plot(list(range(10,20)), scores)
plt.show()
# plt.plot(list(range(10,20)), scores)
# from pandas.plotting import scatter_matrix
# plt.show()
