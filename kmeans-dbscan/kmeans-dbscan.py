# beer dataset
import pandas as pd
beer = pd.read_csv('data.txt', sep=' ')
# print(beer)
X = beer[["calories","sodium","alcohol","cost"]]
# 引入keams聚类
from sklearn.cluster import KMeans
# n_clusters表示分类的数量
km = KMeans(n_clusters=3).fit(X)
km2 = KMeans(n_clusters=2).fit(X)
# km.labels_查看聚类的结果
# 按照cluster排序
beer['cluster'] = km.labels_
beer['cluster2'] = km2.labels_
beer.sort_values('cluster')

from pandas.plotting import scatter_matrix
# %matplotlib inline
cluster_centers = km.cluster_centers_
cluster_centers_2 = km2.cluster_centers_
# 将聚类结果分类求均值
beer.groupby("cluster").mean()
beer.groupby("cluster2").mean()
# 中心点坐标
centers = beer.groupby("cluster").mean().reset_index()

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
import numpy as np
colors = np.array(['red', 'green', 'blue', 'yellow'])
# 绘图
# 绘制分布点
plt.scatter(beer["calories"], beer["alcohol"],c=colors[beer["cluster"]])
# 绘制中心点
plt.scatter(centers.calories, centers.alcohol, linewidths=3, marker='+', s=300, c='black')
plt.xlabel("Calories")
plt.ylabel("Alcohol")
# 两两维度查看结果
scatter_matrix(beer[["calories","sodium","alcohol","cost"]],s=100, alpha=1, c=colors[beer["cluster"]], figsize=(10,10))
plt.suptitle("With 3 centroids initialized")

scatter_matrix(beer[["calories","sodium","alcohol","cost"]],s=100, alpha=1, c=colors[beer["cluster2"]], figsize=(10,10))
plt.suptitle("With 2 centroids initialized")
# plt.show()


# 数据的标准化StandardScaler
from sklearn.preprocessing import StandardScaler
# 实例化
scaler = StandardScaler()
# 标准化变换
X_scaled = scaler.fit_transform(X)
# 继续用keam进行分类预测
km = KMeans(n_clusters=3).fit(X_scaled)
beer["scaled_cluster"] = km.labels_
beer.sort_values("scaled_cluster")

beer.groupby("scaled_cluster").mean()

pd.scatter_matrix(X, c=colors[beer.scaled_cluster], alpha=1, figsize=(10,10), s=100)
# 聚类评估：轮廓系数（Silhouette Coefficient
# 计算样本i到同簇其他样本的平均距离ai。ai 越小，说明样本i越应该被聚类到该簇。将ai 称为样本i的簇内不相似度。
# 计算样本i到其他某簇Cj 的所有样本的平均距离bij，称为样本i与簇Cj 的不相似度。定义为样本i的簇间不相似度：bi =min{bi1, bi2, ..., bik}
# si = (bi-ai)/max(bi,ai)
# si接近1，则说明样本i聚类合理
# si接近-1，则说明样本i更应该分类到另外的簇
# 若si 近似为0，则说明样本i在两个簇的边界上。
# 引入评估函数
from sklearn import metrics
score_scaled = metrics.silhouette_score(X,beer.scaled_cluster)
score = metrics.silhouette_score(X,beer.cluster)
print(score_scaled, score)

# 选择聚类的k值
scores = []
for k in range(2,20):
    labels = KMeans(n_clusters=k).fit(X).labels_
    score = metrics.silhouette_score(X, labels)
    scores.append(score)

# print(scores)
plt.plot(list(range(2,20)), scores)
# plt.xlabel("Number of Clusters Initialized")
# plt.ylabel("Sihouette Score")
plt.show()
