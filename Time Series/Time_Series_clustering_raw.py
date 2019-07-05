# -*- coding: utf-8 -*-
# Created by yuanjinshuai at 2019-07-05

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import math

x=np.linspace(0,50,100)
ts1=pd.Series(3.1*np.sin(x/1.5)+3.5)
ts2=pd.Series(2.2*np.sin(x/3.5+2.4)+3.2)
ts3=pd.Series(0.04*x+3.0)

ts1.plot()
ts2.plot()
ts3.plot()

plt.ylim(-2,10)
plt.legend(['ts1','ts2','ts3'])
plt.show()


def DTWDistance(s1, s2):
    # 先构造的边缘
    DTW = {}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return math.sqrt(DTW[len(s1) - 1, len(s2) - 1])


# 生成可以运算的数据
combined = np.vstack((ts1, ts2, ts3, ts1+ts2, ts2+ts3))
data = combined


distance_matrix = np.zeros(0)  # 设定初始距离矩阵

for i in range(len(data)):
    for j in range(len(data)):
        distance = DTWDistance(data[i], data[j])
        distance_matrix = distance


from dtaidistance import clustering
# Custom Hierarchical clustering
model1 = clustering.Hierarchical(distance_matrix, {})
# Augment Hierarchical object to keep track of the full tree
model2 = clustering.HierarchicalTree(model1)
cluster_idx = model2.fit(data)

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

Z = linkage(X, 'ward')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)

Z = linkage(X, 'single')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)
plt.show()