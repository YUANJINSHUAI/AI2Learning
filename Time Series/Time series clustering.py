# -*- coding: utf-8 -*-
# Created by yuanjinshuai at 2019-07-04


'''
while len(data) > 1:
    print("☞ 第 {} 次迭代\n".format(10 - len(data) + 1))
    min_distance = float('inf')  # 设定初始距离为无穷大
    for i in range(len(data)):
        print("---")
        for j in range(i + 1, len(data)):
            distance = DTWDistance(data[i], data[j])
            print("计算 {} 与 {} 距离为 {}".format(data[i], data[j], distance))
            if distance < min_distance:
                min_distance = distance
                min_ij = (i, j)

    i, j = min_ij  # 最近数据点序号
    data1 = data[i]
    data2 = data[j]
    data = np.delete(data, j, 0)  # 删除原数据
    data = np.delete(data, i, 0)  # 删除原数据

    b = np.atleast_2d([(data1[0] + data2[0]) / 2, (data1[1] + data2[1]) / 2])  # 计算两点新中心
    data = np.concatenate((data, b), axis=0)  # 将新数据点添加到迭代过程
    print("\n最近距离:{} & {} = {}, 合并后中心:{}\n".format(data1, data2, min_distance, b))

'''


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import math
import scipy.cluster.hierarchy
import seaborn as sns

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


def dtw_distance(s1, s2):
    # 先构造的边缘
    dtw = {}

    for i in range(len(s1)):
        dtw[(i, -1)] = float('inf')
    for i in range(len(s2)):
        dtw[(-1, i)] = float('inf')
    dtw[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            dtw[(i, j)] = dist + min(dtw[(i - 1, j)], dtw[(i, j - 1)], dtw[(i - 1, j - 1)])

    return math.sqrt(dtw[len(s1) - 1, len(s2) - 1])


# 生成可以运算的数据
combined = np.vstack((ts1, ts2, ts3, ts1+ts2, ts2+ts3))
data = combined
print(len(data))

distance_matrix = np.zeros([len(data), len(data)])  # 设定初始距离矩阵
min_distance = float('inf')  # 设定初始距离为无穷大

# 找到最小的距离的两簇
for i in range(len(data)):
    for j in range(len(data)):
        distance = dtw_distance(data[i], data[j])
        distance_matrix[i, j] = distance
        print("计算 {}曲线 与 {}曲线 距离为 {}".format(i, j, distance))
        if distance < min_distance and i != j:
            min_distance = distance
            min_ij = (i, j)
print(min_ij)

# compute linkage matrix for HAC clusters
links = scipy.cluster.hierarchy.linkage(distance_matrix, method='ward')
sns.clustermap(distance_matrix, xticklabels = False, yticklabels=False,
               method='ward', col_linkage=links, row_linkage=links)
