# -*- coding: utf-8 -*-
# Created by yuanjinshuai at 2019-07-04

import math
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def generate_data(nT,nC,mG,A,sg,eg):
    timeSeries = pd.DataFrame()
    basicSeries = pd.DataFrame()
    β = 0.5*np.pi
    ω = 2*np.pi/nT
    t = np.linspace(0,nT,nT)
    for ic,c in enumerate(np.arange(nC)):
        slope = sg*(-(nC-1)/2 + c)
        s = A * (-1**c -np.exp(t*eg))*np.sin(t*ω*(c+1) + c*β) + t*ω*slope
        basicSeries[ic] = s
        sr = np.outer(np.ones_like(mG),s)
        sr = sr + 1*np.random.rand(mG,nT) + 1.0*np.random.randn(mG,1)
        timeSeries = timeSeries.append(pd.DataFrame(sr))
    return basicSeries, timeSeries

def plot_basicSeries(basicSeries):
    with plt.style.context('seaborn'):      # 'fivethirtyeight'
         fig = plt.figure(figsize=(20,8)) ;
         ax1 = fig.add_subplot(111);
         plt.title('Basice patterns to generate Longitudinal data',fontsize=25, fontweight='bold')
         plt.xlabel('Time', fontsize=15, fontweight='bold')
         plt.ylabel('Signal of the observed feature', fontsize=15, fontweight='bold')
         plt.plot(basicSeries, lw=10, alpha = 1.8)

def plot_timeSeries(timeSeries):
    with plt.style.context('seaborn'):      # 'fivethirtyeight'
         fig = plt.figure(figsize=(20,8)) ;
         ax1 = fig.add_subplot(111);
         plt.title('Longitudinal data',fontsize=25, fontweight='bold')
         plt.xlabel('Time', fontsize=15, fontweight='bold')
         plt.ylabel('Signal of the observed feature', fontsize=15, fontweight='bold')
         plt.plot(timeSeries.T)
         #ax1 = sns.tsplot(ax=ax1, data=timeSeries.values, ci=[68, 95])

def plot_dendogram(Z):
    with plt.style.context('fivethirtyeight' ):
         plt.figure(figsize=(15, 5))
         plt.title('Dendrogram of time series clustering',fontsize=25, fontweight='bold')
         plt.xlabel('sample index', fontsize=25, fontweight='bold')
         plt.ylabel('distance', fontsize=25, fontweight='bold')
         hac.dendrogram( Z, leaf_rotation=90.,    # rotates the x axis labels
                            leaf_font_size=15., ) # font size for the x axis labels
        #plt.show()

def plot_results(timeSeries, D, cut_off_level):
    result = pd.Series(hac.fcluster(D, cut_off_level, criterion='maxclust'))
    clusters = result.unique()
    figX = 20; figY = 15
    fig = plt.subplots(figsize=(figX, figY))
    mimg = math.ceil(cut_off_level/2.0)
    gs = gridspec.GridSpec(mimg,2, width_ratios=[1,1])
    for ipic, c in enumerate(clusters):
        cluster_index = result[result==c].index
        print(ipic, "Cluster number %d has %d elements" % (c, len(cluster_index)))
        ax1 = plt.subplot(gs[ipic])
        ax1.plot(timeSeries.T.iloc[:,cluster_index])
        ax1.set_title(('Cluster number '+str(c)), fontsize=15, fontweight='bold')
    #plt.show()

def plot_basic_cluster(X):
    with plt.style.context('fivethirtyeight' ):
         plt.figure(figsize=(17,3))
         D1 = hac.linkage(X, method='ward', metric='euclidean')
         dn1= hac.dendrogram(D1)
         plt.title("Clustering: method='ward', metric='euclidean'")

         plt.figure(figsize=(17, 3))
         D2 = hac.linkage(X, method='single', metric='euclidean')
         dn2= hac.dendrogram(D2)
         plt.title("Clustering: method='single', metric='euclidean'")
         plt.show()

#---- number of time series
nT = 101  # number of observational point in a time series
nC = 6    # number of charakteristic  signal groups
mG = 10   # number of time series in a charakteristic signal group

#---- control parameters for data generation
Am = 0.3; # amplitude of the signal
sg = 0.3  # rel. weight of the slope
eg = 0.02 # rel. weight of the damping

#---- generate the data
basicSeries,timeSeries = generate_data(nT,nC,mG,Am,sg,eg)
plot_basicSeries(basicSeries)
plot_timeSeries(timeSeries)

#--- Here we use spearman correlation as distance metric
def myMetric(x, y):
    r = stats.pearsonr(x, y)[0]
    return 1 - r


#--- run the clustering
D = hac.linkage(timeSeries, method='single', metric=myMetric)
plot_dendogram(D)

#---- evaluate the dendogram
cut_off_level = 6   # level where to cut off the dendogram
plot_results(timeSeries, D, cut_off_level)