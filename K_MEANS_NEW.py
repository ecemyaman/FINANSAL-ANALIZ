import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
import random

stock1 = pd.read_csv('C:/Users/ECEM YAMAN/Desktop/MATLAB/FİNANSAL ANALİZ/marketing_data.csv')  # dosyamızı okuduk

stock3 = stock1.drop(stock1.columns[[0]], axis=1, inplace=True)

from sklearn.cluster import KMeans
wcss =[]

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)  # 1' den 15'e kadar sırayla k degerleri denenir.
    kmeans.fit(stock1)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,15),wcss)
plt.xlabel("STOK1 K (CLUSTER) VALUE")
plt.ylabel("wcss")
plt.show()

############### 

kmeans2 = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(stock1)
stock1["label"] = clusters
x=stock1.iloc[:, 0:8].values
y= stock1.iloc[:, 0:8].values
plt.scatter(stock1.x[stock1.label ==0], stock1.y[stock1.label ==0],color ="red")
plt.scatter(stock1.x[stock1.label ==1], stock1.y[stock1.label ==1],color ="green")
plt.scatter(stock1.x[stock1.label ==2], stock1.y[stock1.label ==2],color ="blue")
plt.scatter(stock1.x[stock1.label ==3], stock1.y[stock1.label ==3],color ="purple")
plt.scatter(stock1.x[stock1.label ==4], stock1.y[stock1.label ==4],color ="yellow")
plt.scatter(stock1.x[stock1.label ==5], stock1.y[stock1.label ==5],color ="black")
plt.scatter(stock1.x[stock1.label ==6], stock1.y[stock1.label ==6],color ="orange")
plt.scatter(stock1.x[stock1.label ==7], stock1.y[stock1.label ==7],color ="pink")
plt.scatter(stock1.x[stock1.label ==8], stock1.y[stock1.label ==8],color ="cyan")
plt.scatter(stock1.x[stock1.label ==9], stock1.y[stock1.label ==9],color ="brown")
plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1], color = "dark blue")
plt.show()

