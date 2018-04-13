# -*- coding: ms949 -*-
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pylab as plt
from sklearn.cluster import KMeans

X, Y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

kmeans = KMeans(n_clusters=4)   # n_clusters: 군집의 개수
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
print(X)
print(y_kmeans)

plt.scatter(X[:,0], X[:,1], c=y_kmeans, s=50)
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1],
            c='red', s=200, alpha=0.5)  # center값 설정; c: 색, s: 사이즈, alpha: 투명도
plt.show()