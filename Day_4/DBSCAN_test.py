# -*- encoding: ms949 -*-
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=30, random_state=0)
dbscan = DBSCAN(eps=0.1, min_samples=2)
clusters = dbscan.fit_predict(X)
print("cluster Label:\n{}".format(clusters))    # -1은 군집에 속하지 못했다는 것을 의미

print(dbscan.eps)
print(dbscan.min_samples)