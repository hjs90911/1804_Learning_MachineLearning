from sklearn.neighbors import DistanceMetric

dist = DistanceMetric.get_metric('euclidean')
X = [[0,1,2], [3,4,5]]
d = dist.pairwise(X)
# print(d)

X2 = [[1,2,3,4,5,6,7,8,9,10],
      [11,12,13,14,15,16,17,18,19,20],
      [61,62,63,64,65,66,66,66,66,69]]
d2 = dist.pairwise(X2)
print(d2)