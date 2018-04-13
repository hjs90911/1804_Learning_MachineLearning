from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=12, random_state=0)
linkage_array = ward(X)
dendrogram(linkage_array)

ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')

ax.text(bounds[1], 7.25, ' two\nclusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, ' two\nclusters', va='center', fontdict={'size': 15})
plt.xlabel("sample num")
plt.ylabel("cluster distance")

plt.show()