# -*- coding: ms949 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from matplotlib.colors import ListedColormap

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# ����� 0, �л��� 1�� �ǵ��� �������� �������� ����
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

fig, axes = plt.subplots(1, 4, figsize=(15, 3),
                         subplot_kw={'xticks': (), 'yticks': ()})

# ����� �˰��� ���� ����Ʈ�� ����
algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2),
              DBSCAN()]

# �񱳸� ���� �������� Ŭ������ �Ҵ�
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))

# ������ �Ҵ��� Ŭ������
cm3 = ListedColormap(['#0000aa', '#ff2020', '#50ff50'])
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters,
                cmap=cm3, s=60, edgecolors='black')
axes[0].set_title("random - ARI: {:.2f}".format(
        adjusted_rand_score(y, random_clusters)))

for ax, algorithm in zip(axes[1:], algorithms):
    # Ŭ������ �Ҵ�� Ŭ������ �߽� ǥ��
    clusters = algorithm.fit_predict(X_scaled)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters,
               cmap=cm3, s=60, edgecolors='black')
    ax.set_title("{} - ARI: {:.2f}".format(algorithm.__class__.__name__,
                                           adjusted_rand_score(y, clusters)))
plt.show()