# -*- coding: ms949 -*-
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import mglearn

X, y = make_blobs(centers=2,            # centers: y���� ������ ����
                  random_state=5,       # random_state: ���� �õ尪 (�������� ������ ������ ������ �������� ����)
                  n_samples=30)         # n_samples: ���� ����
print(X.shape, y.shape)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["class1","class2"], loc=4)
plt.xlabel("feature1")
plt.ylabel("feature2")
plt.show()

# Training�� Testing ��������
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)                       # �����ϴ� ����
clf.fit(X_train, y_train)                                       # �н��ϴ� ��ɾ�

print("test prediction : {}".format(clf.predict(X_test)))       # ���� ����� ���
print("test label : {}".format(y_test))
print("test accuracy : {}".format(clf.score(X_test, y_test)))

