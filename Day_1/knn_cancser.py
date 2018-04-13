# -*- coding: ms949 -*-
from sklearn.datasets import load_breast_cancer
import numpy as np

cancer = load_breast_cancer()
print(cancer.keys())
print(cancer.data.shape)    # (��, ��)�� ��� 569���� �����Ͱ� ����ִ�
print(cancer.target)        # 0�� 1�θ� ���
print(cancer.feature_names)

print("count per class: \n{}".format(
      {n: v for n,
       v in zip(cancer.target_names,            # zip�� �����ִ� ����
                np.bincount(cancer.target))}))  # ������ �����ִ� �Լ�

# Training�� Testing�� ����
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    stratify=cancer.target, # �����͸� ������
                                                    random_state=66)


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train, y_train)       # �н� ����

print(clf.score(X_test, y_test))

# Draw graph
import matplotlib.pylab as plt
training_accuracy = []
test_accuracy = []
# 1 ~ 10���� n_neighbors�� ����
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))
    
plt.plot(neighbors_settings, training_accuracy, label="train accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()