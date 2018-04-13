# -*- coding: ms949 -*-
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

svc = SVC()
svc.fit(X_train, y_train)
print("train accuracy: {:.2f}".format(svc.score(X_train, y_train)))
print("test accuracy: {:.2f}".format(svc.score(X_test, y_test)))

# import matplotlib.pylab as plt
# plt.boxplot(X_train, manage_xticks=False)
# plt.yscale("symlog")
# plt.xlabel("feature list")
# plt.ylabel("feature size")
# plt.show()

min_on_training = X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)

# 훈련 데이터에 최솟값을 빼고 범위로 나누면 각 특성에 대해 최솟값은 0, 최댓값은 1
X_train_scaled = (X_train - min_on_training) / range_on_training
X_test_scaled = (X_test - min_on_training) / range_on_training
print("min:\n{}".format(X_train_scaled.min(axis=0)))
print("max:\n{}".format(X_train_scaled.max(axis=0)))

svc = SVC()
svc.fit(X_train_scaled, y_train)
print("train accuracy: {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("test accuracy: {:.3f}".format(svc.score(X_test_scaled, y_test)))

print()

print("C=1000일 때 score")
svc = SVC(C=1000)     # C값이 증가하면 규제가 작아짐 (반비레 관계)
svc.fit(X_train_scaled, y_train)
print("train accuracy: {:.3f}".format(svc.score(X_train_scaled, y_train)))
print("test accuracy: {:.3f}".format(svc.score(X_test_scaled, y_test)))