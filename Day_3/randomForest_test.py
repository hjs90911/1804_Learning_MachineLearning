# -*- coding: ms949 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,     # y에 대해서 골고루 섞이도록 함
                                                    random_state=42)

forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)

print("RandomForest Result")
print("Train Accuracy: {}".format(forest.score(X_train, y_train)))
print("Test Accuracy: {}".format(forest.score(X_test, y_test)))

""" scatter plot으로 데이터의 분포도를 그래프로 확인하기 """
import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1], c=y)    # c=y는 0과 1을 color로 구분
plt.show()

print()

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("DecisionTree Result")
print("train accuracy: {:.3f}".format(tree.score(X_train, y_train)))
print("test accuracy: {:.3f}".format(tree.score(X_test, y_test)))
