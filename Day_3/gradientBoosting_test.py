# -*- coding: ms949 -*-
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection._split import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

# learning rate: Gradient descent에서 얼만큼의 크기로 측정하여 내려갈 지를 설정하는 값
# default depth:3, tree:100개, learning rate=0.1
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
print("Train Accuracy: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Test Accuracy: {:.3f}".format(gbrt.score(X_test, y_test)))

gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
print("Train Accuracy: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Test Accuracy: {:.3f}".format(gbrt.score(X_test, y_test)))

gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)
print("Train Accuracy: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Test Accuracy: {:.3f}".format(gbrt.score(X_test, y_test)))
