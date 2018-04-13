# -*- coding: ms949 -*-
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection._split import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

# learning rate: Gradient descent���� ��ŭ�� ũ��� �����Ͽ� ������ ���� �����ϴ� ��
# default depth:3, tree:100��, learning rate=0.1
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
