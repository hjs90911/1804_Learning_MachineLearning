# -*- encoding: ms949 -*-
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    random_state=1)
print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
print("��ȯ�� �� ũ��: {}".format(X_train_scaled.shape))
print("������ ���� �� Ư���� �ּڰ�:\n {}".format(X_train.min(axis=0)))
print("������ ���� �� Ư���� �ִ�:\n {}".format(X_train.max(axis=0)))
print("������ ���� �� Ư���� �ּڰ�:\n {}".format(X_train_scaled.min(axis=0)))
print("������ ���� �� Ư���� �ִ�:\n {}".format(X_train_scaled.max(axis=0)))