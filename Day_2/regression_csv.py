# -*- coding: ms949 -*-
import pandas as pd                     # pandas�� ��� ���ڿ��� ���� �� ����
from sklearn.model_selection import train_test_split
from sklearn.linear_model.base import LinearRegression

# pandas�� csv ���� �б�
data = pd.read_csv("test-score.csv",
                   header=None,         # header�� ���ٰ� ����
                   names=['1st','2nd','3rd','final'])
print(data.head())                      # ���� �����͸� ���
print(data.shape)
print(data.isnull().sum().sum())        # null�� ���� ��� sum
print(data.describe())                  # min, ����25%, ����25%, ��� ���� ���

X = data[['1st','2nd','3rd']]
y = data['final']

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.3,  # train�� test�� 7:3���� ���� (default: .75)
                                                    random_state=107)
lr = LinearRegression().fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))