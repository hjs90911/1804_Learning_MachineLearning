# -*- coding: ms949 -*-
import pandas as pd                     # pandas의 경우 문자열도 읽을 수 있음
from sklearn.model_selection import train_test_split
from sklearn.linear_model.base import LinearRegression

# pandas로 csv 파일 읽기
data = pd.read_csv("test-score.csv",
                   header=None,         # header가 없다고 전달
                   names=['1st','2nd','3rd','final'])
print(data.head())                      # 상위 데이터만 출력
print(data.shape)
print(data.isnull().sum().sum())        # null인 값을 모두 sum
print(data.describe())                  # min, 상위25%, 하위25%, 평균 등을 출력

X = data[['1st','2nd','3rd']]
y = data['final']

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.3,  # train과 test를 7:3으로 나눔 (default: .75)
                                                    random_state=107)
lr = LinearRegression().fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))