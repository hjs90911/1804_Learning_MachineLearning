# -*- coding: ms949 -*-
import pandas as pd
from sklearn.model_selection._split import train_test_split
from sklearn.linear_model.logistic import LogisticRegression

data = pd.read_csv("diabetes.csv",
                   header=None,
                   names=['1st','2nd','3rd','4th','5th','6th','7th','8th','result'])

print(data.head())

X = data[['1st','2nd','3rd','4th','5th','6th','7th','8th']]
y = data['result']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.25,
                                                    random_state=20)

lr = LogisticRegression()
lr.fit(X_train, y_train)

print(lr.predict(X_test[0:10]))

print(lr.score(X_test, y_test))
