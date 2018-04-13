# -*- coding: ms949 -*-
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mglearn
import matplotlib.pyplot as plt

X, y = mglearn.datasets.make_wave(n_samples=60)
print(X.shape, y.shape)
plt.plot(X, y,'o')  # 'o' make scatter plot
plt.ylim(-3, 3)
plt.xlabel("feature")
plt.ylabel("target")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)       # Transfer train data

plt.plot(X, y, 'o', X, lr.predict(X))
plt.show()

print("lr.coef_L: {}".format(lr.coef_))             # 기울기 (slope)
print("lr.intercept_: {}".format(lr.intercept_))    # y졀편

"""Accuracy will Increase if dataset increase"""
print("train score:{:.2f}".format(lr.score(X_train, y_train)))
print("test score:{:.2f}".format(lr.score(X_test, y_test)))
