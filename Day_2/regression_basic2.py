# -*- coding: ms949 -*-
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import numpy as np

boston = load_boston()
print(boston.data.shape)
print(boston.target.shape)

X = boston.data
y = boston.target

X = MinMaxScaler().fit_transform(X)
X = PolynomialFeatures(degree=2,        # X값에 X^2값도 추가함
                       include_bias=False
                       ).fit_transform(X)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lr = LinearRegression().fit(X_train, y_train)
print("lr.coef_: {}.".format(lr.coef_))
print("train score: {:.2f}".format(lr.score(X_train, y_train)))
print("test score: {:.2f}".format(lr.score(X_test, y_test)))

# import numpy as np
# print("maximum value of lr.coef_: {}".format(np.max(lr.coef_)))
# print("minimum value of lr.coef_: {}".format(np.min(lr.coef_)))
# print(np.argmax(lr.coef_))

"""Draw boxplot"""
# import matplotlib.pyplot as plt
# plt.boxplot(X)
# plt.show()

print()

print("Ridge regression")
rd = Ridge().fit(X_train, y_train)
print("train score: {:.2f}".format(rd.score(X_train, y_train)))
print("test score: {:.2f}".format(rd.score(X_test, y_test)))

print()

print("Ridge regression with alpha=10")
rd10 = Ridge(alpha=10).fit(X_train, y_train)
print("train score: {:.2f}".format(rd10.score(X_train, y_train)))
print("test score: {:.2f}".format(rd10.score(X_test, y_test)))

print()

print("Ridge regression with alpha=0.1")
rd01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("train score: {:.2f}".format(rd01.score(X_train, y_train)))
print("test score: {:.2f}".format(rd01.score(X_test, y_test)))

print()

print("Lasso regression with alpha=0.01")
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("train score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("test score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("feature count: {}".format(np.sum(lasso001.coef_ != 0)))

print("Lasso regression with alpha=0.0001")
lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("train score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("test score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("feature count: {}".format(np.sum(lasso00001.coef_ != 0)))
