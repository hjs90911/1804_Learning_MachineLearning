# -*- coding: ms949 -*-
import pandas as pd
import matplotlib.pylab as plt
from sklearn.tree.tree import DecisionTreeRegressor

""" Show data by graph """
ram_prices = pd.read_csv("ram_price.csv")

plt.semilogy(ram_prices.date, ram_prices.price)  # logScale�� �ٲ���
plt.xlabel("year")
plt.ylabel("prices ($/Mbyte")
plt.show()

""" Prediction of RAM prices """
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

# ���� ������ ���� ��¥ Ư������ �̿�
X_train = data_train.date[:, np.newaxis]    # 1������ 2�������� �������(rank�� ����)
# �����Ϳ� Ÿ���� ���踦 �����ϰ� ����� ���� �α� �����Ϸ� �ٲ�
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)

# ������ ��ü �Ⱓ�� ���ؼ� ����
X_all = ram_prices.date[:, np.newaxis]

pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

# ������ ���� �α� �������� �ǵ���
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

plt.semilogy(data_train.date, data_train.price, label="train data")
plt.semilogy(data_train.date, data_train.price, label="test data")
plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
plt.semilogy(ram_prices.date, price_lr, label="Linear regression")
plt.legend()
plt.show()