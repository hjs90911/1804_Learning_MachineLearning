# -*- encoding: ms949 -*-
import numpy as np
import mglearn

X, y = mglearn.datasets.make_wave(n_samples=100)
bins = np.linspace(start=-3, stop=3, num=11)

print(bins)

which_bin = np.digitize(X, bins=bins)
# print(which_bin)    # ������ ����� �� ��° �������� ���
print(which_bin[:5])
print(X[:5])