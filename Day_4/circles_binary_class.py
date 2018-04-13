# -*- coding: ms949 -*-
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import numpy as np
X, Y = make_circles(noise=0.25, random_state=1, factor=0.5)

# change class name
y_named = np.array(["blue", "red"])[Y]

X_train, X_test, y_train_named, y_test_named, y_train, y_test =\
train_test_split(X, y_named, Y, random_state=0)

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)

print("X_test.shape: {}".format(X_test.shape))
print("결정함수결과형태: {}".format(gbrt.decision_function(X_test).shape))