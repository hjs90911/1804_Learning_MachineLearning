# -*- coding: ms949 -*-
from sklearn.datasets import load_digits
from sklearn.model_selection._split import train_test_split
from sklearn.linear_model.logistic import LogisticRegression

digits = load_digits()

print("Image Data Shape", digits.data.shape)
print("Label Data Shape", digits.target.shape)

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)    # 1행 5열 각각의 plot을 넣어줌
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)     # image를 보여줌
                                                                # 8X8로 다시 만들어줌
    plt.title('Training: %i\n' % label, fontsize=20)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=0, test_size=.25)

logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)

print(logisticRegr.score(X_train, y_train))
print(logisticRegr.score(X_test, y_test))
# print(logisticRegr.predict(X_test[0].reshape[0:10]))

print(logisticRegr.predict(X_test[9].reshape(1, -1)))   # 예측한 값
                                                        # 2차원으로 맞춰주기 위해 reshape
                                                        # 1차원 벡터를 2차원으로 (1, -1)을
print(y_test[9])                                        # 정답