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
    plt.subplot(1, 5, index + 1)    # 1�� 5�� ������ plot�� �־���
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)     # image�� ������
                                                                # 8X8�� �ٽ� �������
    plt.title('Training: %i\n' % label, fontsize=20)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=0, test_size=.25)

logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)

print(logisticRegr.score(X_train, y_train))
print(logisticRegr.score(X_test, y_test))
# print(logisticRegr.predict(X_test[0].reshape[0:10]))

print(logisticRegr.predict(X_test[9].reshape(1, -1)))   # ������ ��
                                                        # 2�������� �����ֱ� ���� reshape
                                                        # 1���� ���͸� 2�������� (1, -1)��
print(y_test[9])                                        # ����