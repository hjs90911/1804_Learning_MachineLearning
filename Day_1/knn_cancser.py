# -*- coding: ms949 -*-
from sklearn.datasets import load_breast_cancer
import numpy as np

cancer = load_breast_cancer()
print(cancer.keys())
print(cancer.data.shape)    # (행, 열)로 출력 569개의 데이터가 들어있다
print(cancer.target)        # 0과 1로만 출력
print(cancer.feature_names)

print("count per class: \n{}".format(
      {n: v for n,
       v in zip(cancer.target_names,            # zip은 묶어주는 역할
                np.bincount(cancer.target))}))  # 도수를 구해주는 함수

# Training과 Testing을 나눔
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    stratify=cancer.target, # 데이터를 섞어줌
                                                    random_state=66)


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train, y_train)       # 학습 시작

print(clf.score(X_test, y_test))

# Draw graph
import matplotlib.pylab as plt
training_accuracy = []
test_accuracy = []
# 1 ~ 10까지 n_neighbors를 적용
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))
    
plt.plot(neighbors_settings, training_accuracy, label="train accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()