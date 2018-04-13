# -*- coding: ms949 -*-
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import mglearn

X, y = make_blobs(centers=2,            # centers: y값의 개수를 설정
                  random_state=5,       # random_state: 랜덤 시드값 (랜덤값이 변하지 않으면 일정한 랜덤값이 나옴)
                  n_samples=30)         # n_samples: 샘플 개수
print(X.shape, y.shape)

mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.legend(["class1","class2"], loc=4)
plt.xlabel("feature1")
plt.ylabel("feature2")
plt.show()

# Training과 Testing 나누어줌
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)                       # 구분하는 개수
clf.fit(X_train, y_train)                                       # 학습하는 명령어

print("test prediction : {}".format(clf.predict(X_test)))       # 예측 결과를 출력
print("test label : {}".format(y_test))
print("test accuracy : {}".format(clf.score(X_test, y_test)))

