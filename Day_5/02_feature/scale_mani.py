# -*- encoding: ms949 -*-
from sklearn.datasets import make_blobs
import matplotlib.pylab as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import mglearn

#sample data ���� 
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
#train data�� test data�� ���� 
X_train, X_test = train_test_split(X, random_state=5, test_size=.1)

#train data�� test data�� ������
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].scatter(X_train[:, 0], X_train[:, 1],
                c=mglearn.cm2(0), label="train set", s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^',
                c=mglearn.cm2(1), label="test set", s=60)
axes[0].legend(loc='upper left')
axes[0].set_title("original data")

# ����������
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# �������� ������ �������� ������
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                c=mglearn.cm2(0), label="train set", s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^',
                c=mglearn.cm2(1), label="test set", s=60)
axes[1].set_title("scaled data")
#plt.show()

# # �׽�Ʈ ��Ʈ�� �������� ���� �����մϴ�
# # �׽�Ʈ ��Ʈ�� �ּڰ��� 0, �ִ��� 1�� �˴ϴ�
# # �̴� ������ ���� ������ ����� �̷��� ����ؼ��� �ȵ˴ϴ�
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)
 
# �߸� ������ �������� �������� �׸��ϴ�
axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
                c=mglearn.cm2(0), label="training set", s=60)
axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1],
                marker='^', c=mglearn.cm2(1), label="test set", s=60)
axes[2].set_title("mistaken data")
 
for ax in axes:
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
fig.tight_layout()
plt.show()