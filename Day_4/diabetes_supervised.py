from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("diabetes.csv", names=[1,2,3,4,5,6,7,8,9], header=None)
X = data[[1,2,3,4,5,6,7,8]]
y = data[9]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=8)

""" Logistic Regression """
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
print("logistic: {}".format(lr.score(X_test, y_test)))

""" KNN """
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
print("KNN: {}".format(clf.score(X_test, y_test)))

""" DecisionTree """
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
print("Tree: {}".format(tree.score(X_test, y_test)))

""" RandomForest """
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print("Forest: {}".format(rf.score(X_test, y_test)))

""" SupportVectorMachine """
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
print("SVM: {}".format(svc.score(X_test, y_test)))

""" NeuralNetwork """
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(max_iter=10000)
mlp.fit(X_train, y_train)
print("MLP: {}".format(mlp.score(X_test, y_test)))
