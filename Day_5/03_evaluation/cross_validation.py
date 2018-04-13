from sklearn.datasets.base import load_iris
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection._split import train_test_split, KFold
from sklearn.model_selection._validation import cross_val_score
iris = load_iris()
logreg = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

logreg = LogisticRegression().fit(X_train, y_train)
print("test accuracy: {:.2f}".format(logreg.score(X_test, y_test)))

scores = cross_val_score(logreg, iris.data, iris.target, cv=3)
print("cross validation: {}".format(scores))

kfold = KFold(n_splits=3, shuffle=True, random_state=0)

scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold)
print("cross validation: {}".format(scores))