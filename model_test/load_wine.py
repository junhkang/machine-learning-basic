import matplotlib
import numpy as np;
import matplotlib.pylab as plt
import pandas as pd
import sklearn as svm

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

wine = load_wine()
wine_data = wine.data
wine_label = wine.target
# print(digits.keys());
print(wine.target_names);
print(wine.DESCR);

X_train, X_test, y_train, y_test = train_test_split(wine_data,
                                                    wine_label,
                                                    test_size=0.2,
                                                    random_state=15)

decision_tree = DecisionTreeClassifier(random_state=15)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Accuracy = " + str(accuracy))

random_forest = RandomForestClassifier(random_state=32)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy = " + str(accuracy))

svm_model = svm.SVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("SVM = " + str(accuracy))

sgd_model = SGDClassifier()
sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("SGD Classifier = " + str(accuracy))

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
print("Logistic Regression = " + str(accuracy))
print("선택 알고리즘 -> Random forest = 테스트데이터 대비 훈련데이터의 정확도가 가장 높음(Logic regression의 경우 50% 정확도인데 비해 1.0의 정확도는 차이가 너무 크기에 Random forest가 부적합한 결과를 도출해내는 데이터 조건이 있는지 스터디 중)")