import matplotlib
import numpy as np;
import matplotlib.pylab as plt
import pandas as pd
import sklearn as svm

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()
cancer_data = cancer.data
cancer_label = cancer.target
# print(digits.keys());
print(cancer.target_names);
print(cancer.DESCR);

X_train, X_test, y_train, y_test = train_test_split(cancer_data,
                                                    cancer_label,
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
print("선택 알고리즘 -> Random forest = 테스트데이터 대비 훈련데이터의 정확도는 decision tree가 미세하게 더 높으나, 결정트리의 집합이고, 서로 상관관계가 없는 데이터를 기준으로 효율성이 좋은 Random forest 선택)")
print("샘플데이터를 대량 추가하여 정확도를 비교해보기 위해 데이터 소스 찾는중...")