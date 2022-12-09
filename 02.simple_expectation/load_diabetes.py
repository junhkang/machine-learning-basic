import matplotlib
import numpy as np;
import matplotlib.pylab as plt
import pandas as pd
import sklearn as svm

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

diabetes = load_diabetes()
LEARNING_RATE = 0.99
r = 100000
df_x = diabetes.data
df_y = diabetes.target
np_x = np.array(df_x)
np_y = np.array(df_y)               #0.2            0.1         0.01            0.5                 0.001/200000
np_x = np.delete(np_x, 9, axis=1)   #2946.4605                                  2937.2121               3554.7540
#np_x = np.delete(np_x, 8, axis=1)   #3153.8699                  3930.6088       3100.9387              3930
#np_x = np.delete(np_x, 7, axis=1)   #3211.1227  3290.9454       4047                            3302
#np_x = np.delete(np_x, 6, axis=1)
# np_x = np.delete(np_x, 5, axis=1)
# np_x = np.delete(np_x, 4, axis=1)
# np_x = np.delete(np_x, 3, axis=1)
# np_x = np.delete(np_x, 2, axis=1)
X_train, X_test, y_train, y_test = train_test_split(np_x,
                                                    np_y,
                                                    test_size=0.2,
                                                    random_state=15)

W = np.random.rand(len(np_x[1]))
b = np.random.rand()
losses = []

def model(X, W, b):
    predictions = 0
    for i in range(len(W)):
        predictions += X[:, i] * W[i]
    predictions += b
    return predictions

def MSE(a, b):
    mse = ((a - b) ** 2).mean()  # 두 값의 차이의 제곱의 평균
    return mse
def loss(X, W, b, y):
    predictions = model(X, W, b)
    L = MSE(predictions, y)
    return L




def gradient(X, W, b, y):
    # N은 데이터 포인트의 개수
    N = len(y)
    # y_pred 준비
    y_pred = model(X, W, b)
    # 공식에 맞게 gradient 계산
    dW = 1 / N * 2 * X.T.dot(y_pred - y)
    # b의 gradient 계산
    db = 2 * (y_pred - y).mean()
    return dW, db

for i in range(1, r):
    dW, db = gradient(X_train, W, b, y_train)
    W -= LEARNING_RATE * dW
    b -= LEARNING_RATE * db
    L = loss(X_train, W, b, y_train)
    losses.append(L)
    if i % 1000 == 0:
        print('Iteration %d : Loss %0.4f' % (i, L))
prediction = model(X_test, W, b)
mse = loss(X_test, W, b, y_test)
plt.scatter(X_test[:, 0], y_test)
plt.scatter(X_test[:, 0], prediction)
plt.show()
