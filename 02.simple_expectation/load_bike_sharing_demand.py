import matplotlib
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import sklearn as svm
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

train = pd.read_csv('~/data/data/bike-sharing-demand/train.csv')

train['year'] = pd.DatetimeIndex(train['datetime']).year
train['month'] = pd.DatetimeIndex(train['datetime']).month
train['day'] = pd.DatetimeIndex(train['datetime']).day
train['hour'] = pd.DatetimeIndex(train['datetime']).hour
train['minute'] = pd.DatetimeIndex(train['datetime']).minute
train['second'] = pd.DatetimeIndex(train['datetime']).second
sns.countplot(train)
fig, ax= plt.subplots(ncols=3, nrows=2)
sns.scatterplot(train['year'], ax=ax[0,0])
sns.scatterplot(train['month'], ax=ax[0,1])
sns.scatterplot(train['day'], ax=ax[0,2])
sns.scatterplot(train['hour'], ax=ax[1,0])
sns.scatterplot(train['minute'], ax=ax[1,1])
sns.scatterplot(train['second'], ax=ax[1,2])
plt.show()

X = train[['season', 'holiday', 'workingday', 'weather', 'temp',
       'atemp', 'humidity', 'windspeed',
       'year', 'month', 'day', 'hour', 'minute', 'second']].values
y = train["count"].values

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=15)

model = LinearRegression()

model.fit(X_train, y_train)
predictions = model.predict(X_test)


mse = mean_squared_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5


plt.scatter(X_test[:, 4], y_test, label="true")
plt.scatter(X_test[:, 4], predictions, label="pred")
plt.legend()
plt.show()


plt.scatter(X_test[:, 6], y_test, label="true")
plt.scatter(X_test[:, 6], predictions, label="pred")
plt.legend()
plt.show()


