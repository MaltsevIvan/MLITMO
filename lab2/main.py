import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib.colors import ListedColormap
from linearRegressor import linearRegressor
from sklearn.linear_model import LinearRegression


def getDataset():
  data = pd.read_csv('data/forestfires.csv', delimiter=',')
  del data['month']
  del data['day']
  return (data-data.min())/(data.max()-data.min())

def mse(y_true, y_predict):
  return np.mean((y_true - y_predict)**2)

def sklearLinearReg(X_train, y_train, X_test, y_test):
  reg = LinearRegression().fit(X_train, y_train)
  y_predict = reg.predict(X_test)
  print('sklearLinearReg MSE:',mse(y_test, y_predict))

data = getDataset()

X = data.drop(['area'], axis=1)
y = data[['area']].to_numpy().flatten()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

regression = linearRegressor(lr=0.0025)
regression.fit(X_train, y_train)
weights = regression.getWeights()
y_predicted = regression.predict(X_test)

print(mse(y_test, y_predicted))

sklearLinearReg(X_train, y_train, X_test, y_test)
