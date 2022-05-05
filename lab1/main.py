import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib.colors import ListedColormap
from knn import KnnClassifier
cmap = ListedColormap([ '#00FF00', '#FF0000', '#0000FF'])

def getDataset():
  zurich = ['A','B','C','D','E','F','H']
  spot_size = ['X','R','S','A','H','K']
  spot_dist = ['X','O','I','C']
  data = pd.read_csv('data/data2.csv', delimiter=' ', header=None)
  for i in range(len(zurich)):
    data[0] = data[0].replace([zurich[i]], i)
  for i in range(len(spot_size)):
    data[1] = data[1].replace([spot_size[i]], i)
  for i in range(len(spot_dist)):
    data[2] = data[2].replace([spot_dist[i]], i)
  return data

def LOO():
  test_score = []
  k = []
  for i in range(3, 9, 2):
    print(i)
    clf = KnnClassifier(i)
    clf.fit(X_train.to_numpy(), y_train.to_numpy().flatten())
    predictions = clf.predict(X_test.to_numpy())
    test_score.append(np.sum(predictions == y_test.to_numpy().flatten()) / len(y_test.to_numpy().flatten()))
    k.append(i)
  return [k, test_score]
  
# TODO: recall and precision metrics

data = getDataset()
X = data[[i for i in range(10)]]
y = data[[10]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
plt.plot(LOO())
plt.show()

print(res)

