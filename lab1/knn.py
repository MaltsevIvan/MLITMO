import numpy as np
from collections import Counter

def euclidean_dist(x1, x2):
  return np.sqrt(np.sum(x1-x2)**2)

class KnnModel: 
  def __init__(self, k=3):
    self.k = k
  
  def fit(self, X, y):
    self.X_train = X
    self.y_train = y
    return self
  
  def predict (self, X):
    raise NotImplementedError
  
class KnnClassifier(KnnModel):
  def predict(self, X):
    predicred_labels = [self._predict(x) for x in X]
    return np.array(predicred_labels)

  def _predict(self, x):
    distances = [euclidean_dist(x, X_train) for X_train in self.X_train]
    # TODO: Kernal realization
    k_indixes = np.argsort(distances)[:self.k]
    k_nearest_lables  = [self.y_train[i] for i in k_indixes]
    most_common = Counter(k_nearest_lables).most_common(1)
    return most_common[0][0]