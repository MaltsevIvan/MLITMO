import numpy as np
from collections import Counter

def euclidean_dist(x1, x2):
  return np.sqrt(np.sum(x1-x2)**2)

class KnnModel: 
  def __init__(self, k=3):
    self.k = k

  def getMargin (self, X, Y, x, y):
    y_unique = np.unique(Y)
    return self.getMaxDist(X, Y, x, y) - self.getMaxDist(X, Y, x, np.delete(y_unique, np.where(y_unique==y)))
  
  def stolp (self, noises, threshold):
    # Delete all the outliers
    start = len(self.y_train)
    countDeleted = 0
    for i in range(len(self.y_train)):
      if (i > start - countDeleted): break
      if (self.getMargin(self.X_train, self.y_train, self.X_train[i-countDeleted], self.y_train[i-countDeleted]) <= noises):
        self.X_train = np.delete(self.X_train, i-countDeleted, 0)
        self.y_train = np.delete(self.y_train, i-countDeleted)
        countDeleted = countDeleted + 1
    print(start - len(self.y_train))

    # Get one standard value from each class(y)
    uniqueClasses = np.unique(self.y_train)
    X_standards = [] 
    y_standrds = np.array([]) 
    for uClass in uniqueClasses:
      margins_i = np.array([], dtype=np.uint32)
      margins = np.array([])
      for j in range(len(self.y_train)):
        if (self.y_train[j] == uClass):
          margins = np.append(margins, self.getMargin(self.X_train, self.y_train, self.X_train[j], uClass))
          margins_i = np.append(margins_i, j)
      max_i = margins_i[np.argsort(margins)[::-1][0]]

      X_standards.append(np.array(self.X_train[max_i]))
      y_standrds = np.append(y_standrds, uClass)
      self.X_train = np.delete(self.X_train, max_i, 0)
      self.y_train = np.delete(self.y_train, max_i)
      countDeleted = countDeleted + 1

    # Get more standard values till threshold will be less than current count of margins 
    print('standards', countDeleted)
    print(X_standards)
    n = len(self.y_train)
    while (n > 0):
      margins_i = np.array([], dtype=np.uint32)
      margins = np.array([])
      for i in range(n):
        m = self.getMargin(X_standards, y_standrds, self.X_train[i], self.y_train[i])
        if (m <= 0):
          margins = np.append(margins, m)
          margins_i = np.append(margins_i, i)
      print(len(margins_i))
      if (len(margins_i) <= threshold): break
      min_i = margins_i[np.argsort(margins)[0]]
      X_standards.append(np.array(self.X_train[min_i]))
      y_standrds = np.append(y_standrds, self.y_train[min_i])
      self.X_train = np.delete(self.X_train, min_i, 0)
      self.y_train = np.delete(self.y_train, min_i)
      countDeleted = countDeleted + 1
      n = n - 1

    self.y_train = y_standrds
    self.X_train = X_standards

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y 
    return self

  def getMaxDist (self):
    raise NotImplementedError

  def predict (self, X):
    raise NotImplementedError
  
class KnnClassifier(KnnModel):
  def predict(self, X):
    predicred_labels = [self._predict(x) for x in X]
    return np.array(predicred_labels)

  def _predict(self, x):
    distances = [euclidean_dist(x, X_train) for X_train in self.X_train]
    k_indixes = np.argsort(distances)[:self.k]
    k_nearest_lables  = [self.y_train[i] for i in k_indixes]
    
    kerenel_values = np.zeros(max(k_nearest_lables)+1)
    max_distance = max([distances[i] for i in k_indixes])
    if (max_distance == 0 ): return k_nearest_lables[0]
    for i in range(self.k):
      distance = distances[k_indixes[i]]
      kerenel_values[k_nearest_lables[i]] += 1 - ((max_distance - distance) / max_distance)
    return kerenel_values.argmin()
  
  def getMaxDist (self, X, Y, x, y):
    distances = [euclidean_dist(x, X_train) for X_train in X]
    k_indixes = np.argsort(distances)[:self.k]
    k_distances = np.sort(distances)[:self.k]
    k_nearest_lables  = [Y[i] for i in k_indixes]
    k_d = np.array([])

    for i in range(len(k_nearest_lables)):
      if (np.any(y == k_nearest_lables[i])):
        k_d = np.append(k_d, k_distances[i])
    if(len(k_d) == 0): return 0
    return k_d.argmax()
