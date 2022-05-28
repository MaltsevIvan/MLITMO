import numpy as np

class linearRegressor:
  def __init__(self, lr=0.001, n_iters=1000):
    self.lr = lr
    self.n_iters = n_iters
    self.weights = None
    self.bias = None
  
  def fit(self, X, y, lam=0):
    n_samples, n_features = X.shape
    X = np.concatenate((np.ones((n_samples, 1)), X), axis=1)
    self.weights = np.random.randn(n_features + 1, )
    self.bias = 0 

    for _ in range(self.n_iters):
      y_predicted = np.dot(X, self.weights) + self.bias
      error = y_predicted - y

      # gradients
      dw = (1/n_samples) * np.dot(X.T, (error)) + (lam * self.weights)
      db = (1/n_samples) * np.sum(error) + (lam * self.weights)
      
      self.weights -= self.lr * dw
      self.bias -= self.lr * db

  def predict(self, X): 
    y_approximated = np.dot(X, self.weights) + self.bias
    return y_approximated
  
  def getWeights(self):
    return self.weights