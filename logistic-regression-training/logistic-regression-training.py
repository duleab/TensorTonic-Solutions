import numpy as np

def _sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    
    for _ in range(steps):
        linear_model = np.dot(X, w) + b
        y_predicted = _sigmoid(linear_model)
        
        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
        db = (1 / n_samples) * np.sum(y_predicted - y)
        
        w -= lr * dw
        b -= lr * db
        
    return w, b