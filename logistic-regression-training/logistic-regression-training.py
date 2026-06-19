import numpy as np

def _sigmoid(z):
    # Numerically stable sigmoid function
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples, n_features = X.shape
    
    # Initialize parameters
    w = np.zeros(n_features)
    b = 0.0
    
    for _ in range(steps):
        # Forward pass
        linear_model = np.dot(X, w) + b
        y_predicted = _sigmoid(linear_model)
        
        # Gradient calculation
        error = y_predicted - y
        dw = (1 / n_samples) * np.dot(X.T, error)
        db = (1 / n_samples) * np.sum(error)
        
        # Gradient descent update
        w -= lr * dw
        b -= lr * db
        
    return w, b