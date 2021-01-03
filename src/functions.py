import numpy as np

# Activation
def relu(X):
    return np.maximum(0, X)


def relu_derivative(X_passed):
    """"Derivative of the relu function"""
    X_ = np.zeros_like(X_passed)
    X_[X_passed > 0] = 1
    return X_


def softmax(X):
    def _softmax(X):
        const = np.max(X)
        exp_X = np.exp(X - const)
        return exp_X / np.sum(exp_X)

    logits = []
    for x in X:
        logit = _softmax(x)
        logits.append(logit)
    return np.array(logits)
