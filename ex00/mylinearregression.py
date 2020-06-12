import numpy as np


class MyLinearRegression():
    """ My personnal linear regression class to fit like a boss """

    def __init__(self, thetas, alpha=0.001, n_cycle=1000):
        self.alpha = alpha
        self.max_iter = n_cycle
        self.thetas = np.array(thetas, copy=True).flatten().astype("float64")

    def fit_(self, x, y):
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.flatten()

        if (x.size == 0 or y.size == 0 or self.thetas.size == 0
            or x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]
                or x.shape[1] + 1 != self.thetas.shape[0]):
            return None

        x_padded = np.c_[np.ones(x.shape[0]), x]
        for _ in range(self.max_iter):
            nabla = x_padded.T.dot(x_padded.dot(self.thetas) - y) / y.shape[0]
            self.thetas = self.thetas - self.alpha * nabla

        return self.thetas

    def predict_(self, x):
        if x.ndim == 1:
            x = x[:, np.newaxis]

        if (x.size == 0 or self.thetas.size == 0
                or x.ndim != 2 or x.shape[1] + 1 != self.thetas.shape[0]):
            return None

        # np.dot(a,b) if a is an N-D array and b is a 1-D array
        # => it is a sum product over the last axis of a and b.
        x_padded = np.c_[np.ones(x.shape[0]), x]
        return x_padded.dot(self.thetas)

    def cost_elem_(self, x, y):
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.flatten()

        y_hat = self.predict_(x)

        if (y.size == 0 or y.ndim != 1
                or y_hat is None or y.shape != y_hat.shape):
            return None

        return ((y_hat - y) ** 2) / (2 * y.shape[0])

    def cost_(self, x, y):
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.flatten()

        y_hat = self.predict_(x)

        if (y.size == 0 or y.ndim != 1
                or y_hat is None or y.shape != y_hat.shape):
            return None

        y_diff = y_hat - y
        return np.dot(y_diff, y_diff) / (2 * y.shape[0])
