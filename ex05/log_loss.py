import numpy as np


# To test only
def logistic_predict_(x, theta):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    if (x.size == 0 or theta.size == 0
            or x.ndim != 2 or theta.ndim != 2
            or x.shape[1] + 1 != theta.shape[0] or theta.shape[1] != 1):
        return None

    x_prime = np.c_[np.ones(x.shape[0]), x]
    return 1 / (1 + np.exp(-x_prime @ theta))


def log_loss_(y, y_hat, eps=1e-15):
    """
    Computes the logistic loss value.
    Args:
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.
        eps: has to be a float, epsilon (default=1e-15)
    Returns:
        The logistic loss value as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y_hat.ndim == 1:
        y_hat = y_hat.reshape(-1, 1)

    if (y.size == 0 or y_hat.size == 0
        or y.ndim != 2 or y_hat.ndim != 2
            or y.shape != y_hat.shape):
        return None

    return -np.sum(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)) / y.shape[0]


if __name__ == "__main__":
    # Example 1:
    y1 = np.array([1])
    x1 = np.array([4])
    theta1 = np.array([[2], [0.5]])
    y_hat1 = logistic_predict_(x1, theta1)
    print(log_loss_(y1, y_hat1))
    # Output:
    # 0.01814992791780973

    # Example 2:
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    y_hat2 = logistic_predict_(x2, theta2)
    print(log_loss_(y2, y_hat2))
    # Output:
    # 2.4825011602474483

    # Example 3:
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    y_hat3 = logistic_predict_(x3, theta3)
    print(log_loss_(y3, y_hat3))
    # Output:
    # 2.9938533108607053
