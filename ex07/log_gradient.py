import numpy as np


def log_gradient(x, y, theta):
    """
    Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatible dimensions.
    Args:
        x: has to be an numpy.ndarray, a matrix of dimension m * n.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector (n +1) * 1.
    Returns:
        The gradient as a numpy.ndarray, a vector of dimensions n * 1, containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    if (x.size == 0 or y.size == 0 or theta.size == 0
        or x.ndim != 2 or y.ndim != 2 or theta.ndim != 2
            or x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]
            or y.shape[1] != 1 or theta.shape[1] != 1):
        return None

    x_prime = np.c_[np.ones(x.shape[0]), x]

    h = 1 / (1 + np.exp(-x_prime @ theta))
    return (x_prime.T @ (h - y)) / y.shape[0]


if __name__ == "__main__":
    # Example 1:
    y1 = np.array([1])
    x1 = np.array([4])
    theta1 = np.array([[2], [0.5]])
    print(log_gradient(x1, y1, theta1))
    # Output:
    # array([[-0.01798621],
    # [-0.07194484]])

    # Example 2:
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(log_gradient(x2, y2, theta2))
    # Output:
    # array([[0.3715235 ],
    # [3.25647547]])

    # Example 3:
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(log_gradient(x3, y3, theta3))
    # Output:
    # array([[-0.55711039],
    # [-0.90334809],
    # [-2.01756886],
    # [-2.10071291],
    # [-3.27257351]])
