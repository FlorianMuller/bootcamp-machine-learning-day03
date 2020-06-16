import numpy as np


def data_spliter(x, y, proportion, seed=42):
    """
    Shuffles and splits the dataset (given by x and y) into a training and a
    test set, while respecting the given proportion of examples to be kept in
    the traning set.
    Args:
        x: has to be an numpy.ndarray, a matrix of dimension m * n.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        proportion: has to be a float, the proportion of the dataset that will
        be assigned to the training set.
    Returns:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.ndarray
        None if x or y is an empty numpy.ndarray.
        None if x and y do not share compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if (x.size == 0 or y.size == 0
            or x.ndim != 2 or y.ndim != 2
            or x.shape[0] != y.shape[0]) or y.shape[1] != 1:
        return None

    # copy to not shuffle original x and y
    x = x.copy()
    y = y.copy()

    # Shuffling x and y the same way
    r = np.random.RandomState(seed)
    r.shuffle(x)
    r.seed(seed)
    r.shuffle(y)

    # Slicing
    i = np.ceil(x.shape[0] * proportion).astype(int)
    return (x[:i], x[i:], y[:i], y[i:])


def print_res(splited, **kwargs):
    print(
        (f"x_train: {splited[0]}\n"
         f"x_test: {splited[1]}\n"
         f"y_train: {splited[2]}\n"
         f"y_test: {splited[3]}"),
        **kwargs
    )


if __name__ == "__main__":
    x1 = np.array([1, 42, 300, 10, 59])
    y = np.array([0, 1, 0, 1, 0])

    # Example 1:
    print_res(data_spliter(x1, y, 0.8), end="\n\n")
    # Output:
    # (array([1, 59, 42, 300]), array([10]), array([0, 0, 0, 1]), array([1]))

    # Example 2:
    print_res(data_spliter(x1, y, 0.5), end="\n\n")
    # Output:
    # (array([59, 10]), array([1, 300, 42]), array([0, 1]), array([0, 1, 0]))

    x2 = np.array([[1, 42],
                   [300, 10],
                   [59, 1],
                   [300, 59],
                   [10, 42]])
    y = np.array([0, 1, 0, 1, 0])

    # Example 3:
    print_res(data_spliter(x2, y, 0.8), end="\n\n")
    # Output:
    # (array([[10, 42],
    #         [300, 59],
    #         [59, 1],
    #         [300, 10]]), array([[1, 42]]), array([0, 1, 0, 1]), array([0]))

    # Example 4:
    print_res(data_spliter(x2, y, 0.5), end="\n\n")
    # Output:
    # (array([[59, 1],
    #         [10, 42]]), array([[300, 10],
    #                            [300, 59],
    #                            [1, 42]]), array([0, 0]), array([1, 1, 0]))

    # Be careful! The way tuples of arrays are displayed
    # could be a bit confusing...
    #
    # In the last example, the tuple returned contains the following arrays:
    # array([[59, 1],
    # [10, 42]])
    #
    # array([[300, 10],
    # [300, 59]
    #
    # array([0, 0])
    #
    # array([1, 1, 0]))
