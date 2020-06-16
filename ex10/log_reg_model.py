import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_spliter import data_spliter
from my_logistic_regression import MyLogisticRegression as MyLogR


def select_category(y, category):
    new_y = y.copy()
    other_cat = np.unique(new_y)
    # Remvoving `category` from other_cat
    other_cat = other_cat[other_cat != category]

    # Selecting element
    to_0 = np.isin(new_y, other_cat)
    to_1 = new_y == category

    # Changing there values
    new_y[to_0] = 0
    new_y[to_1] = 1

    return new_y


def train_cat(x, y, cat, theta=None, alpha=0.001, n_cycle=1000):
    if theta is None:
        theta = np.ones(x.shape[1] + 1)

    y_prime = select_category(y, cat)
    lr = MyLogR(theta, alpha, n_cycle)
    lr.fit_(x, y_prime)
    # print(f"theta: {lr.theta}")

    return lr


def cost_cat(lr, x_test, y_test, cat):
    y_test_prime = select_category(y_test, cat)
    cost = lr.cost_(x_test, y_test_prime)

    print(f"cost: {lr.cost_(x_test, y_test_prime)}")
    return cost


def predict_with_one_vs_all(lr_lst, cat_lst, x):
    y_hat_lst = [lr.predict_(x) for lr in lr_lst]

    idx = np.argmax(np.concatenate(y_hat_lst, axis=1), axis=1)
    return cat_lst[idx].reshape(-1, 1)


def main():
    citizen_data = pd.read_csv("../resources/solar_system_census.csv")
    origin_data = pd.read_csv("../resources/solar_system_census_planets.csv")

    X = np.array(citizen_data[["height", "weight", "bone_density"]])
    Y = np.array(origin_data["Origin"])

    # Spliting train / test set
    x, x_test, y, y_test = data_spliter(X, Y, 0.7)

    # Training one logistic regression by categorie
    lr0 = train_cat(x, y, 0., theta=[[8.99820209], [-0.04211505],
                                     [-0.05411521], [3.46389832]], alpha=4e-4, n_cycle=1000)
    lr1 = train_cat(x, y, 1., theta=[[1.66331158e+00], [-5.00887941e-02],
                                     [2.81684154e-03], [7.91124245e+00]], alpha=4e-4, n_cycle=1000)
    lr2 = train_cat(x, y, 2., theta=[[-5.44251112], [-0.02352356],
                                     [0.13866041], [-5.77201278]], alpha=4e-4, n_cycle=1000)
    lr3 = train_cat(x, y, 3., theta=[[-4.81852712], [0.10656838],
                                     [-0.10904182], [-9.14504867]], alpha=4e-4, n_cycle=1000)

    # Predict x_test
    y_hat = predict_with_one_vs_all(
        [lr0, lr1, lr2, lr3], np.array([0., 1., 2., 3.]), x_test)

    # Checking if prediction is right
    unique, counts = np.unique(y_hat == y_test, return_counts=True)
    print(dict(zip(unique, counts)))


if __name__ == "__main__":
    main()
