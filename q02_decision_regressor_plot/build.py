# default imports
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("./data/house_pricing.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

param_grid = {"max_depth": [2, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70]}
parameter = 'max_depth'


# Write your solution here :


def decision_regressor_plot(grid_obj1, param_grid, parameter, X_train, X_test, y_train, y_test):
    rs_cv = RandomizedSearchCV(grid_obj1, param_distributions=param_grid, n_iter=12)
    grid_obj = rs_cv.fit(X_train, y_train)
    mean_test_scores = grid_obj.cv_results_['mean_test_score']
    mean_train_scores = grid_obj.cv_results_['mean_train_score']
    plt.figure(figsize=(10, 6))
    x = np.arange(1, len(grid_obj.param_distributions[parameter]) + 1)
    plt.plot(x, mean_train_scores, c='b', label='Train set')
    plt.xticks(x, grid_obj.param_distributions[parameter])
    plt.plot(x, mean_test_scores, c='g', label='Test set')
    plt.legend(loc='upper left')
    plt.xlabel(parameter)
    plt.ylabel('mean scores')
    plt.show()


decision_regressor_plot(DecisionTreeRegressor(), param_grid, parameter, X_train, X_test, y_train, y_test)
