

from unittest import TestCase

import pandas as pd
import numpy as np
from  sklearn.model_selection import train_test_split

param_grid = {"max_depth" : [ 2, 3, 5, 6, 8, 10, 15, 20, 30, 50],
             "max_leaf_nodes": [2,3,4,5, 10, 15, 20],
             "min_impurity_decrease": [0.1, 0.2, 0.3,0.5]}

data = pd.read_csv("data/house_pricing.csv")

X = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

from ..build import my_decision_regressor

class TestMy_decision_regressor(TestCase):
    def test_my_decision_regressor(self):
        np.random.seed(9)
        r_square, best_params = my_decision_regressor(X_train, X_test, y_train, y_test, param_grid)
        self.assertAlmostEqual(r_square,0.61841868188037852, places=3)
