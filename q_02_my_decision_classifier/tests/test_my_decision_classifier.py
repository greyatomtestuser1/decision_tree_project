import sys, os
sys.path.append(os.path.join(os.path.dirname(os.curdir)))

from unittest import TestCase

import pandas as pd
import numpy as np
from  sklearn.model_selection import train_test_split

data = pd.read_csv("./data/loan_prediction.csv")
X = data.iloc[:,0:4]
y = data.iloc[:,4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

param_grid = { "max_depth" : [ 8, 10, 15, 20],
             "max_leaf_nodes": [2, 5, 9, 15, 20],
             "min_impurity_decrease": [0.1, 0.2, 0.3, 0.5]}

from q_02_my_decision_classifier.build import my_decision_classifier

class TestMy_decision_classifier(TestCase):
    def test_my_decision_classifier(self):
        np.random.seed(9)
        accuracy, best_params = my_decision_classifier(X_train, X_test, y_train, y_test, param_grid, 10)
        self.assertAlmostEqual(accuracy, 0.762162162162, places=3)


