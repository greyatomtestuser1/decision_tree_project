# decision_tree_project

### Task 2: Write a function called my_decision_classifier()

- Accepts the following parameters:
    * X_train, X_test, y_train, y_test, param_grid, n_iter_search   
***
param_grid = {

             "max_depth" : [ 8, 10, 15, 20],
             "max_leaf_nodes": [2, 5, 9, 15, 20],
             "min_impurity_decrease": [0.1, 0.2, 0.3, 0.5]
             }

***             
  - use randomized search cv and input decisionTree Classifier( use random state = 9, * mandatory*), paramgrid and n_iter_search (Number of iterations the search will be run) = 10
  - find predictions for X_test
  - find accuracy and best parameters


- Should return
    * predictions for X_test
    * trained RandomizedSearchCV object