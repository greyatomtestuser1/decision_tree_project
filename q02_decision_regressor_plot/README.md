# decision_tree_project

### Task 1: Write a function called my_decision_regressor():

- Accepts the following parameters:
    * X_train, X_test, y_train, y_test & paramgrid
***
we made your life simpler use this parameters in paramgrid

paramgrid  =   {

              "max_depth" : [ 2, 3, 5, 6, 8, 10],
              "max_leaf_nodes": [5, 10, 15, 20], 
             "min_impurity_split": [0.1, 0.2, 0.3]
             }
 ***
- use gridsearch cv and input decisionTree regressor( use random state = 9, * mandatory), and paramgrid
- find predictions for X_test
- find r_square and best parameters
- use np.random.seed(9) in function


- Should return
    * r_square error
    * best paramaters