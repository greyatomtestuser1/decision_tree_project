# Observing variation between error and max depth

If we can observe what is actually happening as we increase the maximum depth( the length of the longest path from a root to a leaf) 
along with the variation in errors we can get to know of that how depth is having an effect over error. 
 
## Write a function called decision_classifier_plot that :
- Plots the variation between depth and mean score.
- Plots mean_test_score and mean_train_score vs max_depth.
- Uses RandomizedSearchCV. 

### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- |
| grid_obj1 | Model | compulsory | |Model to be implemented  |
| param_grid | Dictionary | compulsory | | Dictionary containing ten values of max_depth |
| parameter | String | compulsory | | string named as max_depth |
| X_train | DataFrame | compulsory | | Dataframe containing feature variables for training |
| y_train | Series/DataFrame | compulsory | | Training dataset target Variable |
    
    
Note :
- While using the RandomizedSearchCV use the n_iter parameter as according to the values you are giving in max_depth,
like for 10 values use n_iter as 10.

Hint :
Use grid_obj1 as DecisionTreeClassifier.