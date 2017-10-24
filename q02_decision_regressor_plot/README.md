# Observing variation between error and max depth

If we can observe what is actually happening as we increase the maximum depth( the length of the longest path from a root to a leaf) 
along with the variation in errors we can get to know of that how depth is having an effect over error. 
 
## Write a function called decision_regressor_plot that :
- Plots the variation between depth and mean square error.
- Plots test_scores vs max_depth and train_scores vs max_depth (in the same plot).


### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- |
| X_train | DataFrame | compulsory | | Dataframe containing feature variables for training |
| X_test | DataFrame | compulsory | | Dataframe containing feature variables for testing |
| y_train | Series/DataFrame | compulsory | | Training dataset target Variable |
| y_test | Series/DataFrame | compulsory | | Test dataset target Variable |
| depths | List | compulsory | | List of depths to be checked for model performance|


### Returns
None

Please Compare your plot with the decision_regressor_plot.png in the images directory
https://github.com/commit-live-students/decision_tree_project/blob/master/images/decision_regressor_plot.png
