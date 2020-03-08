# XGBOOST_pipeline
This **Jupyter** notebook is about a cross validated approach in building an xgboost model for customer churn prediction on a **Kaggle** dataset [here](https://www.kaggle.com/blastchar/telco-customer-churn).
The building comes after some preprocessing, in particular
- feature engineering
- 2 different encoding (mean and one hot)
- correlation analysis and following feature dropping

## Main Noticeble:
- single parameter evaluation and elbow method choice for a neg loss function
  - n_estimators number of boosted trees to fit. To a point, more are better but will take more time.
  - max_depth Maximum tree depth for base learners. Used to control overfitting.
  - learning_rate learning rate is commonly set = 0.3 but in this case could be smaller due to the shape of the dataset
  - min_child_weight Minimum sum of instance weight (hessian) needed in a child. Used to control over-fitting. Values that are too high can lead to under-fitting. If the classes are highly unbalanced, lower values (even 1) can be alright.
  - scale_pos_weight Control the balance of positive and negative weights, useful for unbalanced classes, that is why upsampling is not used in this model. The unbalance issue can be sized by setting the parameter, as documentiation suggests equal to the ratio of negative values (majority) over the positive value (minority)
- randomized grid search on a 5-k fold (not stratified because we have already train-test splitted with a Stratified Shuffle Split) with 5000 and 2500 n_interations (computing time: 5 hours and 2 and half hours, big-o:n )
- best performing model according to a summary table that reaches all the summaries among random forest, logisting with smoothing SVM gradient boosting.

