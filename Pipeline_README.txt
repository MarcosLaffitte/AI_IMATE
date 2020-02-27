> Obj: Predict MEDV ($1000's) with linear model over some of the other 13 variables

> Input: Minimum value for abs of linear correlation (0.7 default)

0. get data from scikit, append target MEDV, count isnull
1. visual of data rows
2. visual of  MEDV distrubition
3. visual of linear correlation matrix with and without abs > 0.7 (or defined), ordered by correlation with MEDV
4. determine increasing tuples of variables (models)
5. for each model do
   - concatenate column names into X and spill MEDV into Y
   - for each trainingset-size in (60, 65, 70, 75, 80) do
     	 - determine training set as in [1] 
	 - train sklearn's linear regression model
	 - evaluate with rmss and aic
	 - save error
6. get best error
7. plot error lines of models and autput best model(s)
8. plot best model scatter of target vs variables in model

> Pipeline Refs

[1] https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155
[2] https://subscription.packtpub.com/book/programming/9781789804744/1/ch01lvl1sec11/our-first-analysis-the-boston-housing-dataset

> Dataset Ref

http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html

> Some technical Refs

- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- https://seaborn.pydata.org/generated/seaborn.heatmap.html
