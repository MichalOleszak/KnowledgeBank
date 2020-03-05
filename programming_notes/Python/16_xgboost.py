import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pandas import DataFrameMapper
from sklearn_pandas import CategoricalImputer

# Classification -------------------------------------------------------------------------------------------------------
# Example use
X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)
xg_cl.fit(X_train, y_train)
preds = xg_cl.predict(X_test)
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))

# XGB is usually used with decision trees (CARTs) as base learners
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
dt_clf_4 = DecisionTreeClassifier(max_depth=4)
dt_clf_4.fit(X_train, y_train)
y_pred_4 = dt_clf_4.predict(X_test)
accuracy = float(np.sum(y_pred_4==y_test))/y_test.shape[0]
print("accuracy:", accuracy)

# Measuring accuracy via cross-validation
# XGBoost gets its lauded performance and efficiency gains by utilizing its own optimized data structure for
# datasets called a DMatrix. For just running the model, the input datasets are converted into DMatrix data on the fly,
# but when you use the xgboost cv object, you have to first explicitly convert your data into a DMatrix.
churn_dmatrix = xgb.DMatrix(data=X, label=y)
params = {"objective":"reg:logistic", "max_depth":3}
# Accuracy
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=3, num_boost_round=5,
                    metrics="error", as_pandas=True, seed=123)
print(cv_results)
print(((1-cv_results["test-error-mean"]).iloc[-1]))
# AUC
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=3, num_boost_round=5,
                    metrics="auc", as_pandas=True, seed=123)
print(cv_results)
print((cv_results["test-auc-mean"]).iloc[-1])


# Regression -----------------------------------------------------------------------------------------------------------
df = pd.read_csv('data/ames_housing_trimmed_processed.csv')
y = df[['SalePrice']]
X = df.drop('SalePrice', 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Decision trees as base learners
xg_reg = xgb.XGBRegressor(objective='reg:linear', booster='gbtree', n_estimators=10, seed=123)
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

# Linear base learners
# Rather uncommon, so one has to use XGBoost's own non-scikit-learn compatible functions to build the model.
# Boosted model is a weighted sum of linear models, thus it is linear itself.
DM_train = xgb.DMatrix(X_train, y_train)
DM_test = xgb.DMatrix(X_test, y_test)
params = {"booster":"gblinear", "objective":"reg:linear"}
xg_reg = xgb.train(params = params, dtrain=DM_train, num_boost_round=5)
preds = xg_reg.predict(DM_test)
rmse = np.sqrt(mean_squared_error(y_test,preds))
print("RMSE: %f" % (rmse))


# Regularization in XGBoost --------------------------------------------------------------------------------------------
# Regularization = control on model's complexity - we want models accurate and as simple as possibile
# Regularization parameters in XGBoost:
#  - gamma: minimum loss reduction allowed for a split to occur (for tree-based learners); higher values = fewer splits
#  - alpha: L1 regularization on leaf weights; higher values makes more leaf weights in base learners go to zero
#  - lambda: L2 regularization on leaf weights
housing_dmatrix = xgb.DMatrix(data=X, label=y)
reg_params = [1, 10, 100]
params = {"objective": "reg:linear", "max_depth": 3}
# Create an empty list for storing rmses as a function of l2 complexity
rmses_l2 = []
for reg in reg_params:
    params["lambda"] = reg
    cv_results_rmse = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2, num_boost_round=5,
                             metrics="rmse", as_pandas=True, seed=123)
    # Append best rmse (final round) to rmses_l2
    rmses_l2.append(cv_results_rmse["test-rmse-mean"].tail(1).values[0])
print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2", "rmse"]))


# Visualizing individual XGBoost trees ---------------------------------------------------------------------------------
# XGBoost has a plot_tree() function that makes this type of visualization easy. Once you train a model using the
# XGBoost learning API, you can pass it to the plot_tree() function along with the number of trees you want to plot
# using the num_trees argument.
params = {"objective":"reg:linear", "max_depth":2}
xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)

xgb.plot_tree(xg_reg, num_trees=0)
plt.show()

xgb.plot_tree(xg_reg, num_trees=9, rankdir="LR")
plt.show()


# Visualizing feature importances --------------------------------------------------------------------------------------
# One simple way of doing this involves counting the number of times each feature is split on across all boosting rounds
# (trees) in the model, and then visualizing the result as a bar graph, with the features ordered according to how many
# times they appear. XGBoost has a plot_importance() function that allows you to do exactly this.
xgb.plot_importance(xg_reg)
plt.show()


# Tuning the parameters ------------------------------------------------------------------------------------------------
# Tune number of boosting rounds
housing_dmatrix = xgb.DMatrix(data=X, label=y)
params = {"objective": "reg:linear", "max_depth": 3}
num_rounds = [5, 10, 15]
final_rmse_per_round = []

for curr_num_rounds in num_rounds:
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=curr_num_rounds,
                        metrics="rmse", as_pandas=True, seed=123)
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses, columns=["num_boosting_rounds", "rmse"]))

# Automated boosting round selection using early stopping
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, num_boost_round=50, early_stopping_rounds=10,
                    metrics='rmse', as_pandas=True, seed=123)
print(cv_results)

# XGBoost's hyperparams:
# For tree base learners:
# - learning rate (eta): affects how quickly model fits the residual error using additional base learners;
#                        low value = more boosting rounds needed to achieve the same reduction in residual error
# - gamma: minimum loss reduction allowed for a split to occur (for tree-based learners); higher values = fewer splits
# - alpha: L1 regularization on leaf weights; higher values makes more leaf weights in base learners go to zero
# - lambda: L2 regularization on leaf weights
# - max_depth: max depth per tree, a positive integer; how deeply each tree is allowed to grow
# - subsample: % samples used per tree, a value in (0,1); too low = underfitting, too high = overfitting
# - colsample_bytree: % features used per tree, a value in (0,1); small values ~ additional regularization to the model
# For linear base learners:
# - lambda: L2 reg on weights
# - alpha: L1 reg on weights
# - lambda_bias: L2 reg term on bias
# For both type of base learners, one can also tune number of estimators and number of boosting rounds.

# Grid search
gbm_param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'n_estimators': [50],
    'max_depth': [2, 5]
}
gbm = xgb.XGBRegressor()
grid_mse = GridSearchCV(param_grid=gbm_param_grid, estimator=gbm, scoring='neg_mean_squared_error', cv=4, verbose=1)
grid_mse.fit(X, y)
print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))

# Random search
gbm_param_grid = {
    'n_estimators': [25],
    'max_depth': range(2, 12)
}
gbm = xgb.XGBRegressor(n_estimators=10)
randomized_mse = RandomizedSearchCV(param_distributions=gbm_param_grid, estimator=gbm, scoring='neg_mean_squared_error',
                                    n_iter=5, cv=4, verbose=1)
randomized_mse.fit(X, y)
print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))


# Using XGBoost in pipelines -------------------------------------------------------------------------------------------
df = pd.read_csv('data/ames_unprocessed_data.csv')

# Encoding categorical columns I: LabelEncoder
# Scikit-learn has a LabelEncoder function that converts the values in each categorical column into integers.
df.LotFrontage = df.LotFrontage.fillna(0)
# Create a boolean mask for categorical columns
categorical_mask = (df.dtypes == object)
# Get list of categorical column names
categorical_columns = df.columns[categorical_mask].tolist()
# Create LabelEncoder object
le = LabelEncoder()
# Apply LabelEncoder to categorical columns
df[categorical_columns] = df[categorical_columns].apply(lambda x: le.fit_transform(x))

# Encoding categorical columns II: OneHotEncoder
# Categorical columns are now numeric, but one neighbourhood is not twice as 'big' as the other - need to create dummies
ohe = OneHotEncoder(categorical_features=categorical_mask, sparse=False)
df_encoded = ohe.fit_transform(df)

# Encoding categorical columns III: DictVectorizer
# The two step process above - LabelEncoder followed by OneHotEncoder - can be simplified by using a DictVectorizer.
# Using a DictVectorizer on a DataFrame that has been converted to a dictionary allows you to get label encoding
# as well as one-hot encoding in one go.
# Besides simplifying the process into one step, DictVectorizer has useful attributes such as vocabulary
# which maps the names of the features to their indices.
df_dict = df.to_dict("records")
dv = DictVectorizer(sparse=False)
df_encoded = dv.fit_transform(df_dict)

# Preprocessing within a pipeline
y = df[['SalePrice']]
X = df.drop('SalePrice', 1)
X.LotFrontage = X.LotFrontage.fillna(0)
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor())]
xgb_pipeline = Pipeline(steps)
cross_val_scores = cross_val_score(xgb_pipeline, X.to_dict("records"), y, cv=10, scoring="neg_mean_squared_error")
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))

# Large XGB pipeline example
# The chronic kidney disease dataset contains both categorical and numeric features, but contains lots of missing
# values. The goal here is to predict who has chronic kidney disease given various blood indicators as features.
#
# Library sklearn_pandas, that allows you to chain many more processing steps inside of a pipeline than are currently
# supported in scikit-learn (categorical imputation and the DataFrameMapper() class to apply any arbitrary
# sklearn-compatible transformer on DataFrame columns, where the resulting output can be either a NumPy array
# or DataFrame).
df = pd.read_csv('data/chronic_kidney_disease.csv')
# dtypes not correct, that's why code doesn't work later
df.columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'rbc', 'pc',
              'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class']
y = np.array(df[['class']])
y[y == 'ckd'] = 1
y[y == 'notckd'] = 0
X = df.drop('class', 1)

# Custom transformer to convert Pandas DataFrame into Dict (needed for DictVectorizer)
class Dictifier(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.to_dict('records')


# Create a boolean mask for categorical columns
categorical_feature_mask = X.dtypes == object
# Get list of categorical and non-categorical column names
categorical_columns = X.columns[categorical_feature_mask].tolist()
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()
# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper(
                                            [([numeric_feature], Imputer(strategy="median")) for numeric_feature in non_categorical_columns],
                                            input_df=True,
                                            df_out=True
                                           )
# Apply categorical imputer
categorical_imputation_mapper = DataFrameMapper(
                                                [(category_feature, CategoricalImputer()) for category_feature in categorical_columns],
                                                input_df=True,
                                                df_out=True
                                               )
# Concetrate imputers' results
numeric_categorical_union = FeatureUnion([
                                          ("num_mapper", numeric_imputation_mapper),
                                          ("cat_mapper", categorical_imputation_mapper)
                                         ])

# Create full pipeline
pipeline = Pipeline([
                     ("featureunion", numeric_categorical_union),
                     ("dictifier", Dictifier()),
                     ("vectorizer", DictVectorizer(sort=False)),
                     ("clf", xgb.XGBClassifier(max_depth=3))
                    ])
# Perform cross-validation
cross_val_scores = cross_val_score(pipeline, X, y, scoring="roc_auc", cv=3)
# Print avg. AUC
print("3-fold AUC: ", np.mean(cross_val_scores))

# Tune hyperparams
gbm_param_grid = {
    'clf__learning_rate': np.arange(0.05, 1, 0.05),
    'clf__max_depth': np.arange(3, 10, 1),
    'clf__n_estimators': np.arange(50, 200, 50)
}
# Perform RandomizedSearchCV
randomized_roc_auc = RandomizedSearchCV(estimator=pipeline, param_distributions=gbm_param_grid, n_iter=2, scoring='roc_auc', verbose=1)
# Fit the estimator
randomized_roc_auc.fit(X, y)
# Compute metrics
print(randomized_roc_auc.best_score_)
print(randomized_roc_auc.best_estimator_)