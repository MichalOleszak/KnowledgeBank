import matplotlib.pyplot as plt
from scipy.stats import randint
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import Imputer, StandardScaler, FunctionTransformer, MaxAbsScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_selection import chi2, SelectKBest

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# Classification -------------------------------------------------------------------------------------------------------
digits = datasets.load_digits()


# kNN
# Display digit 1010 (MNIST data)
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
# Create feature and target arrays
X = digits.data
y = digits.target
# Split into training and test set
# stratify=y keeps the same percentages of different classes in training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)
# Fit the classifier to the training data
knn.fit(X_train, y_train)
# Predict test data
y_pred = knn.predict(X_test)


# Logistic regression
X = digits.data
y = digits.target
# Predict if the number is 1
y[y != 1] = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)


# Metrics for classification
# Accuracy
print(knn.score(X_test, y_test))
# Confusion matrix
print(confusion_matrix(y_test, y_pred))
# Whole report
print(classification_report(y_test, y_pred))
# ROC curve
# predict_proba() method returns the probability of a given sample being in a particular class
y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
# AUC
auc = roc_auc_score(y_test, y_pred_prob)
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')


# Overfitting & underfitting - plot accuracy for different values of k
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
# Loop over different values of k
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# Multiclass multilabel classification ---------------------------------------------------------------------------------
# Fit a separate classifier to each target, while train-test splitting such that all labels are in both sets

# Get train-test splitter
with open('multilabel_sample.py') as fd:
    exec(fd.read())
# Split
label_dummies = pd.get_dummies(df[target_cols])
X_train, X_test, y_train, y_test = multilabel_train_test_split(X, label_dummies, size=0.2, seed=123)
# Fit the classifier to the training data
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)


# Regression -----------------------------------------------------------------------------------------------------------
boston = datasets.load_boston()


# Linear regression on Boston Housing data
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.3, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
# Goodness-of-fit measures
R_sq = reg.score(X_test, y_test)
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))


# Cross-validation
reg = LinearRegression()
cv_scores = cross_val_score(reg, boston.data, boston.target, cv=5, scoring=None)  # scoring=roc_auc for classification
print(cv_scores)


# Ridge regression: OLS loss + alpha * sum of squared coeffs
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
ridge.predict(X_test)
ridge.score(X_test, y_test)


# LASSO regression: OLS loss + alpha * sum of absolute coeffs
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(X_train, y_train)
lasso.predict(X_test)
lasso.score(X_test, y_test)
# Using LASSO for feature selection - those coefs that are not shrunk to zero are selected
lasso.fit(X_train, y_train)
lasso_coef = lasso.coef_
names = boston.feature_names
plt.plot(range(len(names)), lasso_coef)
plt.xticks(range(len(names)), names, rotation=60)
plt.show()


# Elsastic Net
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}
elastic_net = ElasticNet()
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)
gm_cv.fit(X_train, y_train)

y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)



# Plot R-sqaured and its standard error for different values of alpha
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []
# Create a ridge regressor
ridge = Ridge(normalize=True)
# Compute scores over range of alphas
for alpha in alpha_space:
    ridge.alpha = alpha
    ridge_cv_scores = cross_val_score(ridge, boston.data, boston.target, cv=10)
    ridge_scores.append(np.mean(ridge_cv_scores))
    ridge_scores_std.append(np.std(ridge_cv_scores))

display_plot(ridge_scores, ridge_scores_std)


# Hyperparams tuning ---------------------------------------------------------------------------------------------------
digits = datasets.load_digits()
X = digits.data[0:100, ]
y = digits.target[0:100, ]

# Grid search
param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y)

knn_cv.best_params_
knn_cv.best_score_

# Random search
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
tree_cv.fit(X, y)

tree_cv.best_params_
tree_cv.best_score_


# Preprocessing data ---------------------------------------------------------------------------------------------------

# Categorical features: sklearn only accepts numeric values
df = pd.get_dummies(df, drop_first=True)

# Missing data & pipelines
features = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
            'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
            'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']
df = pd.read_csv('data/house-votes-84.csv', header=None, names=features)
df[df == 'y'] = 1
df[df == 'n'] = 0
df[df == '?'] = np.nan

# Dropping missings
print(df.isnull().sum())
print("Shape of Original DataFrame: {}".format(df.shape))
df = df.dropna()
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))

# Simple imputation & SVM pipeline
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
clf = SVC()
steps = [('imputation', imp),
        ('SVM', clf)]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(df.drop('party', axis=1), df.party, test_size=0.3, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Centering & scaling
df = df.dropna()
X_train, X_test, y_train, y_test = train_test_split(df.drop('party', axis=1), df.party, test_size=0.3, random_state=42)
steps = [('scaler', StandardScaler()),
         ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
knn_scaled = pipeline.fit(X_train, y_train)
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))



# Pipeline for classification ------------------------------------------------------------------------------------------
# Get data
features = ['party', 'infants', 'water', 'budget', 'physician', 'salvador',
            'religious', 'satellite', 'aid', 'missile', 'immigration', 'synfuels',
            'education', 'superfund', 'crime', 'duty_free_exports', 'eaa_rsa']
df = pd.read_csv('data/house-votes-84.csv', header=None, names=features)
df[df == 'y'] = 1
df[df == 'n'] = 0
df[df == '?'] = np.nan
# Setup the pipeline
steps = [('imputer', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
         ('scaler', StandardScaler()),
         ('SVM', SVC())]
pipeline = Pipeline(steps)
# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}
# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('party', axis=1), df.party, test_size=0.2, random_state=21)
# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters, cv=3)
# Fit to the training set
cv.fit(X_train, y_train)
# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)
# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))


# Pipeline for regression ----------------------------------------------------------------------------------------------
# Get data
boston = datasets.load_boston()
# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]
# Create the pipeline
pipeline = Pipeline(steps)
# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}
# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.3, random_state=42)
# Create the GridSearchCV object
gm_cv = GridSearchCV(pipeline, parameters, cv=3)
# Fit to the training set
gm_cv.fit(X_train, y_train)
# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))


# NLP: bag-of-words ----------------------------------------------------------------------------------------------------
# Create the token pattern: split on non-alpha numeric characters
tokens_alphanumeric = '[A-Za-z0-9]+(?=\\s+)'
# or split on whitespace
tokens_whitespace = '\\S+(?=\\s+)'
# Fill missing values in col of interes
df.col.fillna('', inplace=True)
# Instantiate the CountVectorizer
vec_alphanumeric = CountVectorizer(token_pattern=tokens_alphanumeric)
# Fit to the data
vec_alphanumeric.fit(df.col)

# Combining text columns for tokenization
# In order to get a bag-of-words representation for all of the text data in a DataFrame, one must first convert
# the text data in each row of the DataFrame into a single string.


def combine_text_columns(data_frame, to_drop):
    """ converts all text in each row of data_frame to single vector;
        to_drop should contain names of numeric cols and target variables """
    # Drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1)
    # Replace nans with blanks
    text_data.fillna('', inplace=True)
    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)


# Fit to data
text_vector = combine_text_columns(df)
vec_alphanumeric.fit_transform(text_vector)
print("There are {} alpha-numeric tokens in the dataset".format(len(vec_alphanumeric.get_feature_names())))


# Features interactions ------------------------------------------------------------------------------------------------
# Degree 2 means multiply 2 columns at a time, interaction_only=True excludes cols multiplied by themselves
# and include_bias=False excludes the intercept
x = np.array([[0, 1], [1, 1]])
interactions = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
interactions.fit_transform(x)
# If the data is large, use another version of PolynomialFeatures, based on sparse matrices: sparse_interactions.py


# Features vectorization using hashing ---------------------------------------------------------------------------------
# A way of inreasing memory efficiency without sacrificing much accuracy
# Hash function takes a token as input and outputs a hash value; we can limit the number of these values. Thus, each
# hash value may have multiple tokens assigned to it. Interestingly, this has little effect on model accuracy.
# Some problems are memory-bound and not easily parallelizable, and hashing enforces a fixed length computation instead
# of using a mutable datatype (like a dictionary).
text_data = combine_text_columns(X_train)
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
hashing_vec = HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC)
hashed_text = hashing_vec.fit_transform(text_data)


# Pipeline for numeric and categorical variables -----------------------------------------------------------------------
# Any step in the pipeline must be an object that implements the fit and transform methods. The FunctionTransformer
# creates an object with these methods out of any Python function that you pass to it.
get_text_data = FunctionTransformer(lambda x: x['text_col'], validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[['numeric_col_1', 'numeric_col_2']], validate=False)
just_text_data = get_text_data.fit_transform(df)
just_numeric_data = get_numeric_data.fit_transform(df)

# FeatureUnion joins results of multiple pipelines together
process_and_join_features = FeatureUnion(
            transformer_list=[
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )
# Instantiate nested pipeline
pl = Pipeline([
        ('union', process_and_join_features),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
# Fit pl to the training data
pl.fit(X_train, y_train)


# A complete pipeline in one piece -------------------------------------------------------------------------------------
# Get the dummy encoding of the labels
dummy_labels = pd.get_dummies(df[LABELS])
# Get the columns that are features in the original df
NON_LABELS = [c for c in df.columns if c not in LABELS]
# Split into training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(df[NON_LABELS], dummy_labels, 0.2, seed=123)
# Preprocess the text data
get_text_data = FunctionTransformer(combine_text_columns, validate=False)
# Preprocess the numeric data
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)
# Select 300 best features: The dim_red step uses a scikit-learn function called SelectKBest(), applying something
# called the chi-squared test to select the K "best" features.
chi_k = 300
# Create the token pattern
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
# The scale step uses a scikit-learn function called MaxAbsScaler() in order to squash the relevant features
# into the interval -1 to 1.
# The pipeline
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                     non_negative=True, norm=None, binary=False,
                                                     ngram_range=(1, 2))),  # Use both 1-grams and 2-grams of words
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(RandomForestClassifier(n_estimators=15)))
    ])
# Fit to the training data
pl.fit(X_train, y_train)
# Compute accuracy on test data
accuracy = pl.score(X_test, y_test)

