import sklearn
import matplotlib.pyplot as plt
import numpy as np


from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()

# deterministic random numbers
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))

# add noise features to the dataset
X_w_noise = np.hstack([cancer.data, noise])

X_train, X_test, y_train, y_test = train_test_split(
    X_w_noise, cancer.target, random_state=0, test_size=.5)

# use f_classif (the default classification test) and SelectPercentile
# to get 50% of features

select = SelectPercentile(percentile=50)
select.fit(X_train,y_train)

# transform training set
X_train_selected = select.transform(X_train)

print 'X train shape:', X_train.shape
print 'X train selected shape:', X_train_selected.shape

# number of features reduced from 80 to 40 (50% of the original number of features)
# can see which features were selected with get_support method
# Returns a boolean mask of the selected features

# mask = select.get_support()
# print mask

# lets compare performance of logit with univariate vs logit with only selected

X_test_selected = select.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print 'score with all features', lr.score(X_test, y_test)

lr.fit(X_train_selected, y_train)
print 'score with selected features', lr.score(X_test_selected, y_test)

# removing noise features helped improve performance, despite losing some original
# features. Univariate selection can be helpful if too many features, or many
# uninformative features.

# MODEL BASED FEATURE SELECTION
# Model based feature selection uses supervised ML model
# to judge feature importance, keeps only the most important

print '\n Starting Model-Based Feature Selection... \n'

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

select = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')

# SelectFromModel class selects all features that have an importance mceasure
# greater than the provided threshold. We used a median as a threshold, so
# half of the features will be selected.

select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)

# we can look at the features in the same way as before, we see all but 2 of the
# original features were selected, but since we specified 40 features, also took
# some noise

X_test_l1 = select.transform(X_test)
score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test)
print 'L1 test score: ', score

# ITERATIVE FEATURE SELECTION
# series of models are built, in two ways:
# 1) start with 0, add features one by one until some stopping criteria
# 2) start with all, drop features one by one until some stopping criteria
# RFE - Recursive Feature Elimination (RFE)
#  - starts with all, builds model, discards least important feature according to model
#  - new model built with remaining features, and so on
#  - to work, model must provide some way to determine feature importance (like model-based)
# Below is an example of RFE using the same RandomForestClassifier as above.

print '\n Starting Iterative Feature Selection... \n'

from sklearn.feature_selection import RFE
select = RFE(RandomForestClassifier(n_estimators=100, random_state=42),
             n_features_to_select=40)

select.fit(X_train, y_train)
## visualize selected features
# mask = select.get_support()
# print mask

# All but one of the original features were selected, but RFE model took much longer since it
# trained a random forest model 40 times (for each feature dropped)
# Testing Accuracy:
X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)

score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
print 'RFE test score: ', score















