import mglearn
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
citibike = mglearn.datasets.load_citibike()

xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(),
                       freq='D')

# Extract target values (# of rentals)
y = citibike.values
# convert time to POSIX time using '%s'
X = citibike.index.strftime('%s').astype('int').reshape(-1, 1)
# Use first 184 data points as training set, rest as test
n_train = 184

# func to evaluate and plot a regressor on a given feature set
def eval_on_features(features, target, regressor):
    # split the given features into a training and test set
    X_train, X_test = features[:n_train], features[n_train:]
    # split target array
    y_train, y_test = target[:n_train], target[n_train:]
    regressor.fit(X_train, y_train)
    print 'test-set R^2:', regressor.score(X_test, y_test)

    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)

    plt.figure(figsize=(10,3))
    plt.xticks(range(0, len(X), 8), xticks.strftime("%a %m-%d"), rotation=90, ha="left")

    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="test")
    plt.plot(range(n_train), y_pred_train, '--', label="prediction train")
    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--', label="prediction test")
    plt.legend(loc=(1.01, 0))
    plt.xlabel("Date")
    plt.ylabel("Rentals")


regressor = RandomForestRegressor(n_estimators=500, random_state=0)
# eval_on_features(X, y, regressor)

# Predictions on training set are great, but random forest predicts a straight line on test set?
# Due to the nature of the POSIX time feature. It lies outside of the range of values in the
# training set, so trees cannot extrapolate to these ranges. The model predicts the target
# value of the closes training point, aka the last observed time

# we use our 'expert' knowledge to look at other factors in the data
# time of day and time of week seem to be very important. Add these features, drop POSIX

print '\n Using time of day feature...'
X_hour = citibike.index.hour.reshape(-1, 1)
# eval_on_features(X_hour, y, regressor)


# Pred is better, but clearly misses the weekly pattern in data
print '\n Using time of day and day of week...'
X_hour_week = np.hstack([citibike.index.dayofweek.reshape(-1, 1),
                         citibike.index.hour.reshape(-1, 1)])
# eval_on_features(X_hour_week, y, regressor)

# Now we have this model that effectively captures the periodic behavior. Let's try and use
# a less complex model than Random Forest, like LinearRegression
print '\n Now using Linear Regression...'
from sklearn.linear_model import LinearRegression
# eval_on_features(X_hour_week, y, LinearRegression())

# LinearRegression looks odd, since we encoded day of the week using integers, which
# are interpreted as categorical variables. Linear model can only learn a linear function of
# the time of day.
print '\n Now using OneHotEncoder...'
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()
# eval_on_features(X_hour_week_onehot, y, LinearRegression())

# Now, model learns one coefficient for each day of the week, but this pattern
# is shared across all days of the week. We can use interaction features
# to learn one coefficient for each combination of day and time of day.
print '\n Now using polynomial features...'
from sklearn.preprocessing import PolynomialFeatures

poly_transformer = PolynomialFeatures(degree = 2, interaction_only=True, include_bias=False)
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)
lr = Ridge()
# eval_on_features(X_hour_week_onehot_poly, y, lr)

# This linear model now performs similarly to the complex random forest. But it is much
# clearer on what is learned, one coefficient for each day and time, and we can plot these
# coefficients, unlike random forest.

hour = ['%02d:00' % i for i in range(0, 24, 3)]
day = ['Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
features = day + hour

features_poly = poly_transformer.get_feature_names(features)
features_nonzero = np.array(features_poly)[lr.coef_ != 0]
coef_nonzero = lr.coef_[lr.coef_ != 0]
plt.figure(figsize=(15, 2))
plt.plot(coef_nonzero, 'o')
plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation=90)
plt.xlabel("Feature magnitude")
plt.ylabel("Feature")

plt.show()