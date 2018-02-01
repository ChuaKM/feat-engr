import sklearn
import matplotlib.pyplot as plt
import mglearn
import scipy
import numpy as np
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# build linear regression and decision tree model
from mglearn.datasets import make_wave
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

# reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
# plt.plot(line, reg.predict(line), label = "decision tree")
#
# reg = LinearRegression().fit(X, y)
# plt.plot(line, reg.predict(line), label = "linear regression")

# plt.plot(X[:, 0], y, 'o', c='k')
# plt.ylabel("Regression Output")
# plt.xlabel("Input Feature")
# plt.legend()
# plt.show()


# partition the inout range into 10 bins. using np.linspace
# use 11 entries which creates 10 bins (spaces between the entries)
bins = np.linspace(-3, 3, 11)
# print("bins: {}".format(bins))

# record what bin each datapoint falls into, using np.digitize
which_bin = np.digitize(X, bins=bins)
# print("\n", X[:5])
# print("\n", which_bin[:5])

# we changed the single continous input feature in wave dataset to a categorical feature
# encodes which bin a data point is in
# now, transform to a one-hot encoding to use scikit learn on dataset
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)
print(X_binned[:5])
# we specified 10 bins, X_binned contains 10 features
print(format(X_binned.shape))

# build new lin reg model and new dec tree model on the one-hot-encoded data
line_binned = encoder.transform(np.digitize(line, bins=bins))
reg = LinearRegression().fit(X_binned, y)

# Binned Linear Regression Plot vs Binned Decision Tree
# plt.plot(line, reg.predict(line_binned), label = "linear regression binned")
# reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
# plt.plot(line, reg.predict(line_binned), 'r--', label = "Decision Tree binned")
#
# plt.plot(X[:, 0], y, 'o', c='k')
# plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
# plt.legend(loc="best")
# plt.ylabel("Regression output")
# plt.xlabel("Input feature")

# Linear models can also fit slopes within each bin. We can do this by adding the original
# feature (the x-axis) back in, leading to an 11-dimensional dataset
#
# X_combined = np.hstack([X, X_binned])
# reg = LinearRegression().fit(X_combined, y)
#
# line_combined = np.hstack([line, line_binned])
# plt.plot(line, reg.predict(line_combined), label ='linear regression combined')
#
# for bin in bins:
#     plt.plot([bin, bin], [-3, 3], ':', c='k')
# plt.ylabel("Regression output")
# plt.xlabel("Input feature")
# plt.plot(X[:, 0], y, 'o', c='k')

# Slope is shared among the single x-axis feature which does not tell us much, we can add an
# interaction feature that indicates which bin a point is in, and where it lies on the
# x - axis, creating a 20 feature dataset.

# X_product = np.hstack([X_binned, X * X_binned])
#
# reg = LinearRegression().fit(X_product, y)
# line_product = np.hstack([line_binned, line * line_binned])
#
# plt.plot(line, reg.predict(line_product), label ='linear regression product')
#
# for bin in bins:
#     plt.plot([bin, bin], [-3, 3], ':', c='k')
# plt.ylabel("Regression output")
# plt.xlabel("Input feature")
# plt.plot(X[:, 0], y, 'o', c='k')

# continuing in this way, we can also add polynomial features. For a given X we may want
# to consider x ** 2, x ** 3, x ** 4, etc. This is baked into the PolynomialFeatures module

from sklearn.preprocessing import PolynomialFeatures

# want to include polynomials up tp x ** 10,
# the default "include_bias = True" adds a feature that's always 1
poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)

# Degree of 10 yields 10 features
# using together with linear regression, we can see polynomial regression

reg = LinearRegression().fit(X_poly, y)

line_poly = poly.transform(line)
plt.plot(line, reg.predict(line_poly), label ='Polynomial Linear Regression')

for bin in bins:
    plt.plot([bin, bin], [-3, 3], ':', c='k')

plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.plot(X[:, 0], y, 'o', c='k')











plt.show()














