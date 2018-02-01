import sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)

X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)

bins = np.bincount(X[:, 0])
# plt.bar(range(len(bins)), bins, color='w')
# plt.ylabel("Number of appearances")
# plt.xlabel("Value")

# lets fit a ridge regression
from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

score = Ridge().fit(X_train, y_train).score(X_test, y_test)
print 'test score:', score

# low R^2... ridge no good for capturing relationship between X and y, lets try
# and apple a log transformation

X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)

y_train_log = np.log(y_train + 1)
y_test_log = np.log(y_test + 1)



plt.hist(np.log(X_train_log[:, 0]+1), bins = 25, color = 'gray')
plt.ylabel('# of appearances')
plt.xlabel('value')


score = Ridge().fit(X_train_log, y_train_log).score(X_test_log, y_test_log)
print 'log test score:', score

# we see a much better R^2




