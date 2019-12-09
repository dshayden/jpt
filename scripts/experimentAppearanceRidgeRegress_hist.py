import du, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut 
import IPython as ip, sys

# this is garbage

# get two targets, each a list of ndarrays of float pixel observations in 0..1
data = du.load('data/datasets/marmoset/color_nocolor')
color, no_color = ( data['color'], data['no_color'] )
nColor, nNoColor = ( len(color), len(no_color) )
nk = nColor + nNoColor # same for both targets

# change color, no_color to histograms
nBins = 4
d = nBins ** 3
rng = [ (0., 1.), (0., 1.), (0., 1.) ]
color = [ np.histogramdd(c, bins=nBins, range=rng)[0].flatten() for c in color ]
no_color = [ np.histogramdd(c, bins=nBins, range=rng)[0].flatten() for c in no_color ]

# plt.figure()
# for c in color: plt.scatter(range(int(nBins**3)), c, s=0.1, color='g')
# plt.figure()
# for c in no_color: plt.scatter(range(int(nBins**3)), c, s=0.1, color='b')
# plt.show()

# set lambda parameter
lam = 0.1

X = np.concatenate((
  np.stack(color),
  np.stack(no_color))
)
X = np.concatenate((X, np.ones( (X.shape[0], 1) )), axis=1) # add column of 1s

y = np.concatenate((np.ones(nColor), -1 * np.ones(nNoColor)))

# fit
w = np.linalg.inv(X.T @ X + lam * np.eye(d+1)) @ X.T @ y

# predict
y_bar = X @ w

# plot
# plt.scatter(range(y.shape[0]), y, label='y', s=3.0, alpha=0.5)
# plt.scatter(range(y.shape[0]), y_bar, label='y_bar', s=5.0, alpha=0.5)
# plt.legend()
# plt.show()

loo = LeaveOneOut()
loo.get_n_splits(X)
acc = 0

cnt = 0.0
for train_index, test_index in loo.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]

  w = np.linalg.inv(X_train.T @ X_train + lam * np.eye(d+1)) @ X_train.T @ y_train

  # predict on test, check accuracy
  y_bar = X_test @ w

  y_bar[y_bar >= 0] = 1.0
  y_bar[y_bar < 0] = -1.0

  acc += np.sum(y_bar == y_test)
  cnt += 1.0

acc /= cnt
print(f'LOOCV Mean Accuracy: {acc}')










# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(random_state=0).fit(X, y)
#
# c = lr.predict(X)
# p = lr.predict_proba(X)
#
# y[y==-1] = 0
# plt.scatter(range(y.shape[0]), y, label='y', s=0.1, alpha=0.5)
# plt.scatter(range(y.shape[0]), p[:,0], label='p', s=0.1, alpha=0.5)
# plt.legend()
# plt.show()
