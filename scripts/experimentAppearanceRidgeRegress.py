import du, numpy as np, matplotlib.pyplot as plt
import sklearn.metrics
import IPython as ip, sys

# this is garbage

# get two targets, each a list of ndarrays of float pixel observations in 0..1
data = du.load('data/datasets/marmoset/color_nocolor')
color, no_color = ( data['color'], data['no_color'] )
nColor, nNoColor = ( len(color), len(no_color) )
nk = nColor + nNoColor # same for both targets
d = color[0].shape[1]

# set lambda parameter
lam = 0.0

X = np.concatenate((
  np.concatenate(color, axis=0),
  np.concatenate(no_color, axis=0)
  ), axis=0
)
X = np.concatenate((X, np.ones( (X.shape[0], 1) )), axis=1)

y = np.concatenate((
  np.concatenate([np.ones(color[i].shape[0]) for i in range(nColor)], axis=0),
  np.concatenate([-1*np.ones(no_color[i].shape[0]) for i in range(nNoColor)], axis=0)
  ), axis=0
)
w = np.linalg.inv(X.T @ X + lam * np.eye(d+1)) @ X.T @ y

y_bar = X @ w

fpr, tpr, thresh = sklearn.metrics.roc_curve(y, y_bar, pos_label=1)
auc = sklearn.metrics.roc_auc_score(y, y_bar)

plt.plot(fpr, tpr, label='per-pixel regression')
plt.plot(np.linspace(0, 1, 10), np.linspace(0,1,10), label='random', linestyle='--')
plt.title(f'Training AUC {auc:.2f}')
plt.legend()
plt.show()


# plt.scatter(range(y.shape[0]), y, label='y', s=0.1)
# plt.scatter(range(y.shape[0]), y_bar, label='y_bar', s=0.1)
# plt.legend()
# plt.show()

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
