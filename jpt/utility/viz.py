import numpy as np, matplotlib.pyplot as plt
import itertools
import du

def plot_tracks2d_global(w):
  """ Plot first two dims of all objects in PointHypothesis w. """
  colors = du.diffcolors(len(w.ks), alpha=0.5)
  for cind, k in enumerate(w.ks):
    it = sorted(w.x()[k].items())
    _, prevX = next(itertools.islice(it, 1))
    for t, xtk in it:
      xs, ys = ( (prevX[0], xtk[0]), (prevX[1], xtk[1]) )
      plt.plot(xs, ys, color=colors[cind])
      prevX = xtk

def plot_points2d_global(y): 
  """ Plot first two dims of all points in ObservationSet y. """
  color = 'k'
  for t in y.ts: plt.scatter(y[t][:,0], y[t][:,1], color=color, s=5, zorder=10)
  plt.gca().set_xticks([]); plt.gca().set_yticks([])
  plt.xlim( *limits(y, 0) )
  plt.ylim( *limits(y, 1) )
  plt.gca().set_aspect('equal', 'box')

def limits(y, i, expand=0.15):
  """ Get limits of PointObservationSet y along axis i with mild expansion. """
  vmin = np.min([np.min(y[t][:,i]) for t in y.ts])
  vmax = np.max([np.max(y[t][:,i]) for t in y.ts])
  delta = np.abs(vmax - vmin) * (expand/2.0)
  return vmin - delta, vmax + delta
