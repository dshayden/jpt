import numpy as np, matplotlib.pyplot as plt
import itertools
import du
import skimage.draw as draw

def plot_associated_masks_on_images(yImgs, yMasks, z, outDir):
  colors = du.diffcolors(len(z.ks)+1, alpha=0.5)
  colors[0] = [0.5, 0.5, 0.5, 0.5]

  for idx, t in enumerate(yImgs.ts):
    im = yImgs[t][0]
    mt, st = yMasks[t]
    for j in range(z.N[t]):
      mtj = mt[j]
      color = colors[ z[t][j] ]
      im = du.DrawOnImage(im, np.where(mtj > 0), color)
    du.imwrite(im, f'{outDir}/img-{idx:08d}.jpg')

def plot_masks(y, yMeans=None):
  """ Plot MaskObservationSets """
  colors = du.diffcolors(max(y.N.keys()), alpha=0.5)
  def show(t):
    mt, st = y[t]
    if y.N[t] == 0: return
    im = np.zeros(mt.shape[1:] + (3,), dtype=np.uint8)
    for mtk, cind in zip(mt, range(y.N[t])):
      im = du.DrawOnImage(im, np.where(mtk > 0), colors[cind])
    if yMeans is not None:
      for mean_tk, cind in zip(yMeans[t], range(yMeans.N[t])):
        coords = draw.circle(mean_tk[1], mean_tk[0], 40, shape=im.shape)
        im = du.DrawOnImage(im, coords, colors[cind])

    plt.title(t)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(im)
    # if len(st) > 0: plt.imshow(np.sum(mt, axis=0).astype(np.bool))
  du.ViewPlots(y.ts, show)
  plt.show()

def plot_tracks2d_global(w):
  """ Plot first two dims of all objects in AnyTracks w. """
  colors = du.diffcolors(len(w.ks), alpha=0.5)
  for cind, k in enumerate(w.ks):
    it = sorted(w.x[k][1].items())
    _, prevX = next(itertools.islice(it, 1))
    for t, xtk in it:
      xs, ys = ( (prevX[0], xtk[0]), (prevX[1], xtk[1]) )
      plt.plot(xs, ys, color=colors[cind])
      prevX = xtk
  plt.gca().set_aspect('equal', 'box')

def plot_points2d_global(y):
  """ Plot first two dims of all points in ObservationSet y. """
  color = 'k'
  for t in y.ts: plt.scatter(y[t][:,0], y[t][:,1], color=color, s=5, zorder=10)
  plt.gca().set_xticks([]); plt.gca().set_yticks([])
  plt.xlim( *limits(y, 0) )
  plt.ylim( *limits(y, 1) )
  plt.gca().set_aspect('equal', 'box')

def limits(y, i, expand=0.15):
  """ Get limits of NdObservationSet y along axis i with mild expansion. """
  vmin = np.min([np.min(y[t][:,i]) for t in y.ts])
  vmax = np.max([np.max(y[t][:,i]) for t in y.ts])
  delta = np.abs(vmax - vmin) * (expand/2.0)
  return vmin - delta, vmax + delta
