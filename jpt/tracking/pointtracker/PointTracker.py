import jpt
import argparse
import numpy as np
import IPython as ip

# PointTracker options: kalman, 
def opts(ifile, dy=2, **kwargs):
  assert dy == 2, 'Only support 2d observations for the moment.'
  o = argparse.Namespace()
  o.kf = jpt.kalman.opts(dy, 2*dy, **kwargs)
  o.ifile = ifile
  return o

def init(o, **kwargs):
  y, z = jpt.io.mot15_point2d_to_assoc_unique(o.ifile)

  x = {}
  for k in z.ks:
    x[int(k)] = jpt.kalman.ffbs(o.kf,
      dict([(t, None) for t in z.to(k).keys()]), # requested latent
      dict([(t, y[t][j]) for t, j in z.to(k).items()]) # available obs
    )
  w = jpt.PointHypothesis(x)

  return y, w, z
