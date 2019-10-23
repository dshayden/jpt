import jpt
import argparse
import numpy as np, scipy.linalg as sla
import du
from sklearn.neighbors import NearestNeighbors
import IPython as ip

def opts(ifile, dy, dx, **kwargs):
  """ Construct PointTracker options.

  INPUT
    ifile: file readable by io.mot15_point2d_to_assoc_unique
    dy, dx: observed and latent dimensions
    kwargs
      F: dynamics matrix
      H: projection matrix
      mu0: prior mean on target track location
      Sigma0: prior cov on target track location
      df0: prior dof on target dynamics noise covariance
      S0: prior scatter on target dynamics noise covariance

  OUTPUT
    o (Namespace): with fields prior, param
  """
  assert dy == 2, 'Only support 2d observations for the moment.'
  o = argparse.Namespace()

  prior, param = ( argparse.Namespace(), argparse.Namespace() )
  # todo: make an alg or run Namespace for I/O details, etc?

  # set parameters
  eye, zer = ( np.eye(dy), np.zeros((dy, dy)) )
  param.dy = dy
  param.dx = dx
  param.ifile = ifile
  param.F = kwargs.get('F', np.block([[eye, eye], [zer, eye]]))
  param.H = kwargs.get('H', np.block([[eye, zer]]))
  param.lambda_track_length = kwargs.get('lambda_track_length', None)
  # todo: sampler scheduling stuff?
  
  # set priors
  ## x_{1k} ~ N(mu0, Sigma0)
  prior.mu0 = kwargs.get('mu0', np.zeros(dx))
  prior.Sigma0 = kwargs.get('Sigma0', np.block([[1e6*eye, zer], [zer, eye]]))
  prior.H_mu0 = np.dot(param.H, prior.mu0)
  prior.H_Sigma0_H = np.dot(param.H, np.dot(prior.Sigma0, param.H.T))
  prior.H_Sigma0_H_inv = np.linalg.inv(prior.H_Sigma0_H)

  ## Q_k ~ IW(df0, S0)
  prior.df0 = kwargs.get('df0', 10)
  prior.S0 = kwargs.get('S0', 100*np.eye(dx))

  ## set prior and parameter, return
  o.prior = prior
  o.param = param
  return o

def data_dependent_priors(y):
  """ Make data-dependent priors for initial track location & dynamics noise cov

    mu0 is overall y mean
    Sigma0 is overall y covariance + large diagonal uncertainty in position
    df0 is # of data points in y
    S0 is mean per-dimension NN L1 distance of observations at adjacent times

  Assumes a random position + random acceleration model.

  INPUT
    y (NdObservationSet)

  OUTPUT
    D (dict): with keys mu0, Sigma0, df0, S0
  """
  dy = y[y.ts[0]].shape[1]

  # 1. collect mean, Sigma of all data to inform mu0
  ys = [ ]
  for t in y.ts: ys.append(y[t])
  ys = np.concatenate(ys)
  mu0 = np.concatenate((np.mean(ys,axis=0), np.zeros(dy)))
  Sigma0 = sla.block_diag( np.cov(ys.T), np.eye(dy) ) + \
    sla.block_diag(1e6*np.eye(dy), np.zeros((dy,dy)))

  # 2. collect per-dimension NN L1 distance of points at adjacent times to
  #    inform df0, S0
  dists = [ ]
  for t1, t2 in list(map(list, zip(y.ts, y.ts[1:]))):
    nbrs = NearestNeighbors(n_neighbors=1).fit(y[t1])
    _, indices = nbrs.kneighbors(y[t2])
    for n in range(y.N[t1]): dists.append(np.abs(y[t1][n] - y[t2][indices[n]]))
  scatter = du.scatter_matrix( np.concatenate((dists)), center=False )

  S0 = sla.block_diag(.1*scatter, scatter)
  df0 = len(dists)
  return dict(mu0=mu0, Sigma0=Sigma0, df0=df0, S0=S0)

def init2d(ifile):
  dy, dx = (2, 4)
  y, z = jpt.io.mot15_point2d_to_assoc_unique(ifile)
  kwargs = data_dependent_priors(y)
  o = opts(ifile, dy, dx, **kwargs)
  
  # sample latent states for each target 


  # todo: build latent states
  #   each track has its own Q, R

  return o, y, z

  # x = {}
  # for k in z.ks:
  #   x[int(k)] = jpt.kalman.ffbs(o.kf,
  #     dict([(t, None) for t in z.to(k).keys()]), # requested latent
  #     dict([(t, y[t][j]) for t, j in z.to(k).items()]) # available obs
  #   )
  # w = jpt.AnyTracks(x)
  # make hypothesis
