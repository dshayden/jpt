import jpt
from . import proposals

import argparse
import numpy as np, scipy.linalg as sla
import du
from scipy.stats import poisson
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

  # set parameters
  eye, zer = ( np.eye(dy), np.zeros((dy, dy)) )
  param.dy = dy
  param.dx = dx
  param.ifile = ifile
  param.F = kwargs.get('F', np.block([[eye, eye], [zer, eye]]))
  param.H = kwargs.get('H', np.block([[eye, zer]]))
  param.lambda_track_length = kwargs.get('lambda_track_length', None)
  param.moveNames = ['update', 'split', 'merge', 'switch']
  param.moveProbs = np.array([0.25, 0.25, 0.25, 0.25])

  # todo: sampler scheduling stuff?
  
  # set priors
  ## x_{1k} ~ N(mu0, Sigma0)
  prior.mu0 = kwargs.get('mu0', np.zeros(dx))
  prior.Sigma0 = kwargs.get('Sigma0', np.block([[1e6*eye, zer], [zer, eye]]))

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
    if y.N[t1] == 0 or y.N[t2] == 0: continue
    nbrs = NearestNeighbors(n_neighbors=1).fit(y[t2])
    _, indices = nbrs.kneighbors(y[t1])
    for n in range(y.N[t1]):
      dists.append(np.abs(y[t1][n] - y[t2][indices[n]]))
  scatter = du.scatter_matrix( np.concatenate((dists)), center=False )

  S0 = sla.block_diag(.1*scatter, scatter)
  df0 = len(dists)
  return dict(mu0=mu0, Sigma0=Sigma0, df0=df0, S0=S0)

def init2d(ifile):
  dy, dx = (2, 4)
  y, z = jpt.io.mot15_point2d_to_assoc_unique(ifile)
  kwargs = data_dependent_priors(y)
  o = opts(ifile, dy, dx, **kwargs)

  # Shared initial parameters for all tracks
  Q = o.prior.S0 / (o.prior.df0 - dx - 1)
  R = 0.1 * Q[:dy,:dy]
  param = dict(Q=Q, R=R, mu0=o.prior.mu0)
  
  okf = jpt.kalman.opts(dy, dx, F=o.param.F, H=o.param.H, Q=Q, R=R)

  x = {}
  for k in z.ks:
    xtks = jpt.kalman.ffbs(okf,
      dict([(t, None) for t in z.to(k).keys()]), # requested latent
      dict([(t, y[t][j]) for t, j in z.to(k).items()]), # available obs
      x0=(o.prior.mu0, o.prior.Sigma0)) 
    x[int(k)] = ( param, xtks )
  w = jpt.AnyTracks(x)

  return o, y, w, z

def init2d_masks(ifile):
  dy, dx = (2, 4)
  data = jpt.io.load(ifile)
  yMasks, yMeans = ( data['yMasks'], data['yMeans'] )

  o = opts(ifile, dy, dx)
  okf = jpt.kalman.opts(dy, dx, F=o.param.F, H=o.param.H)
  param = dict(Q=okf.Q, R=okf.R, mu0=o.prior.mu0)
  w, z = init_assoc_greedy_dumb(yMeans, param)

  return o, yMasks, yMeans, w, z

def init_assoc_greedy_dumb(y, param):
  nTracks = max(y.N.values())

  # x is { k: {t: xtk, ...}, ... }
  # z is { t: (n_t,), ... }
  #   where (n_t,) is filled with indexes into k
  x, z = ( {}, {} )

  nInit = 0
  for t in y.ts:
    ns = np.arange(y.N[t]) # these are the actual indices j
    np.random.shuffle(ns) # these are the shuffled indices
    ns = list(ns)
    zt = np.zeros( len(ns), dtype=np.int )
    for idx, n in enumerate(ns):
      n = int(n)
      xtn = np.concatenate((y[t][n], np.zeros(len(y[t][n]))))
      if idx >= nInit: # initialize new track
        x[idx] = ( param, {t: xtn } )
        nInit += 1
      else:
        x[idx][1][t] = xtn
      zt[n] = int(idx)
    z[t] = list(zt)
  
  w = jpt.AnyTracks(x)
  z = jpt.UniqueBijectiveAssociation(y.N, z)

  return w, z


# def init_assoc_greedy_dumb(y):
#   nTracks = max(y.N.values())
#   
#   Tracks = [ ]
#   
#   z = { }
#   nInit = 0
#   for t in y.ts:
#     ns = np.arange(y.N[t]) # these are the actual indices j
#     np.random.shuffle(ns) # these are the shuffled indices
#     ns = list(ns)
#     zt = np.zeros( len(ns), dtype=np.int )
#     for idx, n in enumerate(ns):
#       n = int(n)
#       if idx >= nInit: # initialize new track
#         tau = Track({t: n})
#         nInit += 1
#         Tracks.append(tau)
#       else:
#         Tracks[idx][t] = n
#       zt[n] = int(idx)
#     z[t] = list(zt)
#
#   w = Hypothesis(y)
#   w.Tracks = Tracks
#   w.z = z
#   assert w.is_valid()
#   return w


def log_joint(o, y, w, z):
  ll = 0.0

  # track likelihoods
  for k in w.ks:
    theta, x_tks = w.x[k]
    y_tks = dict([(t, y[t][z.to(k)[t]]) for t in x_tks.keys()])
    okf = jpt.kalman.opts(o.param.dy, o.param.dx, Q=theta['Q'], R=theta['R'],
      F=o.param.F, H=o.param.H)
    x0=(o.prior.mu0, o.prior.Sigma0)
    ll += jpt.kalman.joint_ll(okf, x_tks, y_tks, x0)
  
  # track association lengths
  if o.param.lambda_track_length is not None:
    for k in w.ks:
      theta, x_tks = w.x[k]
      nAssoc = len(x_tks)
      ll += poisson.logpmf(nAssoc, o.param.lambda_track_length)

  return ll

# how to handle proposals with different arguments?
#   force all to have same inputs
#   lambda fcn to pass through relevant inputs
#   sample-specific logic for each move                 <---
def sample(o, y, w, z, ll):
  info = dict(valid=False, accept=False, move=None, logA=None, logq=None,
    ll_prop=None, ll_old=None)

  # sample move
  info['move'] = np.random.choice(o.param.moveNames, p=o.param.moveProbs)

  if info['move'] in ['switch', 'update']: doAcceptTest = False
  else: doAcceptTest = True

  if info['move'] == 'update':
    w_, z_, valid, logq = jpt.PointTracker.proposals.update(o, y, w, z)
  elif info['move'] == 'split':
    w_, z_, valid, logq = jpt.PointTracker.proposals.split(o, y, w, z)
  elif info['move'] == 'merge':
    w_, z_, valid, logq = jpt.PointTracker.proposals.merge(o, y, w, z)
  elif info['move'] == 'switch':
    w_, z_, valid, logq = jpt.PointTracker.proposals.switch(o, y, w, z)
  else:
    raise NotImplementedError(f"Don't support {move} proposal")

  if not valid: return w, z, info
  else: info['valid'] = True

  # Acceptance test and record info
  ll_new = log_joint(o, y, w_, z_)
  logA = ll_new - ll
  info.update(logA=logA, ll_prop=ll_new, ll_old=ll, logq=logq)
  if not doAcceptTest or logA > 0 or np.random.rand() <= np.exp(logA):
    info['accept'] = True
    return w_, z_, info
  else:
    return w, z, info
