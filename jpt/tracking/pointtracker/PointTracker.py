import jpt
from . import proposals

import argparse
import numpy as np, scipy.linalg as sla
import du
from scipy.stats import poisson
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

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
  # hmm, random acceleration model damages switch statements
  # param.F = kwargs.get('F', np.block([[eye, eye], [zer, eye]]))
  param.F = kwargs.get('F', np.block([[eye, zer], [zer, zer]]))
  param.H = kwargs.get('H', np.block([[eye, zer]]))

  # fixed-K additions
  param.fixedK = kwargs.get('fixedK', -1) # -1 means no target limit
  param.maxK = kwargs.get('maxK', -1) # -1 means no target limit

  # parameters directly impacting noise assignments and # of tracks

  # |omega_k| ~ Pois(lambda_track_length) for all k > 0
  #   |omega_k|: length of target k's track
  #   lambda_track_length: expected track length
  param.lambda_track_length = kwargs.get('lambda_track_length', None)

  # d_t ~ Bin(n_t, p_d) for
  #   d_t: # detections at time t
  #   n_t: # observations at time t
  param.pd = kwargs.get('pd', 0.8) # probability of target detection

  # z_t ~ Bin(e_t, p_z) for
  #   z_t: # targets terminated at time t
  #   p_z: probability of target disappearance
  param.pz = kwargs.get('pz', 0.1)

  # f_t ~ Pois(lambda_f V) for
  #   f_t: # false alarms (i.e. # of noise associations) at time t
  #   lambda_f: false-alarm rate per unit time unit volume
  param.lambda_f = kwargs.get('lambda_f', 0.1) 

  # a_t ~ Pois(lambda_b V) for
  #   a_t: # new targets at time t
  #   lambda_b: new target birth rate per unit time unit volume
  param.lambda_b = kwargs.get('lambda_b', 0.01) 

  # gather proposal skip probability
  param.ps = kwargs.get('ps', 0.01) 

  # log annotation agreement probability
  #   stored as log because we'll underflow around log(1e-300) == -690
  #   and we might want very strong penalties for inconsistency
  param.log_pa = kwargs.get('log_pa', -1000)

  # generic sampler scheduling
  param.moveNames = ['update', 'switch', 'gather', 'disperse', 'extend', 'split', 'merge']
  param.moveProbs = np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1])

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
  # expects ifile to be in mot15 format
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

def init2d_masks(ifile, **kwargs):
  # input is list of mask-based detections, will slowly compute mean statistics
  # will take ~4 mins for ~3k masks
  dy, dx = (2, 4)
  yMasks = jpt.io.masks_to_obs(ifile)
  yMeans = jpt.io.mean_from_masks(yMasks)

  o = opts(ifile, dy, dx, **kwargs)
  okf = jpt.kalman.opts(dy, dx, F=o.param.F, H=o.param.H)
  param = dict(Q=okf.Q, R=okf.R, mu0=o.prior.mu0)
  w, z = init_assoc_greedy_dumb(yMeans, param, maxK=o.param.maxK)

  return o, yMasks, yMeans, w, z

def init2d_masks_precomputed_mean(ifile, **kwargs):
  # expects JPT-serialized dictionary with keys yMasks, yMeans
  dy, dx = (2, 4)
  data = jpt.io.load(ifile)
  yMasks, yMeans = ( data['yMasks'], data['yMeans'] )

  o = opts(ifile, dy, dx, **kwargs)
  okf = jpt.kalman.opts(dy, dx, F=o.param.F, H=o.param.H)
  param = dict(Q=okf.Q, R=okf.R, mu0=o.prior.mu0)
  w, z = init_assoc_greedy_dumb(yMeans, param, maxK=o.param.maxK)

  return o, yMasks, yMeans, w, z

def init_assoc_noise(y):
  x, z = ( {}, {} )
  for t in y.ts:
    ns = np.arange(y.N[t])
    z[t] = list(np.zeros(len(ns), dtype=np.int))

  w = jpt.AnyTracks(x)
  z = jpt.UniqueBijectiveAssociation(y.N, z)

  return w, z

# def init_assoc_random_fixedK(y, fixedK):
#   x, z = ( {}, {} )
#
#   # randomly assign labels
#   for t in y.ts:
#     zt = np.zeros(y.N[t])
#     curK = 1
#     for n in range(y.N[t]):
#       zt[n] = curK
#       curK += 1
#       if curK > fixedK: break
#     np.random.shuffle(zt)
#     z[t] = list(zt)
#
#     for n in range(y.N[t]):
#       k = zt[n]
#       if k == 0: continue
#       xtn = np.concatenate((y[t][k], np.zeros(len(y[t][n]))))
#
#
#   w = jpt.AnyTracks(x)
#   z = jpt.UniqueBijectiveAssociation(y.N, z)
#
#   return w, z


def init_assoc_greedy_dumb(y, param, maxK=-1):
  nTracks = max(y.N.values())

  # x is { k: {t: xtk, ...}, ... }
  # z is { t: (n_t,), ... }
  #   where (n_t,) is filled with indexes into k
  x, z = ( {}, {} )

  nInit = 1
  for t in y.ts:
    ns = np.arange(y.N[t]) # these are the actual indices j
    np.random.shuffle(ns) # these are the shuffled indices
    ns = list(ns)
    zt = np.zeros( len(ns), dtype=np.int )
    for idx, n in enumerate(ns):

      # noise associations
      if maxK != -1 and idx+1 > maxK:
        zt[n] = 0
        continue

      # target associations
      n = int(n)
      xtn = np.concatenate((y[t][n], np.zeros(len(y[t][n]))))

      # idx : 0..n-1
      # nInit : 1, 2, ...
      # x, z dictionaries
      if idx >= nInit-1: # initialize new track
        x[idx+1] = ( param, {t: xtn } )
        nInit += 1
      else:
        x[idx+1][1][t] = xtn
      zt[n] = int(idx+1)
    z[t] = list(zt)
  
  w = jpt.AnyTracks(x)
  z = jpt.UniqueBijectiveAssociation(y.N, z)

  return w, z


def log_joint(o, y, w, z, **kwargs):
  ll = 0.0
  
  # track likelihoods
  for k in w.ks:
    theta, x_tks = w.x[k]
    y_tks = dict([(t, y[t][z.to(k)[t]]) for t in x_tks.keys()])
    okf = jpt.kalman.opts(o.param.dy, o.param.dx, Q=theta['Q'], R=theta['R'],
      F=o.param.F, H=o.param.H)
    x0=(o.prior.mu0, o.prior.Sigma0)
    ll += jpt.kalman.joint_ll(okf, x_tks, y_tks, x0)


  ### comment everything else out during testing
  # # noise observations
  nNoise = np.sum( [ len(v) for v in z.to(0).values() ] )
  ll += -10 * nNoise


  n_t = y.N #observations
  d_t = { t: 0 for t in y.ts } #detections
  f_t = { t: 0 for t in y.ts } #false alarms

  for t in y.ts:
    d_t[t] = np.sum(z[t] > 0)
    f_t[t] = np.sum(z[t] == 0)

  z_t = { t: 0 for t in y.ts } #track terminations
  for k in w.ks: z_t[ max(w.x[k][1].keys()) ] += 1

  e_t1 = { t: 0 for t in y.ts } #existing targets
  a_t = { t: 0 for t in y.ts } #new targets
  for k in z.ks:
    minTime = min(z.to(k).keys())
    maxTime = max(z.to(k).keys())

    # new targets at time t
    a_t[minTime] += 1

    # targets at time t
    # for t in range(minTime, maxTime+1): e_t1[t] += 1
    for t in range(minTime, maxTime+1):
      if t in e_t1: e_t1[t] += 1

  logPz = np.log(o.param.pz)
  log1Pz = np.log(1 - o.param.pz)
  logPd = np.log(o.param.pd)
  log1Pd = np.log(1 - o.param.pd)
  logLambda_b = np.log(o.param.lambda_b)
  logLambda_f = np.log(o.param.lambda_f)
  for t in y.ts:
    if t-1 in e_t1: et = e_t1[t-1]
    else: et = 0
    ct = et - z_t[t]
    ut = ct + a_t[t] - d_t[t]

    # # test: remove counts
    ll += z_t[t] * logPz + ct * log1Pz
    ll += d_t[t] * logPd + ut * log1Pd
    ll += a_t[t] * logLambda_b
    ll += f_t[t] * logLambda_f

  # Annotations, if any. 
  #   Note: don't need to explicitly add
  #     ll += consistent * log(1 - p_a)
  #   because log(1 - p_a) ~= 0 since we assume p_a = 1 - eps
  A = kwargs.get('A_pairwise', None)
  if A is None: return ll
  consistent, inconsistent = A.consistencyCounts(z)
  ll += inconsistent * o.param.log_pa

  return ll

# how to handle proposals with different arguments?
#   force all to have same inputs
#   lambda fcn to pass through relevant inputs
#   sample-specific logic for each move                 <---
def sample(o, y, w, z, ll, **kwargs):
  A_pairwise = kwargs.get('A_pairwise', None)

  info = dict(valid=False, accept=False, move=None, logA=None, logq=None,
    ll_prop=None, ll_old=None)

  # sample move
  info['move'] = np.random.choice(o.param.moveNames, p=o.param.moveProbs)
  logq = 0.0

  if info['move'] in ['switch', 'update']: doAcceptTest = False
  # if info['move'] in ['update',]: doAcceptTest = False
  else: doAcceptTest = True

  # print(f"Trying {info['move']}")

  if info['move'] == 'update':
    w_, z_, valid, logq = jpt.PointTracker.proposals.update(o, y, w, z)
  elif info['move'] == 'split':
    w_, z_, valid, logq = jpt.PointTracker.proposals.split(o, y, w, z)
  elif info['move'] == 'merge':
    w_, z_, valid, logq = jpt.PointTracker.proposals.merge(o, y, w, z)
  elif info['move'] == 'switch':
    w_, z_, valid, logq = jpt.PointTracker.proposals.switch(o, y, w, z)
  elif info['move'] == 'gather':
    w_, z_, valid, logq = jpt.PointTracker.proposals.gather(o, y, w, z)
  elif info['move'] == 'disperse':
    w_, z_, valid, logq = jpt.PointTracker.proposals.disperse(o, y, w, z)
  elif info['move'] == 'extend':
    w_, z_, valid, logq = jpt.PointTracker.proposals.extend(o, y, w, z)
  else:
    raise NotImplementedError(f"Don't support {move} proposal")

  if not valid: return w, z, info
  else: info['valid'] = True

  # Acceptance test and record info
  ll_new = log_joint(o, y, w_, z_, A_pairwise=A_pairwise)
  logA = ll_new - ll + logq
  info.update(logA=logA, ll_prop=ll_new, ll_old=ll, logq=logq)
  if not doAcceptTest or logA > 0 or np.random.rand() <= np.exp(logA):
    info['accept'] = True
    return w_, z_, info
  else:
    # if info['move'] == 'switch':
    #   print(f'Rejected Switch: ll_prop: {ll_new:.2f}, ll_old: {ll:.2f}, logq: {logq:.2f}, logA: {logA:.2f}')

    return w, z, info

def stlc_cost_matrix(p, q, lam):
  # For AnyTracks p, q, compute cost matrix of size (K, K') where
  #   p contains K tracks
  #   q contains K' tracks
  # Costs are computed as the Spatio-Temporal Linear Combine Distance
  # with combine factor lam. See:
  #   S. Shang, L. Chen, Z. Wei, C. S. Jensen, K. Zheng, and P. Kalnis.
  #   Trajectory similarity join in spatial networks. Proceedings of the VLDB
  #   Endowment, 10(11):1178–1189, 2017.

  nTracks = [ len(tau.ks) for tau in (p, q) ]
  size = [ float(len(tau.ts)) for tau in (p, q) ]

  identify_track(p)
  identify_track(q)

  def dist_spatial(ps, T):
    # ps is D-dim vector
    # T is single trajectory
    Tx = np.array([ T[t] for t in T.keys() ]) # get all latent states

    # NOTE: hard-coded for 2D right here
    dists = cdist( ps[:2][np.newaxis], Tx[:,:2] ).squeeze()
    return np.min(dists)

  def dist_spatial_debug(ps, T):
    # ps is D-dim vector
    # T is single trajectory
    Tx = np.array([ T[t] for t in T.keys() ]) # get all latent states

    # NOTE: hard-coded for 2D right here
    dists = cdist( ps[:2][np.newaxis], Tx[:,:2] ).squeeze()
    return np.min(dists), np.argmin(dists)

  def dist_temporal(pt, T):
    # pt is float time
    # T is single trajectory
    Tt = np.array([ t for t in T.keys() ]) # get all latent states
    dists = np.abs( Tt - pt )
    # dists = cdist( np.array([pt])[np.newaxis], Tt ).squeeze()
    return np.min(dists)

  cost = np.zeros((nTracks[0], nTracks[1]))
  for i in range(nTracks[0]):
    for j in range(nTracks[1]):
      k1, k2 = ( p.ks[i], q.ks[j] )

      T1, T2 = ( p.x[k1][1], q.x[k2][1] )
      sij1 = sij2 = tij1 = tij2 = 0.0

      # spatial terms
      for t in T1.keys(): sij1 += np.exp(-dist_spatial(T1[t], T2))
      for t in T2.keys(): sij2 += np.exp(-dist_spatial(T2[t], T1))

      # temporal terms
      for t in T1.keys(): tij1 += np.exp(-dist_temporal(t, T2))
      for t in T2.keys(): tij2 += np.exp(-dist_temporal(t, T1))


      s_spa = (sij1 / size[0]) + (sij2 / size[1])
      s_tem = (tij1 / size[0]) + (tij2 / size[1])

      assert 0 <= s_spa and s_spa <= 2.0
      assert 0 <= s_tem and s_tem <= 2.0
      cost[i,j] = - ( lam*s_spa + (1-lam)*s_tem )

  #     print(i,j,k1,k2,cost[i,j])
  #     if i==0:
  #       print(f'i={i}, j={j}\n  T1: {T1[1][:2]}\n  T2: {T2[1][:2]}\n  cost: {cost[i,j]:.2f}')
  #
  #       sdists1, sidx1 = zip(*[ dist_spatial_debug(T1[t], T2) for t in T1.keys() ])
  #       sdists2, sidx2 = zip(*[ dist_spatial_debug(T2[t], T1) for t in T2.keys() ])
  #
  #       print('sidx1')
  #       for tIdx, t_ in enumerate(T1.keys()):
  #         T2Keys = list(T2.keys())
  #         t2_ = T2Keys[sidx1[tIdx]]
  #
  #         print(f'p_{i}[{t_}] -> q_{j}[{t2_}], dist: {sdists1[tIdx]:.2f}')
  #         print(T1[t_][:2])
  #         print(T2[t2_])
  #
  #       # print(i,j, 'sidx1')
  #       # print(sidx1)
  #       #
  #       # print('T1')
  #       # print(T1)
  #       #
  #       # print('T2')
  #       # print(T2)
  #
  #       # for t in T1.keys(): sij1 += np.exp(-dist_spatial(T1[t], T2))
  #       # for t in T2.keys(): sij2 += np.exp(-dist_spatial(T2[t], T1))
  #       # if j==2: ip.embed()
  #
  #
  # # testing
  # from scipy.optimize import linear_sum_assignment as lsa
  # import matplotlib.pyplot as plt
  # from matplotlib.lines import Line2D
  #
  # plt.subplot(2,1,1)
  # jpt.viz.plot_tracks2d_global(p)
  # colors = du.diffcolors(len(p.ks), alpha=0.5)
  # # custom_lines = [ Line2D([0],[0],color=colors[k]) for k in range(len(p.ks)) ]
  # custom_lines = reversed([Line2D([0],[0],color=colors[k])
  #   for k in range(len(q.ks))])
  # custom_strings = [ f'{k}' for k in range(len(p.ks)) ]
  # plt.legend(custom_lines, custom_strings)
  # plt.title('p')
  # plt.xticks([]); plt.yticks([])
  #
  # plt.subplot(2,1,2)
  # jpt.viz.plot_tracks2d_global(q)
  # colors = du.diffcolors(len(q.ks), alpha=0.5)
  # custom_lines = reversed([Line2D([0],[0],color=colors[k])
  #   for k in range(len(q.ks))])
  # custom_strings = [ f'{k}' for k in range(len(q.ks)) ]
  # plt.legend(custom_lines, custom_strings)
  # plt.title('q')
  # plt.xticks([]); plt.yticks([])
  #
  # plt.figure()
  # plt.imshow(cost)
  # plt.xlabel('q'); plt.ylabel('p')
  # # plt.xlabel('p'); plt.ylabel('q') # intentionally bad ordering to diagnose
  # plt.xticks(range(len(q.ks)))
  # plt.yticks(range(len(p.ks)))
  # plt.colorbar()
  #
  # rows, cols = lsa(cost)
  # for r, c in zip(rows, cols):
  #   plt.scatter(r, c, s=1.0, c='k')
  #
  # plt.show()
  # sys.exit()

  return cost

def identify_track(w):
  # order AnyTracks w based on y-position of first location of each track
  ks = np.array(w.ks)
  if len(ks) <= 1: return
  kt = { k: min(w.x[k][1].keys()) for k in w.ks }
  ky = [ w.x[k][1][kt[k]][1] for k in w.ks ]
  new_ks = [ int(v) for v in ks[np.argsort(ky)] ]
  w.ks = new_ks
