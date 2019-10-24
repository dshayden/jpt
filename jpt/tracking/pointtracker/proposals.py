import jpt
from scipy.spatial.distance import mahalanobis
import numpy as np
import IPython as ip

def switch(o, y, w, z):
  windows = possible_switch_windows(o, z, reverse=False)
  if len(windows) == 0: return w, z, False, 0.0
  switchIdx = np.random.choice(len(windows))
  ks, t0, ts = windows[switchIdx]
  x0 = []
  for idx, k in enumerate(ks): x0.append( w.x[k][1][t0[idx]] )

  Qi = dict([(k, np.linalg.inv( w.x[k][0]['Q'])) for k in ks])
  Ri = dict([(k, np.linalg.inv( w.x[k][0]['R'])) for k in ks])
  kf = dict([(k, jpt.kalman.opts(o.param.dy, o.param.dx,
    F=o.param.F, H=o.param.H, Q=w.x[k][0]['Q'], R=w.x[k][0]['R'] ))
    for k in ks])

  def costxx(t1, x1, t2, x2, k):
    # todo: handle separated t1, t2
    return -0.5 * mahalanobis(x1, np.dot(o.param.F, x2), Qi[k])

  def costxy(t, k, xtk, ytk):
    return -0.5 * mahalanobis(ytk, np.dot(o.param.H, xtk), Ri[k])

  def val(t,k):
    if t in w.x[k][1]:
      jtk = z.to(k)[t]
      xtk, ytk = ( w.x[k][1][t], y[t][jtk] )
    else:
      # todo: handle reverse direction
      dct = w.x[k][1].copy()
      dct[t] = None
      muX, SigmaX = jpt.kalman.state_estimate(kf[k], dct)[t]
      muY, SigmaY = jpt.kalman.StateToObs(kf[k], muX, SigmaX)
      xtk, ytk, jtk = (muX, muY, None)
    return xtk, ytk, jtk

  # make hmm, sample
  perms, pi, Psi, psi = jpt.hmm.build(t0, x0, ts, ks, val, costxx, costxy)
  S = jpt.hmm.ffbs(pi, Psi, psi)

  # edit w, z
  ## S indexes into perms; perms is wrt ks
  editDict_tkj = {}
  editDict_k = dict([(k, (w.x[k][0], {})) for k in ks])
  for idx, t in enumerate(ts):
    t = int(t)
    p = perms[S[idx]]
    for kOld_idx, kOld in enumerate(ks):
      kNew = ks[p[kOld_idx]]
      xtk, ytk, jtk = val(t, kNew)
      if jtk is None: editDict_k[kOld][1][t] = None; continue
      editDict_tkj[(t, kOld)] = jtk
      editDict_k[kOld][1][t] = xtk
  z_ = z.edit(editDict_tkj, kind='tkj', inplace=False)
  w_ = w.edit(editDict_k, kind='k', inplace=False)

  return w_, z_, True, 0.0

def split(o, y, w, z):
  """ Sample a random split proposal for UniqueBijectiveAssociation, z. """
  splits = possible_splits(z)
  if len(splits) == 0: return w, z, False, 0.0
  k1, t_ = splits[ np.random.choice(len(splits)) ]

  # split track k at time t inclusive: all associations <= t_ stay with to k1
  ## edit z
  k2 = z.next_k()
  editDict_tkj = { (t, k2): j for t, j in z.to(k1).items() if t <= t_ }
  z_ = z.edit(editDict_tkj, kind='tkj', inplace=False)

  ## edit w
  editDict_k = {
    int(k1): (
      w.x[k1][0],
      dict([(t, None) for t in z.to(k1).keys() if t <= t_])
    ),
    int(k2): (
      w.x[k1][0],
      dict([(t, w.x[k1][1][t]) for t in z.to(k1).keys() if t <= t_])
    )
  }
  w_ = w.edit(editDict_k, kind='k', inplace=False)

  # get move log probability
  q_new_given_old = -np.log(len(splits))
  merges = possible_merges(z_)
  assert len(merges) > 0
  q_old_given_new = -np.log(len(merges))
  return w_, z_, True, q_old_given_new - q_new_given_old

def merge(o, y, w, z):
  """ Sample a random merge proposal for UniqueBijectiveAssociation, z. """
  merges = possible_merges(z)
  if len(merges) == 0: return w, z, False, 0.0
  k1, k2 = merges[ np.random.choice(len(merges)) ]
  k_keep, k_remove = ( min(k1, k2), max(k1, k2) )

  # assign everything from k_remove to k_keep
  editDict_tkj = { (t, k_keep): j for t, j in z.to(k_remove).items() }
  z_ = z.edit(editDict_tkj, kind='tkj', inplace=False)

  # edit w
  editDict_k = {
    int(k_remove): None,
    int(k_keep): (
      w.x[k_keep][0],
      dict([(t, w.x[k_remove][1][t]) for t in z.to(k_remove).keys() ])
    )
  }
  w_ = w.edit(editDict_k, kind='k', inplace=False)

  # get move log probability
  splits = possible_splits(z_)
  assert len(splits) > 0
  q_old_given_new = -np.log(len(splits))
  q_new_given_old = -np.log(len(merges))

  return w_, z_, True, q_old_given_new - q_new_given_old

def possible_merges(z):
  """ Construct all possible merges of UniqueBijectiveAssociation, z.
    
    Valid merges are all unique (k1, k2) which share no association at any time.
  """
  merges = []
  for k1 in z.ks:
    for k2 in z.ks:
      if k1 == k2: continue
      k1_ts = list(z.to(k1).keys())
      k2_ts = list(z.to(k2).keys())
      if len(np.intersect1d(k1_ts, k2_ts)) > 0: continue
      merges.append( (k1, k2) )
  return merges

def possible_splits(z):
  """ Construct all possible splits of UniqueBijectiveAssociation, z.
    
    Valid splits are all (k, t) for which target k has an association at time t
    and at least one association at some time t' > t.
  """
  splits = []
  for k in z.ks:
    for t in list(z.to(k).keys())[:-1]: splits.append( (k, t) )
  return splits

def possible_switch_windows(o, z, reverse=False):
  """ Get possible switch windows for two targets.

  Return is of the form [ (ks, t0, tsInside), ... ] for
    targets k in ks
    most recent per-target times t0 *outside* swap (corresponds to order of ks)
    swap window times tsInside
  """
  windows = [ ]
  for k1 in z.ks:
    t1 = list(z.to(k1).keys())
    if len(t1) < 2: continue

    for k2 in z.ks:
      if k1 == k2: continue
      t2 = list(z.to(k2).keys())
      if len(t2) < 2: continue

      ts = np.sort(np.unique(t1 + t2))
      if reverse: ts = ts[::-1]

      # find smallest/largest time t which, "before" t, both have >= one obs
      for t_ in ts:
        if reverse:
          t1_before = [ t for t in t1 if t <= t_ ]
          t2_before = [ t for t in t2 if t <= t_ ]
          t1_after = [ t for t in t1 if t > t_ ]
          t2_after = [ t for t in t2 if t > t_ ]
          if len(t1_after) == 0 or len(t2_after) == 0: continue
          if len(t1_before) == 0 and len(t2_before) == 0: continue
          t0 = None # TODO: implement

        else:
          t1_before = [ t for t in t1 if t < t_ ]
          t2_before = [ t for t in t2 if t < t_ ]
          t1_after = [ t for t in t1 if t >= t_ ]
          t2_after = [ t for t in t2 if t >= t_ ]
          if len(t1_before) == 0 or len(t2_before) == 0: continue
          if len(t1_after) == 0 and len(t2_after) == 0: continue
          t0 = np.array([max(t1_before), max(t2_before)])
          tsInside = [ t for t in ts if t >= t_ ]
        
        windows.append( ( (k1, k2), t0, tsInside) )
        # windows.append( (k1, k2, t_) )
        return windows
  return []
