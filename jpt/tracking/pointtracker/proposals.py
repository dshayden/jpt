import jpt
from scipy.spatial.distance import mahalanobis, cdist
from scipy.special import logsumexp
from scipy.stats import poisson
import numpy as np, itertools
import copy
import IPython as ip

def extend(o, y, w, z):
  # check that we have tracks to disperse
  if not z.ks: return w, z, False, 0.0

  z0 = z.to(0)

  # randomly sample an existing track
  k = np.random.choice(z.ks)
  zk = z.to(k)
  if len(zk.keys()) == 1 or len(z0.keys()) == 0: return w, z, False, 0.0

  # randomly sample a direction
  # d = np.random.choice([-1, 1])
  d = 1 # don't allow backwards extend for the moment

  # arrange times in ascending (d=1) or descending (d=-1) order, starting from
  # first (d=1) or last (d=-1) track association time
  fixedObs_t = min(zk.keys()) if d==1 else max(zk.keys())
  t0 = [ t for t in z0.keys() if d*t > d*fixedObs_t ]
  tk = [ t for t in zk.keys() if d*t > d*fixedObs_t ]
  ts = sorted(list(np.unique(t0 + tk)))
  if d == -1: ts = ts[::-1]
  t = fixedObs_t

  # construct two filters, one for old -> new (o2n) and one for new -> old (n2o)
  # initialize them on association at fixedObs_t
  okf = jpt.kalman.opts(o.param.dy, o.param.dx, F=o.param.F, H=o.param.H)
  ytn = y[t][zk[fixedObs_t]]
  lastT = fixedObs_t

  muX_o2n, SigmaX_o2n = jpt.kalman.predict(okf, o.prior.mu0, o.prior.Sigma0)
  muX_o2n, SigmaX_o2n, _ = jpt.kalman.filter(okf, muX_o2n, SigmaX_o2n, ytn)
  muX_n2o, SigmaX_n2o = ( muX_o2n.copy(), SigmaX_o2n.copy() )

  # precompute and initialize
  log_ps = np.log(o.param.ps)
  log_1ps = np.log(1 - o.param.ps)
  q_old_given_new = q_new_given_old = 0.0

  # old_given_new: recovering the existing track, no sampling
  # new_given_old: choosing new track, sampling involved

  # just sample new track in this loop, already complicated enough
  editDict_tkj = {}
  for t in ts:
    # skip probability
    if np.random.rand() < o.param.ps:
      q_new_given_old += log_ps
      if t in zk.keys():
        # add zk[t] to noise
        if t in z0.keys() and t in zk.keys():
          noiseObs = sorted([int(j) for j in z0[t] + [zk[t],]])
        elif t in z0.keys() and t not in zk.keys():
          noiseObs = z0[t]
        elif t in zk.keys() and t not in z0.keys():
          noiseObs = [ int(zk[t]), ]

        # noiseObs = sorted([int(j) for j in z0[t] + [zk[t],]])

        editDict_tkj[(t,0)] = noiseObs
      continue
    q_new_given_old += log_1ps

    # predict step
    for t_ in range(np.abs(lastT - t)):
      muX_o2n, SigmaX_o2n = jpt.kalman.predict(okf, muX_o2n, SigmaX_o2n)
    muY_o2n, SigmaY_o2n = jpt.kalman.StateToObs(okf, muX_o2n, SigmaX_o2n)
    SigmaYi_o2n = np.linalg.inv(SigmaY_o2n)
    lastT = t

    # compute track-obs distances
    if t in z0.keys() and t in zk.keys():
      obs_t = np.concatenate((z0[t], [zk[t], ] ))
    elif t in z0.keys() and t not in zk.keys():
      obs_t = z0[t]
    elif t in zk.keys() and t not in z0.keys():
      obs_t = [ zk[t], ]

    # if t in zk.keys(): obs_t = np.concatenate((z0[t], [zk[t], ] ))
    # else: obs_t = z0[t]

    logpObs = np.zeros(len(obs_t))
    for idx, j in enumerate(obs_t):
      # track <-> obs distance to obs
      ytn = y[t][j]
      logpObs[idx] = -0.5 * mahalanobis(ytn, muY_o2n, SigmaYi_o2n)

    pObs = np.exp(logpObs - logsumexp(logpObs))
    jIdx = np.random.choice(np.arange(len(pObs)), p=pObs)
    if t in zk.keys() and zk[t] != obs_t[jIdx]:
      # adjust noise associations at time t
      noiseObs = sorted([int(j) for j in z0[t] + [zk[t],] if j != obs_t[jIdx]])
      editDict_tkj[(t,0)] = noiseObs
    editDict_tkj[(t, k)] = obs_t[jIdx]
    q_new_given_old += logpObs[jIdx]

    # propagate kalman filter forward for track-obs distance
    ytn = y[t][obs_t[jIdx]]
    muX_o2n, SigmaX_o2n, _ = jpt.kalman.filter(okf, muX_o2n, SigmaX_o2n, ytn)

  z_ = z.edit(editDict_tkj, kind='tkj', inplace=False)

  # now resample w

  editDict_k = {}
  theta = dict(Q=okf.Q, R=okf.R, mu0=o.prior.mu0)
  xkNew, ykNew = ( {}, {} )
  for t, j in z_.to(k).items():
    xkNew[t] = None
    ykNew[t] = y[t][j]
  xkNew = jpt.kalman.ffbs(okf, xkNew, ykNew, x0=(o.prior.mu0, o.prior.Sigma0))
  editDict_k[int(k)] = ( theta, xkNew )

  # remove then put back in
  w_ = w.edit({k: None}, kind='k', inplace=False)
  w_ = w_.edit(editDict_k, kind='k', inplace=False)

  # reverse probability move
  lastT = fixedObs_t
  for t in ts:
    # skip probability
    if t not in zk:
      q_old_given_new += log_ps
      continue
    q_old_given_new += log_1ps

    # predict step
    for t_ in range(np.abs(lastT - t)):
      muX_n2o, SigmaX_n2o = jpt.kalman.predict(okf, muX_n2o, SigmaX_n2o)
      muY_n2o, SigmaY_n2o = jpt.kalman.StateToObs(okf, muX_n2o, SigmaX_n2o)
      SigmaYi_n2o = np.linalg.inv(SigmaY_n2o)
    lastT = t

    # compute track-obs distances
    if t in z0.keys() and t in zk.keys():
      obs_t = np.concatenate((z0[t], [zk[t], ] ))
    elif t in z0.keys() and t not in zk.keys():
      obs_t = z0[t]
    elif t in zk.keys() and t not in z0.keys():
      obs_t = [ zk[t], ]

    # obs_t = np.concatenate((z0[t], [zk[t], ] ))

    ytn = y[t][zk[t]]
    logpObs_single = -0.5 * mahalanobis(ytn, muY_n2o, SigmaYi_n2o)
    q_old_given_new += logpObs_single

    # propagate kalman filter forward for track-obs distance
    muX_n2o, SigmaX_n2o, _ = jpt.kalman.filter(okf, muX_n2o, SigmaX_n2o, ytn)

  # debugging
  for k_ in w_.ks:
    theta, x_tks = w_.x[k_]
    x_ts = list(x_tks.keys())
    z_ts = list(z_.to(k_).keys())
    missing = np.setdiff1d(np.union1d(x_ts,z_ts), np.intersect1d(x_ts,z_ts))
    if len(missing)>0:
      print('Problem in extend')
      ip.embed()


  # end debugging
  


  return w_, z_, True, q_old_given_new - q_new_given_old

def disperse(o, y, w, z):
  # check that we have tracks to disperse
  if not z.ks: return w, z, False, 0.0

  # randomly sample an existing track
  k = np.random.choice(z.ks)
  q_new_given_old = -np.log(len(z.ks))

  # determine reverse move probability
  q_old_given_new = 0.0

  ## get all times with noise-associated observations
  z0 = z.to(0)

  ## get all times from track associations
  zk = z.to(k)
  zkLast_t = max(zk.keys())

  ## list of all times from which track k was constructed
  ts = sorted(list(np.unique((list(z0.keys()) + list(zk.keys())))))

  # precompute
  log_ps = np.log(o.param.ps)
  log_1ps = np.log(1 - o.param.ps)

  ## setup filter
  muX = o.prior.mu0.copy()
  SigmaX = o.prior.Sigma0.copy()
  okf = jpt.kalman.opts(o.param.dy, o.param.dx, F=o.param.F, H=o.param.H)

  lastT = -1
  noTrackYet = True
  for t in ts:

    # skip probability
    if t not in zk:
      q_old_given_new += log_ps
      continue
    q_old_given_new += log_1ps

    # predict step from lastT .. now
    if lastT != -1:
      for t_ in range(lastT, t+1):
        muX, SigmaX = jpt.kalman.predict(okf, muX, SigmaX)
      muY, SigmaY = jpt.kalman.StateToObs(okf, muX, SigmaX)
      SigmaYi = np.linalg.inv(SigmaY)
    lastT = t

    # compute track-obs distances
    if t in z0.keys() and t in zk.keys():
      obs_t = np.concatenate((z0[t], [zk[t], ] ))
    elif t in z0.keys() and t not in zk.keys():
      obs_t = z0[t]
    elif t in zk.keys() and t not in z0.keys():
      obs_t = [ zk[t], ]

    logpObs = np.zeros(len(obs_t))
    for idx, j in enumerate(obs_t):
      # if there's no track yet
      if noTrackYet:
        noTrackYet = False
        break

      # track <-> obs distance to obs
      ytn = y[t][j]
      logpObs[idx] = -0.5 * mahalanobis(ytn, muY, SigmaYi)

    jIdx = len(obs_t)-1
    q_old_given_new += logpObs[jIdx]

    # propagate kalman filter forward for track-obs distance
    ytn = y[t][obs_t[jIdx]]
    muX, SigmaX, _ = jpt.kalman.filter(okf, muX, SigmaX, ytn)

  editDict_tkj = { (t,0): j for t, j in zk.items() }
  z_ = z.edit(editDict_tkj, kind='tkj', inplace=False)
  w_ = w.edit({k: None}, kind='k', inplace=False)

  return w_, z_, True, q_old_given_new - q_new_given_old


def gather(o, y, w, z):
  # check if we can create a new track
  if len(z.ks) == o.param.maxK: return w, z, False, 0.0

  # get all times with noise-associated observations
  z0 = z.to(0)
  ts = z0.keys()
  if len(ts) == 0: return w, z, False, 0.0

  q_new_given_old = 0.0
  log_ps = np.log(o.param.ps)
  log_1ps = np.log(1 - o.param.ps)

  # maintain filter distribution
  muX = o.prior.mu0.copy()
  SigmaX = o.prior.Sigma0.copy()
  okf = jpt.kalman.opts(o.param.dy, o.param.dx, F=o.param.F, H=o.param.H)

  lastT = -1

  kNew = z.next_k()
  editDict_tkj = {}
  for t in ts:
    # skip probability
    if np.random.rand() < o.param.ps:
      q_new_given_old += log_ps
      continue
    q_new_given_old += log_1ps

    # predict step from lastT .. now
    if lastT != -1:
      for t_ in range(lastT, t+1):
        muX, SigmaX = jpt.kalman.predict(okf, muX, SigmaX)
      muY, SigmaY = jpt.kalman.StateToObs(okf, muX, SigmaX)
      SigmaYi = np.linalg.inv(SigmaY)
    lastT = t

    # compute track-obs distances
    obs_t = z0[t]
    logpObs = np.zeros(len(obs_t))
    for idx, j in enumerate(obs_t):
      # if there's no track yet
      if len(editDict_tkj) == 0: break

      # track <-> obs distance to obs
      ytn = y[t][j]
      logpObs[idx] = -0.5 * mahalanobis(ytn, muY, SigmaYi)

    pObs = np.exp(logpObs - logsumexp(logpObs))
    jIdx = np.random.choice(np.arange(len(pObs)), p=pObs)
    q_new_given_old += logpObs[jIdx]
    editDict_tkj[(t, kNew)] = obs_t[jIdx]

    # propagate kalman filter forward for track-obs distance
    ytn = y[t][obs_t[jIdx]]
    muX, SigmaX, _ = jpt.kalman.filter(okf, muX, SigmaX, ytn)

  if len(editDict_tkj) == 0: return w, z, False, 0.0

  z_ = z.edit(editDict_tkj, kind='tkj', inplace=False)

  # now sample, make edited w with this new track
  editDict_k = {}
  theta = dict(Q=okf.Q, R=okf.R, mu0=o.prior.mu0)
  xkNew, ykNew = ( {}, {} )
  for t, j in z_.to(kNew).items():
    xkNew[t] = None
    ykNew[t] = y[t][j]
  xkNew = jpt.kalman.ffbs(okf, xkNew, ykNew, x0=(o.prior.mu0, o.prior.Sigma0))
  editDict_k[int(kNew)] = ( theta, xkNew )
  w_ = w.edit(editDict_k, kind='k', inplace=False)

  q_old_given_new = -np.log(len(w_.ks)) # random choice of 1/k

  return w_, z_, True, q_old_given_new - q_new_given_old


def update(o, y, w, z):
  # resample latent parameters, w
  ## 1: resample x_tk for all t, k
  editDict_k = {}
  for k in w.ks:
    # build x, y dictionary for kalman.ffbs
    zk = z.to(k)
    ts_k = zk.keys()
    xk = dict([(t, None) for t in ts_k])
    yk = dict([(t, y[t][zk[t]]) for t in ts_k])
    x0 = ( o.prior.mu0, o.prior.Sigma0 )
    theta = w.x[k][0]
    okf = jpt.kalman.opts(o.param.dy, o.param.dx, F=o.param.F, H=o.param.H,
      Q=theta['Q'], R=theta['R'])
    
    x_ = jpt.kalman.ffbs(okf, xk, yk, x0)
    editDict_k[k] = ( theta, x_ )

  ## 2: todo: resample Q, R (do later)
  
  ## 3: edit w
  w_ = w.edit(editDict_k, kind='k', inplace=False)

  return w_, z, True, 0.0

def switch(o, y, w, z):
  # Sample from swappable tracks
  times = track_swap_begin_times(o, z)
  valid = [kt for kt in times if kt is not None]
  # K = 2 # todo: sample
  K = min(7, len(z.ks))

  nValid = len(valid)
  if nValid < K or K < 2: return w, z, False, 0.0

  choiceIdx = sorted(np.random.choice(range(nValid), size=K, replace=False))
  choice = [ valid[i] for i in choiceIdx ]

  # theta, x are len(ks) tuples of dictionaries in order of ks
  try:
    ks, t0 = zip(*choice)
  except:
    print('ks, t0 problem')
    ip.embed()
    sys.exi()

  theta, x = zip(*[w.x[k] for k in ks])
  x = copy.deepcopy(x) # deep copy w.x so we can modify it

  Ri = [ np.linalg.inv(thet['R']) for thet in theta ]
  okf = [ jpt.kalman.opts(o.param.dy, o.param.dx, Q=theta[i]['Q'], R=theta[i]['R'],
    F=o.param.F, H=o.param.H) for i in range(K) ]
  P0 = [ np.zeros((o.dx, o.dx)) for o in okf ]

  # unique times in the switch window
  max_t0 = max(t0)
  ts = list({ t for k in ks for t in z.to(k).keys() if t >= max_t0 })

  # construct mu, Sigma for each k in ks, up to max_t0-1; these will be
  # propagated throughout the switch
  mu = [ [] for k in ks ]
  Sigma = [ [] for k in ks ]
  for i, k in enumerate(ks):
    lastT_outside_swap = max([ t for t in z.to(k).keys() if t <= max_t0-1])
    if lastT_outside_swap == max_t0-1:
      mu[i] = x[i][lastT_outside_swap]
      Sigma[i] = np.zeros((okf[i].dx, okf[i].dx))
    else:
      # not tested yet
      xsDict = { lastT_outside_swap: x[i][lastT_outside_swap], max_t0-1: None }
      mu[i], Sigma[i] = jpt.kalman.state_estimate(okf[i], xsDict)[max_t0-1]
  lastT = [ max_t0-1 for k in ks ]

  # iterate through time, propagating mu, Sigma, lastT for each switch time
  perms = list(itertools.permutations(range(len(ks))))
  nPerms = len(perms)

  qq = 0.0
  editDict_tkj = {}

  mu_t, Sigma_t = ( [ [] for k in ks ], [ [] for k in ks ] )
  SigmaI_t = [ [] for k in ks ]
  for t in ts:
    logp = np.zeros(nPerms)

    hasAssoc = [ t in z.to(k) for k in ks ]
    for i, k in enumerate(ks):
      xsDict = { lastT[i]: mu[i], t: None }
      xsDict = jpt.kalman.state_estimate(okf[i], xsDict, P0=Sigma[i])
      mu_t[i], Sigma_t[i] = xsDict[t]
      SigmaI_t[i] = np.linalg.inv(Sigma_t[i])

    for logp_idx, perm in enumerate(perms):
      # mu, Sigma, lastT, mu_t, Sigma_t, SigmaI_t remain ordered, only accessed with i
      ll = 0.0
      for i, k in enumerate(ks):
        # i -> i_
        # k -> k_
        i_ = perm[i]
        k_ = ks[i_]

        if not hasAssoc[i_]: continue

        # xtk, ytk use permutation accesses
        xtk_, ytk_ = ( x[i_][t], y[t][z.to(k_)[t]] )
        ll += -0.5 * mahalanobis(xtk_, mu_t[i], SigmaI_t[i])
        ll += -0.5 * mahalanobis(ytk_, okf[i].H @ xtk_, Ri[i])

      logp[logp_idx] = ll

    # normalize logp and sample association permutation
    expp = np.exp(logp - logsumexp(logp))
    permIdx = np.random.choice(range(nPerms), p=expp)
    perm = perms[permIdx]

    # print(logp)
    # print(expp)

    qq += logp[permIdx] # just a simple unnormalized logp accumulator

    # propagate state forward
    xEdits = [ {} for k in ks ]
    for i, k in enumerate(ks):
      # i -> i_
      # k -> k_
      i_ = perm[i]
      k_ = ks[i_]
      
      if hasAssoc[i_]:
        xtk_ = x[i_][t]
        mu[i], Sigma[i], lastT[i] = ( xtk_, P0[i_].copy(), t )
        
        # No change to z.to(0) in editDict_tkj b/c no change in noise obs
        j = z.to(k_)[t]
        editDict_tkj[(t, k)] = j
        xEdits[i][t] = xtk_

      else:
        mu[i], Sigma[i], lastT[i] = ( mu_t[i], Sigma_t[i], t )

        # if there was an association to x[i][t], remove it
        if t in x[i]: xEdits[i][t] = None

    for i in range(K):
      for t in xEdits[i].keys():
        x[i][t] = xEdits[i][t]

  # edit z, w
  z_ = z.edit(editDict_tkj, kind='tkj', inplace=False)

  # tracks must have at least 2 associations and we want to be able to switch
  # the second association of each track
  for k in ks:
    if len(z_.to(k)) < 2: return w, z, False, 0.0

  editDict_k = {}
  for i, k in enumerate(ks): editDict_k[int(k)] = ( theta[i], x[i] )
  w_ = w.edit(editDict_k)

  return w_, z_, True, 0.0


# def switch(o, y, w, z):
#   windows = possible_switch_windows(o, z, reverse=False)
#   if len(windows) == 0: return w, z, False, 0.0
#   switchIdx = np.random.choice(len(windows))
#   ks, t0, ts = windows[switchIdx]
#   x0 = []
#   for idx, k in enumerate(ks): x0.append( w.x[k][1][t0[idx]] )
#
#   Qi = dict([(k, np.linalg.inv( w.x[k][0]['Q'])) for k in ks])
#   Ri = dict([(k, np.linalg.inv( w.x[k][0]['R'])) for k in ks])
#   kf = dict([(k, jpt.kalman.opts(o.param.dy, o.param.dx,
#     F=o.param.F, H=o.param.H, Q=w.x[k][0]['Q'], R=w.x[k][0]['R'] ))
#     for k in ks])
#
#   def costxx(t1, x1, t2, x2, k):
#     # todo: handle separated t1, t2
#     # (t1, x1) is AFTER (t2, x2)
#     assert t1 > t2
#     return -0.5 * mahalanobis(x1, np.dot(o.param.F, x2), Qi[k])
#
#   def costxy(t, k, xtk, ytk):
#     # costxy is also where annotations get handled
#     return -0.5 * mahalanobis(ytk, np.dot(o.param.H, xtk), Ri[k])
#
#   def val(t,k):
#     if t in w.x[k][1]:
#       jtk = z.to(k)[t]
#       xk = w.x[k][1]
#       xtk, ytk = ( xk[t], y[t][jtk] )
#       # xtk, ytk = ( w.x[k][1][t], y[t][jtk] )
#     else:
#       # todo: handle reverse direction
#       dct = w.x[k][1].copy()
#       dct[t] = None
#       muX, SigmaX = jpt.kalman.state_estimate(kf[k], dct)[t]
#       muY, SigmaY = jpt.kalman.StateToObs(kf[k], muX, SigmaX)
#       xtk, ytk, jtk = (muX, muY, None)
#     return xtk, ytk, jtk
#
#   # make hmm, sample
#   perms, pi, Psi, psi = jpt.hmm.build(t0, x0, ts, ks, val, costxx, costxy)
#
#   # S, q_new_given_old, q_old_given_new = jpt.hmm.ffbs(pi, Psi, psi)
#   S, q_new_given_old, q_old_given_new = jpt.hmm.sample(pi, Psi, psi)
#   print(S)
#   print(f'q_old_given_new: {q_old_given_new:.2f}, q_new_given_old: {q_new_given_old:.2f}')
#
#   if len(w.ks) <= 4:
#     None
#     ## dsh: re-enable
#     # print('inside switch')
#     # ip.embed()
#
#   # edit w, z
#   ## S indexes into perms; perms is wrt ks
#   editDict_tkj = {}
#   editDict_k = dict([(k, (w.x[k][0], {})) for k in ks])
#   for idx, t in enumerate(ts):
#     t = int(t)
#     p = perms[S[idx]]
#     for kOld_idx, kOld in enumerate(ks):
#       kNew = ks[p[kOld_idx]]
#       xtk, ytk, jtk = val(t, kNew)
#       if jtk is None: editDict_k[kOld][1][t] = None; continue
#       editDict_tkj[(t, kOld)] = jtk
#       editDict_k[kOld][1][t] = xtk
#   z_ = z.edit(editDict_tkj, kind='tkj', inplace=False)
#   w_ = w.edit(editDict_k, kind='k', inplace=False)
#
#   # experiment: edit w_ again with gibbs sample
#   w_, _, _, _ = update(o, y, w_, z_)
#   # end experiment
#
#   return w_, z_, True, q_old_given_new - q_new_given_old

def split(o, y, w, z):
  """ Sample a random split proposal for UniqueBijectiveAssociation, z. """
  if len(z.ks) == o.param.maxK: return w, z, False, 0.0

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
  if not z.ks: return w, z, False, 0.0

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

def track_swap_begin_times(o, z):
  # record 1 + time of first assocation in each track
  times = [ None for k in range(len(z.ks)) ]
  for idx, k in enumerate(z.ks):
    zkt = list(z.to(k).keys())
    assert len(zkt) >= 2, 'Track lengths must be >= 2'
    if zkt[0] != z.tE: times[idx] = (k, 1+zkt[0])
  return times
