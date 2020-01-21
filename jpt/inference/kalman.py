import numpy as np
from numpy.linalg import multi_dot as mdot
import scipy.stats, argparse, du
import scipy.linalg as sla
from scipy.stats import multivariate_normal as mvn
from scipy.spatial.distance import mahalanobis
import IPython as ip

def opts(dy, dx, **kwargs):
  """ Create options struct for the filter. 

  INPUTS
    dy (numeric): observation space dimension
    dx (numeric): latent space dimension

  KEYWORD INPUTS
    F (ndarray, [dx, dx]): system model, default is random acceleration
    H (ndarray, [dy, dx]): measurement model
    Q (ndarray, [dx, dx]): system noise
    R (ndarray, [dy, dy]): observation noise

  OUTPUTS
    o (Namespace): filter options struct
  """
  o = argparse.Namespace()

  # observed and latent dimensions
  o.dy = dy
  o.dx = dx

  eye = np.eye(dy)
  zer = np.zeros((dy, dy))

  # random acceleration dynamics
  o.F = kwargs.get('F', np.concatenate((
    np.concatenate((eye, eye), axis=1),
    np.concatenate((zer, eye), axis=1))))

  # measurment fcn, just take position from latent space
  o.H = kwargs.get('H', np.concatenate((eye, zer), axis=1))

  # system noise
  # o.Q = kwargs.get('Q', sla.block_diag(2*eye, 10*eye))
  o.Q = kwargs.get('Q', sla.block_diag(0.5*eye, 1*eye))

  # Qd = kwargs.get('Qd', 20*np.eye(dy))
  # o.Q = kwargs.get('Q', np.concatenate((
  #   np.concatenate((zer, zer), axis=1),
  #   np.concatenate((zer,  Qd), axis=1))))

  # observation noise
  # o.R = kwargs.get('R', 20*np.eye(dy))
  o.R = kwargs.get('R', 0.1*np.eye(dy))

  return o

def predict(o, x, P):
  """ Predict new state given current estimate x, with cov P. 

  INPUTS
    o (Namespace): options
    x (ndarray, [o.dx,]): latent state estimate
    P (ndarray, [o.dx, o.dx]): latent state cov estimate 

  OUTPUTS
    xh (ndarray, [o.dx,]): latent state estimate ran thru fwd model
    Ph (ndarray, [o.dx, o.dx]): latent state cov estimate  ran thru fwd model
  """
  return np.dot(o.F, x), mdot((o.F, P, o.F.T)) + o.Q

def filter(o, xh, Ph, y):
  """ filter state estimate given observation y. 

  INPUTS
    o (Namespace): options
    xh (ndarray, [o.dx,]: predicted latent state
    Ph (ndarray, [o.dx, o.dx]: predicted latent state cov estimate
    y (ndarray, [o.dy,]: observation

  OUTPUTS
    x (ndarray, [o.dx,]): filtered latent state estimate update
    P (ndarray, [o.dx, o.dx]): filtered latent state cov estimate update
    ll (numeric): log-likelihood of measurement
  """
  r = y - np.dot(o.H, xh) # residual of predicted state and obs
  S = mdot((o.H, Ph, o.H.T)) + o.R # project system uncertainty into obs space
  # K = du.mrdivide(np.dot(Ph, o.H.T), S).T
  K = mdot((Ph, o.H.T, np.linalg.inv(S)))
  x = xh + np.dot(K, r)

  I_KH = np.eye(o.dx) - np.dot(K, o.H)
  P = mdot((I_KH, Ph, I_KH.T)) + mdot((K, o.R, K.T))

  ll = scipy.stats.multivariate_normal.logpdf(y, np.dot(o.H, x), S)

  return x, P, ll

def StateToObs(o, x, P):
  # Return given state estimate and covariance as projected into observation state
  return np.dot(o.H, x), mdot((o.H, P, o.H.T))+o.R

def filterInnovation(o, xh, Ph, y):
  # project system uncertainty into obs space
  r = y - np.dot(o.H, xh)
  S = mdot((o.H, Ph, o.H.T)) + o.R
  return r, S

def filterInnovationMatrix(o, Ph):
  # project system uncertainty into obs space
  return mdot((o.H, Ph, o.H.T)) + o.R

def smooth(o, xf, Pf, xb, Pb):
  """ Perform RTS smoothing.

  Equations from:
    https://users.aalto.fi/~ssarkka/course_k2011/pdf/handout7.pdf
    (Their A is my F, mk is xf, Pk is Pf)

  INPUTS
    o (Namespace): options
    xf (ndarray, [o.dx,]: filtered latent state (from t)
    Pf (ndarray, [o.dx, o.dx]: filtered latent state cov estimate (from t)
    xb (ndarray, [o.dx,]: smoothed latent state (from t+1)
    Pb (ndarray, [o.dx, o.dx]: smoothed latent state cov estimate (from t+1)

  OUTPUTS
    x (ndarray, [o.dx,]): smoothed latent state estimate
    P (ndarray, [o.dx, o.dx]): smoothed latent state cov estimate
  """
  xh, Ph = predict(o, xf, Pf)
  G = Pf.dot(o.F.T).dot(np.linalg.inv(Ph))
  x = xf + np.dot(G, xb-xh)
  P = Pf + mdot((G, Pb-Ph, G.T))
  return x, P

def state_estimate(o, x, **kwargs):
  # x: dictionary of time : observed latent, None means give marginal dist for that time
  #   will smooth if x is obsserved after Nones
  #   will filter if x is only obsserved before Nones
  # assumes non-None values of x are samples, hence have 0 covariance. Can
  # change for the first observed x by specifying P0 (as when it is a prior)
  # Note: this doesn't handle observations on x in observation space; could do
  #       this with a y dictionary straightforwardly, but it's not what's
  #       needed for the use of this.
  P0 = kwargs.get('P0', np.zeros((o.dx, o.dx)))

  # get the window to filter/smooth inside of
  ## get all None keys
  tNone = [ t for t, obs in sorted(x.items()) if obs is None ]
  minNone, maxNone = ( min(tNone), max(tNone) )
  
  ## get all observed keys
  tObs = [ t for t, obs in sorted(x.items()) if obs is not None ]

  ## what's the last time to filter to, and do we have info to smooth with?
  lastT = [ t for t in tObs if t > maxNone ]
  goBack, lastT = ( True, min(lastT) ) if len(lastT) > 0 else ( False, maxNone )

  ## what's the latest time before Nones that we have an observation?
  firstT = [ t for t in tObs if t < minNone ]
  assert len(firstT) > 0
  firstT = max(firstT)

  # filter
  ## naively store all between [firstT, lastT] inclusive
  ts = range(firstT, lastT+1)
  nSteps = lastT - firstT + 1
  assert len(ts) == nSteps

  xf = np.zeros((nSteps, o.dx))
  Pf = np.zeros((nSteps, o.dx, o.dx))
  xf[0], Pf[0] = ( x[firstT], P0 ) 
  for i, _ in enumerate(range(firstT+1, lastT+1)):
    xf[i+1], Pf[i+1] = predict(o, xf[i], Pf[i])

  if not goBack:
    xfDict = {}
    for t in tNone:
      i = ts.index(t)
      xfDict[t] = ( xf[i], Pf[i] )
    return xfDict

  xs = np.zeros((nSteps, o.dx))
  Ps = np.zeros((nSteps, o.dx, o.dx))
  xs[-1], Ps[-1] = ( x[lastT], np.zeros((o.dx, o.dx)) )
  for i in reversed(range(1, nSteps-1)):
    xs[i], Ps[i] = smooth(o, xf[i], Pf[i], xs[i+1], Ps[i+1])
  xs[0] = x[firstT]

  assert np.all(np.isclose(xf[0], x[firstT]))
  assert np.all(np.isclose(xs[lastT - firstT], x[lastT]))

  xsDict = {}
  for t in tNone:
    i = ts.index(t)
    xsDict[t] = ( xs[i], Ps[i] )
  return xsDict

def ffbs(o, x, y, x0=None, return_ll=False, no_resample=False):
  # x: dictionary of {t: None} for all times with desired inference
  # y: dictionary of {t: vals} for all times with at least one observation
  # x0: (mu, Sigma) tuple signifying prior dist; very broad if not specified
  assert np.all(np.linalg.eigvals(o.Q) > 0), "ffbs non-ergodic for degenerate Q"
  ts = [ t for t in y ]
  # if x0 is None: x0 = ( np.zeros(o.dx), 1e9*np.eye(o.dx) )
  if x0 is None: x0 = ( np.zeros(o.dx), sla.block_diag(1e9*np.eye(o.dy), 100*np.eye(o.dy)) )
  xPrev, PPrev = x0

  obs = [ (t, yt) for t, yt in y.items() ]
  nObs = len(obs)

  # need to store #obs of filter values
  xf = np.zeros((nObs, o.dx))
  Pf = np.zeros((nObs, o.dx, o.dx))

  # filter
  tPrev = obs[0][0] - 1
  for idx, (t, yt) in enumerate(obs):
    tDelta = t-tPrev
    xh, Ph = (xPrev.copy(), PPrev.copy())

    # predict & update, powering up predict as much as needed
    for t_ in range(tDelta): xh, Ph = predict(o, xh, Ph)
    xf[idx], Pf[idx] = (xh, Ph)
    for ytj in np.atleast_2d(yt):
      xf[idx], Pf[idx], _ = filter(o, xf[idx], Pf[idx], ytj)
    xPrev, PPrev, tPrev = ( xf[idx], Pf[idx], t )

  ll = 0.0

  xs = np.zeros((nObs, o.dx))

  if no_resample:
    ts = [ ob[0] for ob in obs ]
    assert all([ x[t] is not None for t in ts ])
    xs[-1] = x[t]
    ll += mvn.logpdf(xs[-1], xf[-1], Pf[-1])
    assert type(ll) == np.float64
  else:
    xs[-1] = mvn.rvs(xf[-1], Pf[-1])
    ll += mvn.logpdf(xs[-1], xf[-1], Pf[-1])

  tNext = obs[-1][0]
  for idx in reversed(range(nObs-1)):
    t, j = obs[idx]
    tDelta = tNext - t

    # power up predict based on number of intervening timesteps
    # can't just use kalman predict because the x_t is latent; so we pass F**d
    # as the "H" of inferNormalNormal
    d = tDelta
    H = np.linalg.matrix_power(o.F, d)
    Q_ = o.Q
    for t_ in range(1, tDelta): Q_ = o.F.dot(Q_).dot(o.F.T) + o.Q 

    # "Observation" is xs[idx+1] with mean F x_t, covariance Q (if no time gap)
    #                                 else integrate out x_{t+1, .., x_{t+d-1}
    # "Prior" is on joint x_t sample, with mean xf[idx], covariance Pf[idx]
    #     
    #           inferNormalNormal(y,     SigmaY, muX,     SigmaX,  H=None, b=None)
    mu, Sigma = inferNormalNormal(xs[idx+1], Q_, xf[idx], Pf[idx], H=H)

    if no_resample: xs[idx] = x[t]
    else: xs[idx] = mvn.rvs(mu, Sigma)
    ll += mvn.logpdf(xs[idx], mu, Sigma)
    assert type(ll) == np.float64
    tNext = t

  # for idx, (t, _) in enumerate(obs): x[int(t)] = xs[idx]

  # only compute new values for x if we didn't already pass them in (if we did,
  # then we're only computing logpdf here)
  for idx, (t, _) in enumerate(obs):
    if x[int(t)] is None: x[int(t)] = xs[idx]

  if return_ll:
    # handle denominators p(x_{t+1} | y_{1:t})
    llDenom = 0.0
    for idx, (t, yt) in enumerate(obs[:-1]):
      tNext = obs[idx+1][0]
      tDelta = tNext - t

      xh, Ph = ( xf[idx].copy(), Pf[idx].copy() )
      for t_ in range(tDelta): xh, Ph = predict(o, xh, Ph)

      # np.set_printoptions(suppress=True, precision=2)
      # print(t, '\n', xs[idx+1], '\n', xh, '\n', np.diag(Ph), '\n\n')
      # print(f'{t}')
      # print('xh:', xh)
      # print('Ph (diag):', np.diag(Ph))
      # print('xs[idx+1]:', xs[idx+1])
      # print(llDenom)
      # ip.embed()

      # llDenom += mvn.logpdf( xs[idx+1], xh, Ph ) # x_{t+1} == xs[idx+1]

      # pull from x, not xs, so we can just compute LL if desired
      llDenom += mvn.logpdf( x[tNext], xh, Ph ) # x_{t+1} == xs[idx+1]
      assert type(llDenom) == np.float64

    # print(f'll: {ll - llDenom:.2f}, llNum: {ll:.2f}, llDen: {llDenom:.2f}')
    ll -= llDenom
    # assert type(ll) == float
    assert type(llDenom) == np.float64


  if not return_ll: return x
  else: return x, ll

def predictive(o, x, t):
  # get predictive distribution of x, y at time t
  if t in x:
    muX, SigmaX = ( x[t], np.zeros((o.dx, o.dx)) )
  else:
    dct = dict([(t_, x) for t_, x in x.items() if x is not None])
    if all( t < np.array(list(dct.keys())) ):
      # treat as reverse filter
      minT = min(dct.keys())
      muX, SigmaX = ( dct[minT], np.zeros((o.dx, o.dx)) )

      # swap F for reverse F, assuming it handles velocity
      Ftrue = o.F.copy()
      o.F[:o.dy, o.dy:] *= -1
      for t_ in range( minT - t ): muX, SigmaX = predict(o, muX, SigmaX)
      o.F = Ftrue
    else:
      dct[t] = None
      muX, SigmaX = state_estimate(o, dct)[t]
  muY, SigmaY = StateToObs(o, muX, SigmaX)
  return muX, SigmaX, muY, SigmaY

def joint_ll(o, x, y, x0):
  # x: dictionary of {t: vals} for all times with desired inference
  # y: dictionary of {t: vals} for all times with at least one observation
  # x0: (mu, Sigma) tuple signifying prior dist; very broad if not specified
  assert len(x) == len(y)
  Ri = np.linalg.inv(o.R)

  ll = 0.0 

  xPrev, PPrev = x0
  tPrev = min(x.keys()) - 1
  for idx, t in enumerate(x.keys()):

    # power up by # intervening times
    tDelta = t-tPrev
    if tDelta > 1:
      xh, Ph = (xPrev.copy(), PPrev.copy())
      for t_ in range(tDelta-1): xh, Ph = predict(o, xh, Ph)
    else:
      xh, Ph = (np.dot(o.F, xPrev), o.Q)

    # x_t, y_t
    ll += -0.5 * mahalanobis(x[t], xh, np.linalg.inv(Ph))
    ll += -0.5 * mahalanobis(y[t], np.dot(o.H, x[t]), Ri)

    xPrev, PPrev, tPrev = ( x[t], np.zeros((o.dx, o.dx)), t )

  return ll

def inferNormalNormal(y, SigmaY, muX, SigmaX, H=None, b=None):
  r''' Conduct a conjugate Normal-Normal update on the below system:

  p(x | y) \propto p(x)               p(y | x)
                 = N(x | muX, SigmaX) N(y | Hx + b, SigmaY)

  INPUT
    y (ndarray, [dy,]): Observation
    SigmaY (ndarray, [dy, dy]): Observation covariance
    muX (ndarray, [dx,]): Prior mean
    SigmaX (ndarray, [dx,dx]): Prior covariance
    H (ndarray, [dy,dx]): Multiplicative dynamics
    b (ndarray, [dy,]): Additive dynamics

  OUPUT
    mu (ndarray, [dx,]): Posterior mean
    Sigma (ndarray, [dx,dx]): Posterior covariance
  '''
  dy, dx = (y.shape[0], muX.shape[0])
  if H is None: H = np.eye(dy,dx)
  if b is None: b = np.zeros(dy)

  SigmaXi = np.linalg.inv(SigmaX)
  SigmaYi = np.linalg.inv(SigmaY)

  SigmaI = SigmaXi + H.T.dot(SigmaYi).dot(H)
  Sigma = np.linalg.inv(SigmaI)
  mu = Sigma.dot(H.T.dot(SigmaYi).dot(y-b) + SigmaXi.dot(muX))

  return mu, Sigma
