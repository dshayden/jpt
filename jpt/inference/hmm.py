import itertools
import numpy as np
from scipy.special import logsumexp
import IPython as ip

def build(t0, ts, ks, val, costxx, costxy):
  """ Construct swap hmm for K targets.

  INPUT
    t0 (len-K list): start times for each target
    ts (len-T list): times inside swap window
    ks (len-K list): target indices
    val (fcn): of the form xt, yt, jt = val(t,k)
    costxx (fcn): of the form cx = costxx(t1,x1,t2,x2,k)
    costxy (fcn): of the form cy = costxy(t,k,x,y)

  OUTPUT
    perms (list): Index permutations of ks (HMM states are in range(nPerms))
    Permutations of indices into ks that HMM states index into.
    pi (nPerms): Prior
    Psi (T-1, nPerms, nPerms): Transition Matrix
    psi (T, nPerms): Observation Matrix
  """
  T, K = ( len(ts), len(ks) )
  assert K > 1 and K <= 4, 'Combinatorics too large for K > 4'
  perms = list(itertools.permutations(range(K)))
  nPerms = len(perms)

  # len-T list of len-K list of tuples (xtk, ytk, jtk)
  vs = [ [val(t,k) for k in ks] for t in ts ]

  # build unnormalized log transition matrix
  Psi = np.zeros((T-1, nPerms, nPerms))
  for idx in range(T-1):
    tPrime, t = ( ts[idx], ts[idx+1] )
    for cPrime, pPrime in enumerate(perms):
      for c, p in enumerate(perms):
        for k in range(K): # compute over each k (pPrime[k], p[k])
          xtPrime, _, _ = vs[idx][pPrime[k]]
          xt, yt, _ = vs[idx+1][p[k]]
          Psi[idx, cPrime, c] += costxx(tPrime, xtPrime, t, xt, k)
          Psi[idx, cPrime, c] += costxy(t, k, xt, yt)

  # build unnormalized log prior
  pi = np.zeros(nPerms)
  idx = -1
  t = ts[idx+1]
  for c, p in enumerate(perms):
    for k in range(K): # compute over each k (pPrime[k], p[k])
      tPrime = t0[k]
      xtPrime, _, _ = vs[idx][k]
      xt, yt, _ = vs[idx+1][p[k]]
      pi[c] += costxx(tPrime, xtPrime, t, xt, k)
      pi[c] += costxy(t, k, xt, yt)

  # normalize and return
  Psi = np.exp(Psi - logsumexp(Psi, axis=2, keepdims=True))
  pi = np.exp(pi - logsumexp(pi))
  psi = [ None for t in ts ]
  return perms, pi, Psi, psi
