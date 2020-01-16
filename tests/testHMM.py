from context import jpt
import numpy as np, matplotlib.pyplot as plt
import warnings, du
import IPython as ip
import copy
from scipy.special import logsumexp as lse
np.set_printoptions(precision=2, suppress=True)

def testMakeFixed

def testHMM():
  # Build two-timestep, two-state HMM with 50/50 prior, where:
  #   state 1 has log cost 2a
  #   state 2 has log cost 2b
  # For a = -1, b = -2. Then do FFBS sampling and check that the frequency of
  # samples is approximately
  #   (1, 1): 0.79
  #   (1, 2): 0.1
  #   (2, 1): 0.1
  #   (2, 2): 0.01
  T = 2
  nStates = 2
  a, b = (-1, -2)
  pi = np.array([1.0, 0.0])

  Psi = np.zeros((T, nStates, nStates))
  psi = np.zeros((T+1, nStates))

  for t in range(T):
    Psi[t, 0, 0] = 2*a
    Psi[t, 0, 1] = 2*b
    Psi[t, 1, 0] = 2*a
    Psi[t, 1, 1] = 2*b

  Psi = np.exp(Psi - lse(Psi, axis=2, keepdims=True))
  psi = np.exp(psi - lse(psi, axis=1, keepdims=True))

  from hmmlearn import hmm
  H = hmm.MultinomialHMM(n_components=nStates)
  H.startprob_ = pi
  H.transmat_ = Psi[0]
  H.emissionprob_ = psi[0][:,np.newaxis]

  nTrials = 10000
  outcomes = np.zeros(4)
  for n in range(nTrials):
    _, x = H.sample(T+1)

    if   x[1] == 0 and x[2] == 0: outcomes[0] += 1
    elif x[1] == 0 and x[2] == 1: outcomes[1] += 1
    elif x[1] == 1 and x[2] == 0: outcomes[2] += 1
    elif x[1] == 1 and x[2] == 1: outcomes[3] += 1
  
  print( outcomes / nTrials )

  outcomes2 = np.zeros(4)
  for n in range(nTrials):
    x, ll, _ = jpt.hmm.ffbs(pi, Psi, psi)

    if   x[1] == 0 and x[2] == 0: outcomes2[0] += 1
    elif x[1] == 0 and x[2] == 1: outcomes2[1] += 1
    elif x[1] == 1 and x[2] == 0: outcomes2[2] += 1
    elif x[1] == 1 and x[2] == 1: outcomes2[3] += 1

  print( outcomes2 / nTrials )
