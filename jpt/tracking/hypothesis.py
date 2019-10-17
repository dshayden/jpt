import jpt
from abc import ABC, abstractmethod

class Hypothesis(ABC):
  """ Proposed set of targets and their latent states.

  Target information minimally includes start/end times and locations.
  Hypothesis DOES NOT store data associations and is not aware of any
  ObservationSet.
  """
  def __init__(self):
    self.t0, self.tE, self.ts, self.ks = (-1, -1, [], [])

  @abstractmethod
  def x(self):
    """ Get dict of dicts of all latent states. """
    pass

  @abstractmethod
  def __getitem__(self, t):
    """ Get all target latent states at time t. """
    pass

import numpy as np
class PointHypothesis(Hypothesis, jpt.Serializable):
  def __init__(self, x):
    # x must be of the form { k: {t: x_tk, ...}, ... }
    #   k, t must be integers. x_tk can be any type (ndarray, tuple, etc.)
    super().__init__()
    ts = [ ]
    for k, kDict in x.items():
      assert type(k) == int
      for t, v in kDict.items(): assert type(t) == int
      ts += [ t for t in kDict.keys() ]
    self.ts = sorted(np.unique(ts))
    self._x = x
    self.ks = sorted(self._x.keys())
    self.t0, self.tE = ( min(self.ts), max(self.ts) )

  def x(self):
    return self._x

  def __getitem__(self, t):
    # return dict of {k: x_tk} at time t
    return dict([ (k, self._x[k][t]) for k in self.ks if t in self._x[k] ])
  
  def serializable(self):
    # return dict with appropriate keys for serialization that does't depend on
    # anything more than basic or numpy types
    return { self._magicKey: [self._x,], self._classKey: 'PointHypothesis' }

  def fromSerializable(_x):
    # return PointHypothesis loaded from args pulled from serialized data
    return PointHypothesis(_x)
