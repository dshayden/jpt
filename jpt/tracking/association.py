import jpt
from abc import ABC, abstractmethod
# from .observation import ObservationSet
from warnings import warn

class Association(ABC):
  """ Stores an association event of targets to observations.
  
  This is the glue between a given Hypothesis and ObservationSet (either of
  which may be resampled over time).
  """

  def __init__(self, y):
    # assert isinstance(y, ObservationSet)
    assert isinstance(y, jpt.ObservationSet)
    self.t0, self.tE, self.N, self.ts, self.ks = (-1, -1, {}, [], [])
    super().__init__()

  @abstractmethod
  def __getitem__(self, t):
    """ Get all associations at time t. """
    pass

  @abstractmethod
  def to(self, k):
    """ Get all associations to option k. """
    pass 

import numpy as np
class UniqueBijectiveAssociation(Association, ):
  """ Association event with constraints.
  
  Constraints:
    All observations must have an associated target.
    All targets must have no more than one association at each time.
  """
  def __init__(self, y, zs):
    """ ObservationSet y, assignments dictionary zs { t: (n_t,) }. """
    super().__init__(y)
    self._z = dict([ (int(t), np.asarray(zt, dtype=np.int))
      for t, zt in zs.items() ])
    assert set(self._z) == set(y.N) # check that keys are all same
    for t in self._z.keys():
      assert self._z[t].ndim == 1 and len(self._z[t]) == y.N[t]
    self.N = y.N
    self.t0, self.tE = ( min(self.N.keys()), max(self.N.keys()) )
    self.ts = sorted(self.N.keys())
    self.__build_lut()
    self.ks = sorted(self._K.keys())
    self.__check_consistency()

  def __build_lut(self):
    # build lookup table of k -> {ts}_k
    self._K = { }
    for t in self._z.keys():
      for j in range(self.N[t]):
        k = self._z[t][j]
        if k not in self._K: self._K[k] = { t: j }
        else: self._K[k][t] = j

  def __check_consistency(self):
    for t in self._z.keys():
      if len(self._z[t]) != len(np.unique(self._z[t])):
        warn(f'Non-unique values in self._z[{t}]: \n{self._z[t]}')
        return False

      for j in range(self.N[t]):
        k = self._z[t][j]
        if k not in self._K:
          warn(f'z_{t} {j} = {k}, but {k} not in self._K')
          return False

        if t not in self._K[k]:
          warn(f'z_{t} {j} = {k}, but {t} not in self._K[k]')
          return False

        if self._K[k][t] != j:
          warn(f'z_{t},{j} = {k}, but self._K[k][t] = {self._K[k][t]}')
          return False

    return True

  def __getitem__(self, t):
    """ Get all associations at time t as an array with values in ks.
    
    If z = UniqueBijectiveAssociation(y, zs)
    Then z[t][j] = k <=> Object k is associated to y[t][j]
    """
    return self._z[t]

  def to(self, k):
    """ Get all associations to option k as {t: j} for j an index into y[t] """
    return self._K[k]
