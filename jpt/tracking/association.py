import jpt
from abc import ABC, abstractmethod
import warnings
from warnings import warn
import copy
import IPython as ip

class Association(ABC):
  """ Stores an association event of targets to observations.
  
  This is the glue between a given Tracks and ObservationSet (either of
  which may be resampled over time).
  """

  def __init__(self):
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
class UniqueBijectiveAssociation(Association, jpt.Serializable):
  """ Association event with constraints.
  
  Constraints:
    All observations must have an associated target.
    All targets must have no more than one association at each time.
  """
  def __init__(self, N, zs):
    """ dict N {t: n_t }, assignments dictionary zs { t: (n_t,) }. """
    super().__init__()
    self._z = dict([ (int(t), np.asarray(zt, dtype=np.int))
      for t, zt in zs.items() ])
    assert set(self._z) == set(N) # check that keys are all same
    for t in self._z.keys():
      assert self._z[t].ndim == 1 and len(self._z[t]) == N[t]
    self.N = N
    self.t0, self.tE = ( min(self.N.keys()), max(self.N.keys()) )
    self.ts = sorted(self.N.keys())
    self.__build_lut()
    self.ks = sorted(self._K.keys())
    self.__check_consistency()

  def edit(self, e, kind='tkj', inplace=False):   
    """ Modify data associations with edit e.

    INPUT
      e: dictionary (if kind in ['tkj', 'tjk']) or list (if kind == 't')
      kind (str): One of 'tkj', 'tjk' or 't'
      inplace (bool): Edit in place or return a new association hypothesis.

    OUTPUT
      z (UniqueBijectiveAssociation): Copy of or reference

    Edit e is either a dictionary (for kind = 'tkj' or 'tjk')
    or a list (for kind = 't').

    Edit kind is one of 'tkj', 'tjk', or 't'. (only tkj supported for now)

    Inplace 
    
    """
    z = self if inplace else copy.deepcopy(self)
    if kind == 'tkj':
      for (t, k), j in e.items(): z._z[t][j] = k
      z.__build_lut()
      z.ks = sorted(z._K.keys())
      z.__check_consistency()
    elif kind == 'tjk':
      None
    elif kind == 't':
      None
    else:
      raise ValueError("kind must be one of 'tjk', 'tkj', or 't'")

    return z

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
        ip.embed()
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

  def serializable(self):
    # return dict with appropriate keys for serialization that does't depend on
    # anything more than basic or numpy types
    return { self._magicKey: [self.N, self._z], 
      self._classKey: 'UniqueBijectiveAssociation'
    }

  def fromSerializable(N, _z):
    # return UniqueBijectiveAssociation loaded from args pulled from serialized data
    return UniqueBijectiveAssociation(N, _z)

  def __getitem__(self, t):
    """ Get all associations at time t as an array with values in ks.
    
    If z = UniqueBijectiveAssociation(y, zs)
    Then z[t][j] = k <=> Object k is associated to y[t][j]
    """
    return self._z[t]

  def to(self, k):
    """ Get all associations to option k as {t: j} for j an index into y[t] """
    return self._K[k]
  
  def next_k(self):
    """ Return next valid index of a new unique label. """
    return max(self.ks) + 1
