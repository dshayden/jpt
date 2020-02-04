import jpt
from abc import ABC, abstractmethod
import warnings
from warnings import warn
import copy

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
    All observations must have an associated target (or noise).
    All targets must have no more than one association at each time
    Targets are represented by k = 1, 2, ...
    Noise represented by k = 0
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
    # build lookup tables of
    #   target associations _K: {k : { t : j }}
    #   noise associations _K0: {t : [j1, j2, ...]}
    self._K = { }
    self._K0 = { }
    for t in self._z.keys():
      for j in range(self.N[t]):
        k = self._z[t][j]
        if k == 0:
          if t not in self._K0: self._K0[t] = [j,]
          else: self._K0[t].append(j)
        elif k not in self._K: self._K[k] = { t: j }
        else: self._K[k][t] = j

  def __check_consistency(self):
    for t in self._z.keys():
      ztPos = self._z[t][ self._z[t] != 0 ]
      if len(ztPos) != len(np.unique(ztPos)):
        warn(f'Non-unique, non-0 values in self._z[{t}]: \n{self._z[t]}')
        ip.embed()
        return False

      for j in range(self.N[t]):
        k = self._z[t][j]
        if k != 0 and k not in self._K:
          warn(f'z_{t} {j} = {k}, but {k} not in self._K')
          return False
        elif k != 0 and t not in self._K[k]:
          warn(f'z_{t} {j} = {k}, but {t} not in self._K[k]')
          return False
        elif k != 0 and self._K[k][t] != j:
          warn(f'z_{t},{j} = {k}, but self._K[k][t] = {self._K[k][t]}')
          return False
        elif k == 0:
          if t not in self._K0 or j not in self._K0[t]:
            warn(f'z_{t} {j} = {k}, but {k} not in self._K0')
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
    if k == 0: return self._K0
    else: return self._K[k]
  
  def next_k(self):
    """ Return next valid index of a new unique label. """
    if len(self.ks) == 0: return int(1)
    else: return max(self.ks) + 1

# want fast look-ups of t1, i1, t2, i2 : same/diff
# want convenient consistency / inconsistency counts

class PairwiseAnnotations(jpt.Serializable):
  """ dict N {t: n_t }, list A [ (t1, i1, t2, i2, 1), ... ]. """

  def __init__(self, N, A):
    self.N = N
    self.A = A
    self.__check_consistency()

  def consistencyCounts(self, z):
    assert type(z) == UniqueBijectiveAssociation

    consistent = inconsistent = 0
    for (t1, i1, t2, i2, val) in self.A:
      nonzeroAndEqual = z[t1][i1] == z[t2][i2] and z[t1][i1] != 0
      if val == 1:
        if nonzeroAndEqual: consistent += 1
        else: inconsistent += 1
      else:
        if nonzeroAndEqual: inconsistent += 1
        else: consistent += 1

    return consistent, inconsistent

  def __check_consistency(self):
    assert type(self.A) == list
    for (t1, i1, t2, i2, val) in self.A:
      assert t1 in self.N.keys() and t2 in self.N.keys()
      assert 0 <= i1 and i1 < self.N[t1]
      assert 0 <= i2 and i2 < self.N[t2]
      assert val == 1 or val == 0
      assert t1 != t2

  def serializable(self):
    # return dict with appropriate keys for serialization that does't depend on
    # anything more than basic or numpy types
    return { self._magicKey: [self.N, self.A], 
      self._classKey: 'PairwiseAnnotations'
    }

  def fromSerializable(N, A):
    # return PairwiseAnnotation loaded from args pulled from serialized data
    return PairwiseAnnotations(N, A)
