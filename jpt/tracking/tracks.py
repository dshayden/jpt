import jpt
import copy
from abc import ABC, abstractmethod

class Tracks(ABC):
  """ Set of tracks with per-track and per-time parameters.

  Tracks DOES NOT store data associations and is not aware of any
  ObservationSet.
  """
  def __init__(self):
    self.t0, self.tE, self.ts, self.ks = (-1, -1, [], [])

  @abstractmethod
  def __getitem__(self, t):
    """ Get all target latent states at time t. """
    pass

import numpy as np
class AnyTracks(Tracks, jpt.Serializable):
  """ Tracks with per-track and per-time data of any type. """

  def __init__(self, x):
    """ Construct AnyTracks from dictionary x of specific form:
    
      x must be of the form { k: ( {theta: v, ...}, {t: x_tk, ...}, ), ... }
        k, t must be integers
        x[k] is 2-tuple of dicts
        x[k][0] is dict of per-track parameters, keys and vals can be any type
        x[k][1] is dict of {t: x_tk} where x_tk can be any type
    """
    super().__init__()
    self.x = x
    self._update()

  def _update(self):
    ts = [ ]
    for k, kTuple in self.x.items():
      assert type(k) == int
      assert len(kTuple) == 2
      assert isinstance(kTuple[0], dict) and isinstance(kTuple[1], dict)
      for t, v in kTuple[1].items(): assert type(t) == int
      ts += [ t for t in kTuple[1].keys() ]
    self.ts = sorted(np.unique(ts))
    self.ks = sorted(self.x.keys())
    self.t0, self.tE = ( min(self.ts), max(self.ts) )

  def edit(self, e, kind='k', inplace=False):
    w = self if inplace else copy.deepcopy(self)
    if kind == 'k':
      for k, v in e.items():
        if v is None: del w.x[k]; continue

        theta_, x_tks_ = v # new
        if k in w.x:
          theta, x_tks = w.x[k] # old
          for t, x_tk_ in x_tks_.items():
            assert type(t) == int
            if x_tk_ is None:
              if t in x_tks: del x_tks[t]
              # del x_tks[t]
            else: x_tks[t] = x_tk_
          theta = theta_
        else:
          theta, x_tks = ( theta_, {} )
          for t, x_tk_ in x_tks_.items():
            assert type(t) == int
            assert x_tk_ is not None, 'None with newly created component.'
            x_tks[t] = x_tk_
          w.x[k] = ( theta, x_tks )

    w._update()
    return w

  def __getitem__(self, t):
    """ return dict of {k: x_tk} at time t """
    return dict([ (k, self.x[k][1][t]) for k in self.ks if t in self.x[k][1] ])
  
  def serializable(self):
    # return dict with appropriate keys for serialization that does't depend on
    # anything more than basic or numpy types
    return { self._magicKey: [self.x,], self._classKey: 'AnyTracks' }

  def fromSerializable(x):
    # return AnyTracks loaded from args pulled from serialized data
    return AnyTracks(x)
