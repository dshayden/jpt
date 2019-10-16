from abc import ABC, abstractmethod

class ObservationSet(ABC):
  """ Batch collection of observations. """

  def __init__(self):
    self.t0, self.tE, self.N, self.ts = (-1, -1, {}, [])

  @abstractmethod
  def __getitem__(self, t):
    """ Get all observations at time t. """
    pass

import numpy as np
class PointObservationSet(ObservationSet):
  """ Collections of variable quantity of D-dim points at each time t. """

  def __init__(self, ys):
    """ Initialize from dictionary ys. """
    super().__init__()
    self._y = dict([ (int(t), np.atleast_2d(yt)) for t, yt in ys.items() ])
    self.t0, self.tE = ( min(self._y.keys()), max(self._y.keys()) )
    self.N = dict([ (t, yt.shape[0]) for t, yt in self._y.items() ])
    self.ts = sorted(self._y.keys())

  def __getitem__(self, t):
    """ Get all observations at time t. """
    return self._y[t]
