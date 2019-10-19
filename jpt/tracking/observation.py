from abc import ABC, abstractmethod
import du
import jpt

class ObservationSet(ABC):
  """ Batch collection of observations. """

  def __init__(self):
    self.t0, self.tE, self.N, self.ts = (-1, -1, {}, [])

  @abstractmethod
  def __getitem__(self, t):
    """ Get all observations at time t. """
    pass

import numpy as np
class NdObservationSet(ObservationSet, jpt.Serializable):
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

  def serializable(self):
    return { self._magicKey: [self._y,], self._classKey: 'NdObservationSet' }

  def fromSerializable(_y):
    return NdObservationSet(_y)

from pycocotools import mask as pcMask
class MaskObservationSet(ObservationSet, jpt.Serializable):
  """ Collections of masks at each time t. """

  def __init__(self, ys):
    """ Initialize from dictionary ys. """
    super().__init__()
    self._y = dict([ (int(t), yt) for t, yt in ys.items() ])
    self.t0, self.tE = ( min(self._y.keys()), max(self._y.keys()) )
    self.N = dict([ (t, len(yt['masks'])) for t, yt in self._y.items() ])
    self.ts = sorted(self._y.keys())

  def __getitem__(self, t):
    """ Get all observations at time t. """
    masks_enc, scores = ( self._y[t]['masks'], self._y[t]['scores'] )
    if len(scores) == 0: return np.zeros((0, 0, 0), dtype=np.uint8), np.zeros(0)
    masks = np.transpose(pcMask.decode(masks_enc), (2, 0, 1))
    return masks, scores

  def serializable(self):
    return { self._magicKey: [self._y,], self._classKey: 'MaskObservationSet' }

  def fromSerializable(_y):
    return MaskObservationSet(_y)

from pycocotools import mask as pcMask
class ImageObservationSet(ObservationSet, jpt.Serializable):
  """ Collections of images at each time t.

  Stores image obserations as a list of image filepaths at each time, t. Loads
  the images for each time into memory on demand (with no caching)
  """

  def __init__(self, ys):
    """ Initialize from dictionary ys. """
    super().__init__()
    self._y = dict([ (int(t), yt) for t, yt in ys.items() ])
    self.t0, self.tE = ( min(self._y.keys()), max(self._y.keys()) )
    self.N = dict([ (t, len(yt)) for t, yt in self._y.items() ])
    self.ts = sorted(self._y.keys())

  def __getitem__(self, t):
    """ Get all observations at time t. """
    imgList = self._y[t]
    imgs_t = du.For(du.imread, imgList)
    return imgs_t

  def serializable(self):
    return { self._magicKey: [self._y,], self._classKey: 'ImageObservationSet' }

  def fromSerializable(_y):
    return ImageObservationSet(_y)
