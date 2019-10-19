import jpt
from abc import ABC, abstractmethod
import numpy as np, pandas as pd
import du

__mot2015 = ['Frame', 'ID', 'BBx', 'BBy', 'BBw', 'BBh', 'Score', 'x', 'y', 'z']

# Load just 2D bbox corner points from MOT 2015 formatted file
def mot15_point2d_to_obs(fname):
  df = pd.read_csv(fname, names=__mot2015)
  y = {}
  uniq = df.Frame.unique()
  for idx, t in enumerate(uniq):
    dft = df[df.Frame == t]
    obs = dft[ ['BBx', 'BBy'] ].values
    y[t] = obs
  return jpt.NdObservationSet(y)

# Load 2D bbox from MOT 2015 formatted file
def mot15_bbox_to_obs(fname):
  df = pd.read_csv(fname, names=__mot2015)
  y = {}
  uniq = df.Frame.unique()
  for idx, t in enumerate(uniq):
    dft = df[df.Frame == t]
    obs = dft[ ['BBx', 'BBy', 'BBw', 'BBh'] ].values
    y[t] = obs
  return jpt.NdObservationSet(y)

def mot15_point2d_to_assoc_unique(fname):
  df = pd.read_csv(fname, names=__mot2015)
  y, z = ( {}, {} )

  uniq = df.Frame.unique()
  for idx, t in enumerate(uniq):
    dft = df[df.Frame == t]
    obs = dft[ ['ID', 'BBx', 'BBy'] ].values
    y[t] = obs[:,1:]
    z[t] = obs[:,0].astype(np.int)
  y = jpt.NdObservationSet(y)

  uniqK = df.ID.unique()
  nextK = max(uniqK)+1
  for t in z.keys():
    for j in range(len(z[t])):
      if z[t][j] == -1: z[t][j] = nextK; nextK += 1
  
  z = jpt.UniqueBijectiveAssociation(y.N, z)
  return y, z

def point_hypothesis_to_mot15_point2d(fname, w):
  fid = open(fname, 'w')
  for t in w.ts:
    for k, xtk in w[t].items():
      fs = f'{t}, {k}, {xtk[0]:.2f}, {xtk[1]:.2f}, -1, -1, 1.0, -1, -1, -1'
      print(fs, file=fid)
  fid.close()

def masks_to_obs(fname, startTime=0):
  masks = du.load(fname)
  yMasks = dict([ (t+startTime, masks[t]) for t in range(len(masks))])
  return jpt.MaskObservationSet(yMasks)

def imgs_to_obs(imgDir, startTime=0):
  """ Load an image directory to an observation set, one for each time. """
  imgPaths = du.GetImgPaths(imgDir)
  T, N_t = ( len(imgPaths), 1 )
  yDict = { t + startTime: [ imgPaths[t], ] for t in range(T) }
  return jpt.ImageObservationSet(yDict)

__fromBytesObjects = {
  'NdObservationSet': jpt.NdObservationSet,
  'ImageObservationSet': jpt.ImageObservationSet,
  'MaskObservationSet': jpt.MaskObservationSet,
  'UniqueBijectiveAssociation': jpt.UniqueBijectiveAssociation,
  'PointHypothesis': jpt.PointHypothesis
}

def save(fname, D):
  assert isinstance(D, dict)
  D_enc = { }
  for k, v in D.items():
    if isinstance(v, jpt.Serializable): D_enc[k] = v.serializable()
    else: D_enc[k] = v
  du.save(fname, D_enc)

def load(fname):
  D_enc = du.load(fname)
  D_dec = { }
  for k, v in D_enc.items():
    if isinstance(v, dict) and jpt.Serializable._magicKey in v:     
      fcn = __fromBytesObjects[ v[jpt.Serializable._classKey] ]
      args = v[jpt.Serializable._magicKey]
      D_dec[k] = fcn.fromSerializable(*args)
    else:
      D_dec[k] = v

  return D_dec
