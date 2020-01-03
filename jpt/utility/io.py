import jpt
from abc import ABC, abstractmethod
import numpy as np, pandas as pd
import cv2
from pycocotools import mask as pcMask
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

def point_tracks_to_mot15_point2d(fname, w):
  fid = open(fname, 'w')
  for t in w.ts:
    for k, xtk in w[t].items():
      fs = f'{t}, {k}, {xtk[0]:.2f}, {xtk[1]:.2f}, -1, -1, 1.0, -1, -1, -1'
      print(fs, file=fid)
  fid.close()

def masks_to_obs(fname, startTime=0):
  """ Load pycocotools masks to MaskObservationSet. """
  masks = du.load(fname)
  yMasks = dict([ (t+startTime, masks[t]) for t in range(len(masks))])
  return jpt.MaskObservationSet(yMasks)

def imgs_to_obs(imgDir, startTime=0):
  """ Load an image directory to an observation set, one for each time. """
  imgPaths = du.GetImgPaths(imgDir)
  T, N_t = ( len(imgPaths), 1 )
  yDict = { t + startTime: [ imgPaths[t], ] for t in range(T) }
  return jpt.ImageObservationSet(yDict)

def mean_from_masks(yMasks):
  """ Construct NdObservationSet from MaskObservationSet using Mean """

  def mask_mean(mask):
    ys, xs = np.nonzero(mask)
    return np.array([ np.mean(xs), np.mean(ys) ])

  def means_t(t):
    masks, _ = yMasks[t]
    means = np.zeros((yMasks.N[t], 2))
    for j in range(yMasks.N[t]):
      mask = masks[j]
      means[j] = mask_mean(mask)
    return (t, means)

  means = du.For(means_t, yMasks.ts, showProgress=True)
  yMeans = jpt.NdObservationSet(dict(means))
  return yMeans

def eroded_mean_from_masks(yMasks):
  """ Construct NdObservationSet from MaskObservationSet using Eroded Mean """  

  def eroded_mean(mask):
    means, k = ( [], 5 )
    for i in range(10):
      m = cv2.erode(mask, np.ones((k, k), dtype=np.uint8))
      y, x = np.nonzero(m)
      if len(y) == 0: break
      pts = np.stack((x, y)).T
      means.append(np.mean(pts, axis=0))
      k *= 2
    return means[-1]

  def means_t(t):
    masks, _ = yMasks[t]
    means = np.zeros((yMasks.N[t], 2))
    for j in range(yMasks.N[t]):
      mask = masks[j]
      means[j] = eroded_mean(mask)
    return (t, means)

  means = du.For(means_t, yMasks.ts, showProgress=True)
  yMeans = jpt.NdObservationSet(dict(means))
  return yMeans


__fromBytesObjects = {
  'NdObservationSet': jpt.NdObservationSet,
  'ImageObservationSet': jpt.ImageObservationSet,
  'MaskObservationSet': jpt.MaskObservationSet,
  'UniqueBijectiveAssociation': jpt.UniqueBijectiveAssociation,
  'AnyTracks': jpt.AnyTracks
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
