import jpt
import numpy as np, pandas as pd

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
  return jpt.PointObservationSet(y)

# Load 2D bbox from MOT 2015 formatted file
def mot15_bbox_to_obs(fname):
  df = pd.read_csv(fname, names=__mot2015)
  y = {}
  uniq = df.Frame.unique()
  for idx, t in enumerate(uniq):
    dft = df[df.Frame == t]
    obs = dft[ ['BBx', 'BBy', 'BBw', 'BBh'] ].values
    y[t] = obs
  return jpt.PointObservationSet(y)

def mot15_point2d_to_assoc_unique(fname):
  df = pd.read_csv(fname, names=__mot2015)
  y, z = ( {}, {} )

  uniq = df.Frame.unique()
  for idx, t in enumerate(uniq):
    dft = df[df.Frame == t]
    obs = dft[ ['ID', 'BBx', 'BBy'] ].values
    y[t] = obs[:,1:]
    z[t] = obs[:,0].astype(np.int)
  y = jpt.PointObservationSet(y)

  uniqK = df.ID.unique()
  nextK = max(uniqK)+1
  for t in z.keys():
    for j in range(len(z[t])):
      if z[t][j] == -1: z[t][j] = nextK; nextK += 1
  
  z = jpt.UniqueBijectiveAssociation(y, z)
  return y, z

def point_hypothesis_to_mot15_point2d(fname, w):
  fid = open(fname, 'w')
  for t in w.ts:
    for k, xtk in w[t].items():
      fs = f'{t}, {k}, {xtk[0]:.2f}, {xtk[1]:.2f}, -1, -1, 1.0, -1, -1, -1'
      print(fs, file=fid)
  fid.close()
