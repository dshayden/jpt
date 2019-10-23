from context import jpt
import numpy as np, matplotlib.pyplot as plt
import warnings, du
import IPython as ip
np.set_printoptions(precision=2, suppress=True)

def testSampling():
  ifile = 'data/datasets/k22/dets.csv'
  tracker = jpt.PointTracker
  o, y, w, z = tracker.init2d(ifile)
  ll = tracker.log_joint(o, y, w, z)

  nSamples = 100
  accept = 0
  lls = np.zeros(nSamples)
  lls[0] = ll
  
  for nS in range(1,nSamples):
    w, z, info = tracker.sample(o, y, w, z, lls[nS-1])
    if info['accept'] == True: lls[nS] = info['ll_prop']; accept += 1
    else: lls[nS] = lls[nS-1]

  print(f'Accepted {accept} proposals, final LL is {lls[-1]:.2f}, initial LL was {lls[0]:.2f}')
  jpt.viz.plot_points2d_global(y)
  jpt.viz.plot_tracks2d_global(w)
  plt.show()

def testPointInit():
  # ifile = 'data/datasets/k22/gt.csv'
  ifile = 'data/datasets/k22/dets.csv'
  tracker = jpt.PointTracker
  o, y, w, z = tracker.init2d(ifile)

  ll = tracker.log_joint(o, y, w, z)

  # o.param.moveNames = ['split']
  # o.param.moveProbs = np.array([1.0,])

  plt.figure()
  jpt.viz.plot_points2d_global(y)
  jpt.viz.plot_tracks2d_global(w)

  w_, z_, info = tracker.sample(o, y, w, z, ll)
  print(info['accept'])
  print(info['ll_prop'])

  # w_, z_, ll_, move, accepted = tracker.sample(o, y, w, z, ll)
  # w_, z_, accepted, ll_ = tracker.sample(o, y, w, z, ll)

  # print(accepted)
  # print(ll_)

  plt.figure()
  jpt.viz.plot_points2d_global(y)
  jpt.viz.plot_tracks2d_global(w_)

  ip.embed()




  # jpt.viz.plot_points2d_global(y)
  # jpt.viz.plot_tracks2d_global(w)
  # plt.show()
  # ip.embed()

def testHMM():
  fname = 'data/datasets/k22/gt.csv'
  y, z = jpt.io.mot15_point2d_to_assoc_unique(fname)
  
  def val(t, k):
    j = z.to(k)[t]
    x = y_ = y[t][j]
    return x, y_, j

  def costxx(t1, x1, t2, x2, k): return -1.0
  def costxy(t, k, x, y): return -1.0

  ks = z.ks
  t0 = y.ts[0] * np.ones_like(ks)
  ts = y.ts[1:]
  perms, pi, Psi, psi = jpt.hmm.build(t0, ts, ks, val, costxx, costxy)

  ip.embed()


def testMerges():
  fname = 'data/datasets/k22/dets.csv'
  y, z = jpt.io.mot15_point2d_to_assoc_unique(fname)
  # fname = 'data/datasets/k22/gt.csv'
  # y, z = jpt.io.mot15_point2d_to_assoc_unique(fname)
  # merges = jpt.proposals.possible_merges(z)
  # splits = jpt.proposals.possible_splits(z)
  z_, valid, logq = jpt.proposals.merge(z)
  z__, valid, logq__ = jpt.proposals.split(z_)
  assert np.isclose(logq, -1*logq__)
  
  ip.embed()

def testImage():
  imgDir = '/Users/dshayden/Research/data/marmoset_segmentations/curated/images'
  y = jpt.io.imgs_to_obs(imgDir)
  du.save('test', {'y': y})
  y_ = du.load('test')['y']
  
def testMask():
  fname = '/Users/dshayden/Research/code/misc/marmoset_shape/marm_dets.gz'
  y = jpt.io.masks_to_obs(fname)
  jpt.viz.plot_masks(y)
  plt.show()

  jpt.io.save('test', {'y': y})
  y_ = jpt.io.load('test')['y']
  ip.embed()

def testViz():
  fname = 'data/datasets/k22/gt.csv'
  y, z = jpt.io.mot15_point2d_to_assoc_unique(fname)
  jpt.viz.plot_points2d_global(y)

  x = { }
  for k in z.ks:
    for t, j in z.to(k).items():
      if k not in x: x[int(k)] = ( {}, {} )
      x[int(k)][1][t] = y[t][j]
  w = jpt.AnyTracks(x)

  jpt.viz.plot_tracks2d_global(w)

  plt.show()

def testIO():
  fname = 'data/datasets/k22/dets.csv'
  y_ = jpt.io.mot15_point2d_to_obs(fname)
  y, z = jpt.io.mot15_point2d_to_assoc_unique(fname)

  yb = jpt.io.mot15_bbox_to_obs(fname)
  for t in yb.ts:
    for j in range(yb.N[t]):
      assert np.all(np.isclose( yb[t][j][2:4], -1*np.ones(2)))

  fname2 = 'data/datasets/k22/gt.csv'
  y2, z2 = jpt.io.mot15_point2d_to_assoc_unique(fname2)
  x = { }
  for k in z2.ks:
    for t, j in z2.to(k).items():
      if k not in x: x[int(k)] = ( {}, {} )
      x[int(k)][1][t] = y2[t][j]
 
  w = jpt.AnyTracks(x)
  # jpt.io.point_tracks_to_mot15_point2d('test.csv', w)

  # test save
  # a = {1: 'a', '2': 'b'}
  # jpt.io.save('test_save', a)
  # a_ = jpt.io.load('test_save')

  jpt.io.save('test_save', {'w': w})
  w_ = jpt.io.load('test_save')['w']

  jpt.io.save('test_save', {'y': y})
  y_ = jpt.io.load('test_save')['y']

  jpt.io.save('test_save', {'z': z})
  z_ = jpt.io.load('test_save')['z']

  ip.embed()

def testMisc():
  warnings.filterwarnings("error")

  # data = {1: np.zeros((2,2)), 2: np.ones((2,3))}
  data = {1: [[2.,3.], [4.,5.]], 2: [6., 7.]}
  y = jpt.NdObservationSet(data)
  z = jpt.UniqueBijectiveAssociation(y.N, {1: [2,3], 2: [3,]})
  x = jpt.AnyTracks({
    2: ({}, { 1: np.array([2.25, 2.75])} ),
    3: ({}, { 1: np.array([4.25, 5.25]), 2: np.array([6.25, 7.25])} )
  })

  okf = jpt.kalman.opts(2, 4, Q=10*np.eye(4), R=0.1*np.eye(2) )
  k = 3
  yk = dict([ (t, y[t][j]) for t, j in z.to(k).items() ])
  xk = jpt.kalman.ffbs(okf, {1: None, 2: None}, yk)

  muX, SX, muY, SY = jpt.kalman.predictive(okf, xk, 0)

  ip.embed()
