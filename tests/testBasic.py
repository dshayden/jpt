from context import jpt
import numpy as np, matplotlib.pyplot as plt
import warnings
import IPython as ip

def testMask():
  fname = '/Users/dshayden/Research/code/misc/marmoset_shape/marm_dets.gz'
  y = jpt.io.masks_to_obs(fname)
  jpt.io.save('test', {'y': y})
  y_ = jpt.io.load('test')['y']

  ip.embed()

def testViz():
  fname = 'data/datasets/k22/gt.csv'
  y, z = jpt.io.mot15_point2d_to_assoc_unique(fname)
  ip.embed()
  jpt.viz.plot_points2d_global(y)

  x = { }
  for k in z.ks:
    for t, j in z.to(k).items():
      if k not in x: x[int(k)] = {}
      x[int(k)][t] = y[t][j]
  w = jpt.PointHypothesis(x)

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
      if k not in x: x[int(k)] = {}
      x[int(k)][t] = y2[t][j]
 
  w = jpt.PointHypothesis(x)
  # jpt.io.point_hypothesis_to_mot15_point2d('test.csv', w)

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
  y = jpt.PointObservationSet(data)
  z = jpt.UniqueBijectiveAssociation(y.N, {1: [2,3], 2: [3,]})
  x = jpt.PointHypothesis({
    2: { 1: np.array([2.25, 2.75]) },
    3: { 1: np.array([4.25, 5.25]), 2: np.array([6.25, 7.25]) }
  })

  okf = jpt.kalman.opts(2, 4, Q=10*np.eye(4), R=0.1*np.eye(2) )
  k = 3
  yk = dict([ (t, y[t][j]) for t, j in z.to(k).items() ])
  xk = jpt.kalman.ffbs(okf, {1: None, 2: None}, yk)

  muX, SX, muY, SY = jpt.kalman.predictive(okf, xk, 0)

  ip.embed()
