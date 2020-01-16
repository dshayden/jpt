from context import jpt
import numpy as np, matplotlib.pyplot as plt
import warnings, du
import IPython as ip
import copy
np.set_printoptions(precision=2, suppress=True)

def testSwitchReverse():
  # ifile = 'data/datasets/k22/gt.csv'
  ifile = 'data/datasets/k22/dets.csv'
  tracker = jpt.PointTracker
  o, y, w, z = tracker.init2d(ifile)
  o.param.fixedK = 2

  okf = jpt.kalman.opts(o.param.dy, o.param.dx, F=o.param.F, H=o.param.H)
  param = dict(Q=okf.Q, R=okf.R, mu0=o.prior.mu0)
  w, z = tracker.init_assoc_greedy_dumb(y, param, maxK=o.param.fixedK)
  # w, z = tracker.init_assoc_noise(y, maxK=2)

  ll = tracker.log_joint(o, y, w, z)

  # draw switch sample
  o.param.moveNames = ['switch']
  o.param.moveProbs = np.array([1.0])

  nSamples = 100
  accept = 0
  lls = np.zeros(nSamples)
  lls[0] = ll

  nS = 0
  print(nS, 'init', lls[nS])

  for nS in range(1,nSamples):
    w, z, info = tracker.sample(o, y, w, z, lls[nS-1])
    if info['accept'] == True: lls[nS] = info['ll_prop']
    else: lls[nS] = lls[nS-1]

    logA = info['logA']
    print(f"logA: {logA:.2f}, ll: {lls[nS]:.2f}")
    # print(info)

  assert len(z.to(0).keys()) == 0

    # if info['accept'] == True:
    #   lls[nS] = info['ll_prop']
    #   accept += 1
    #   print(nS, info['move'], lls[nS])
    # else:
    #   lls[nS] = lls[nS-1]


def testNoise():
  warnings.filterwarnings("error")

  data = {1: [[2.,3.], [4.,5.]], 2: [6., 7.]}
  y = jpt.NdObservationSet(data)
  z = jpt.UniqueBijectiveAssociation(y.N, {1: [0,0], 2: [2,]})
  x = jpt.AnyTracks({
    2: ({}, { 2: np.array([6.5, 7.5])} ),
  })

  okf = jpt.kalman.opts(2, 4, Q=10*np.eye(4), R=0.5*np.eye(2) )
  k = 2
  yk = dict([ (t, y[t][j]) for t, j in z.to(k).items() ])
  xk = jpt.kalman.ffbs(okf, {2: None}, yk)

  muX, SX, muY, SY = jpt.kalman.predictive(okf, xk, 0)

  tracker = jpt.PointTracker
  dy, dx = (2, 4)
  kwargs = tracker.data_dependent_priors(y)
  dummy_ifile = ''
  o = tracker.opts(dummy_ifile, dy, dx, **kwargs)

  # Q = o.prior.S0 / (o.prior.df0 - dx - 1)
  Q = np.diag(np.diag(o.prior.S0))
 
  R = 0.1 * Q[:dy,:dy]
  param = dict(Q=Q, R=R, mu0=o.prior.mu0)
  w2, z2 = tracker.init_assoc_greedy_dumb(y, param, maxK=1)

  ll = tracker.log_joint(o, y, w2, z2)

  ip.embed()

def testExtend():
  ifile = 'data/datasets/k22/dets.csv'
  tracker = jpt.PointTracker
  dy, dx = (2, 4)
  y = jpt.io.mot15_bbox_to_obs(ifile)
  kwargs = tracker.data_dependent_priors(y)
  o = tracker.opts(ifile, dy, dx, **kwargs)
  w, z = tracker.init_assoc_noise(y)
  ll = tracker.log_joint(o, y, w, z)

  o.param.ps = 0.5
  o.param.maxK = 1

  w, z, valid, logq = tracker.proposals.gather(o, y, w, z)

  plt.figure()
  jpt.viz.plot_points2d_global(y)
  jpt.viz.plot_tracks2d_global(w)
  plt.title(f'Initial')
  # plt.savefig(f'{savePath}/initial.pdf', bbox_inches='tight')
  # plt.close()

  o.param.ps = 0.03
  w2, z2, valid2, logq2 = tracker.proposals.extend(o, y, w, z)

  plt.figure()
  jpt.viz.plot_points2d_global(y)
  jpt.viz.plot_tracks2d_global(w2)
  plt.title(f'Extend')
  # plt.savefig(f'{savePath}/initial.pdf', bbox_inches='tight')
  # plt.close()
  plt.show()

  ip.embed()

  # assert np.isclose(logq + logq2, 0.0)
  # assert len(w.ks) == len(z.ks) == (1 + len(z2.ks)) == (1 + len(w2.ks))

# good test
def testGather():
  ifile = 'data/datasets/k22/dets.csv'
  tracker = jpt.PointTracker
  dy, dx = (2, 4)
  y = jpt.io.mot15_bbox_to_obs(ifile)
  kwargs = tracker.data_dependent_priors(y)
  o = tracker.opts(ifile, dy, dx, **kwargs)
  w, z = tracker.init_assoc_noise(y)
  ll = tracker.log_joint(o, y, w, z)

  o.param.ps = 0.25
  o.param.maxK = 2

  w, z, valid, logq = tracker.proposals.gather(o, y, w, z)
  w2, z2, valid2, logq2 = tracker.proposals.disperse(o, y, w, z)

  assert np.isclose(logq + logq2, 0.0)
  assert len(w.ks) == len(z.ks) == (1 + len(z2.ks)) == (1 + len(w2.ks))


def testTrackingWithErodedMean():
  # ipath = 'data/datasets/marmoset'
  # ifile = f'{ipath}/recording-2018-11-21_10_16_16-00000000_masks_and_means'
  # ifile = f'{ipath}/recording-2018-11-21_10_16_16-allMasksAndMeans.gz'

  # outDir = f'{ipath}/trackOutput'
  outDir = f'{ipath}/trackOutputAll_01'
  
  tracker = jpt.PointTracker
  # o, yMasks, y, w, z = tracker.init2d_masks(ifile, fixedK=2)
  o, yMasks, y, w, z = tracker.init2d_masks_precomputed_mean(ifile, fixedK=2)

  # imgDir = f'{ipath}/rgb'
  # yImgs = jpt.io.imgs_to_obs(imgDir)

  o.param.lambda_track_length = len(y.ts)
  ll = tracker.log_joint(o, y, w, z)

  o.param.moveNames = ['update', 'split', 'merge', 'switch']
  o.param.moveProbs = np.array([0.2, 0.1, 0.1, 0.6])


  nSamples = 100
  accept = 0
  lls = np.zeros(nSamples)
  lls[0] = ll
  zs = [ z ]
  ws = [ w ]

  # savePath = 'data/datasets/marmoset/samples'
  # savePath = 'data/datasets/marmoset/samples2'
  savePath = 'data/datasets/marmoset/samplesAll_01'

  for nS in range(1,nSamples):
    w, z, info = tracker.sample(o, y, w, z, lls[nS-1])
    if info['accept'] == True:
      lls[nS] = info['ll_prop']
      accept += 1
      print(nS, info['move'], lls[nS])
    else:
      lls[nS] = lls[nS-1]
    zs.append(copy.deepcopy(z))
    ws.append(copy.deepcopy(w))

    # jpt.io.save(f'{savePath}/sample-{nS:05}', {'z': z, 'w': w, 'yMasks': yMasks,
    #   'y': y, 'yImgs': yImgs, 'o': o, 'll': lls[nS]})
    jpt.io.save(f'{savePath}/sample-{nS:05}', {'z': z, 'w': w, 'y': y, 'o': o,
      'll': lls[nS]})


  print(f'Accepted {accept} proposals, final LL is {lls[-1]:.2f}, initial LL was {lls[0]:.2f}')
  # ip.embed()

  # DSH added
  # z, w = ( zs[-1], ws[-1] )
  # cols = du.diffcolors(len(z.ks))
  # def show(t):
  #   plt.imshow(yImgs[t][0])
  #   for k in w[t].keys():
  #     wtk = w[t][k]
  #     plt.scatter(*wtk[:2], color=cols[k])
  # du.ViewPlots(z.ts, show)
  # plt.show()

  # save z, w in usable format
  # savePath = 'data/datasets/marmoset/samples'
  # for nS in range(nSamples):
  #   z, w = ( zs[nS], ws[nS] )
  #   jpt.io.save(f'{savePath}/sample-{nS:05}', {'z': z, 'w': w})

  # jpt.io.save(ofile, {'yMasks': yMasks, 'yMeans': yMeans})
  


  # # look at association spread  
  # zsBurn = zs[nSamples//4:]
  # colors = du.diffcolors(3)
  # nTargets = 2
  # nObs = max(y.N.values())
  #
  # cnts = np.zeros((len(y.ts), nObs, nTargets))
  # for idx, t in enumerate(y.ts):
  #   for z in zsBurn:
  #     for k in [0, 1]:
  #       if t in z.to(k):
  #         j = z.to(k)[t]
  #         cnts[idx, j, k] += 1
  #
  # for idx, t in enumerate(y.ts):
  #   for n in range(nObs):
  #     # get cnts for "observation" j
  #     s = cnts[idx, n]
  #     if np.sum(s) == 0: continue
  #     s = s / np.sum(s)
  #     ctr = np.array([t, n])
  #     plt.pie(s, colors=colors, center=ctr, radius=0.25)
  # plt.xticks([]); plt.xlabel('time')
  # plt.yticks([]);
  # plt.xlim(-1, len(y.ts)+1)
  # plt.scatter([0, len(y.ts)], [0, 2], s=0.01)
  # plt.gca().set_aspect('auto')
  #
  # plt.savefig('test.pdf', bbox_inches='tight')
  # plt.show()
  #
  #
  # jpt.viz.plot_associated_masks_on_images(yImgs, yMasks, z, outDir)



def testErodedMean():
  ipath = 'data/datasets/marmoset'
  ifile = f'{ipath}/recording-2018-11-21_10_16_16-00000000.gz'

  yMasks = jpt.io.masks_to_obs(ifile)
  yMeans = jpt.io.eroded_mean_from_masks(yMasks)

  ofile = f'{ipath}/recording-2018-11-21_10_16_16-00000000_masks_and_means'
  jpt.io.save(ofile, {'yMasks': yMasks, 'yMeans': yMeans})


def testSwitch():
  # ifile = 'data/datasets/k22/gt.csv'
  ifile = 'data/datasets/k22/dets.csv'
  tracker = jpt.PointTracker
  o, y, w, z = tracker.init2d(ifile)
  o.param.fixedK = 2

  okf = jpt.kalman.opts(o.param.dy, o.param.dx, F=o.param.F, H=o.param.H)
  param = dict(Q=okf.Q, R=okf.R, mu0=o.prior.mu0)
  w, z = tracker.init_assoc_greedy_dumb(y, param, maxK=o.param.fixedK)
  # w, z = tracker.init_assoc_noise(y, maxK=2)

  ll = tracker.log_joint(o, y, w, z)

  # draw switch sample
  o.param.moveNames = ['switch']
  o.param.moveProbs = np.array([1.0, 0.5])

  nSamples = 1000
  accept = 0
  lls = np.zeros(nSamples)
  lls[0] = ll

  savePath = 'data/datasets/k22/samples001'

  # save initial sample
  nS = 0
  jpt.io.save(f'{savePath}/sample-{nS:05}', {'z': z, 'w': w,
    'y': y, 'o': o, 'll': lls[nS]})
  print(nS, 'init', lls[nS])

  jpt.viz.plot_points2d_global(y)
  jpt.viz.plot_tracks2d_global(w)
  plt.title(f'Initial Sample, Log Joint: {lls[nS]:.2f}')
  plt.savefig(f'{savePath}/initial.pdf', bbox_inches='tight')
  plt.close()
  
  for nS in range(1,nSamples):
    w, z, info = tracker.sample(o, y, w, z, lls[nS-1])

    if info['accept'] == True:
      lls[nS] = info['ll_prop']
      accept += 1
      print(nS, info['move'], lls[nS])
    else:
      lls[nS] = lls[nS-1]

    # if info['accept'] == True: lls[nS] = info['ll_prop']; accept += 1
    # else: lls[nS] = lls[nS-1]

    jpt.io.save(f'{savePath}/sample-{nS:05}', {'z': z, 'w': w,
      'y': y, 'o': o, 'll': lls[nS]})

    # if nS % 20 == 0:
    #   jpt.viz.plot_points2d_global(y)
    #   jpt.viz.plot_tracks2d_global(w)
    #   plt.show()
    

  print(f'Accepted {accept} proposals, final LL is {lls[-1]:.2f}, initial LL was {lls[0]:.2f}')

  jpt.viz.plot_points2d_global(y)
  jpt.viz.plot_tracks2d_global(w)
  plt.title(f'Final Sample, Log Joint: {lls[nS]:.2f}')
  plt.savefig(f'{savePath}/final.pdf', bbox_inches='tight')
  plt.close()


  # jpt.viz.plot_points2d_global(y)
  # jpt.viz.plot_tracks2d_global(w)
  # plt.show()



  # w_, z_, info = tracker.sample(o, y, w, z, ll)
  # print(info)



def testSampling():
  ifile = 'data/datasets/k22/dets.csv'
  # ifile = 'data/datasets/k22/gt.csv'
  tracker = jpt.PointTracker

  # init style 1
  # o, y, w, z = tracker.init2d(ifile)
  # o.param.lambda_track_length = len(y.ts)
  # o.param.ps = 0.5
  # ll = tracker.log_joint(o, y, w, z)

  # init style 2
  dy, dx = (2, 4)
  y = jpt.io.mot15_bbox_to_obs(ifile)
  kwargs = tracker.data_dependent_priors(y)
  o = tracker.opts(ifile, dy, dx, **kwargs)
  w, z = tracker.init_assoc_noise(y)
  ll = tracker.log_joint(o, y, w, z)

  nSamples = 500
  # nSamples = 50
  accept = 0
  lls = np.zeros(nSamples)
  lls[0] = ll
  
  for nS in range(1,nSamples):
    w, z, info = tracker.sample(o, y, w, z, lls[nS-1])
    if info['accept'] == True:
      lls[nS] = info['ll_prop']
      accept += 1
      print(f"Sample {nS:05}, Move: {info['move']}, ll: {lls[nS]:06.2f}, K: {len(z.ks):03}")

      # print(nS, info['move'], lls[nS])
    else:
      # if info['move'] == 'gather':
      #   ip.embed()
      lls[nS] = lls[nS-1]

  print(f'Accepted {accept} proposals, final LL is {lls[-1]:.2f}, initial LL was {lls[0]:.2f}')
  jpt.viz.plot_points2d_global(y)
  jpt.viz.plot_tracks2d_global(w)
  
  plt.figure()
  plt.plot(lls)
  plt.show()

def testPointInit():
  ifile = 'data/datasets/k22/gt.csv'
  # ifile = 'data/datasets/k22/dets.csv'
  tracker = jpt.PointTracker
  o, y, w, z = tracker.init2d(ifile)

  ll = tracker.log_joint(o, y, w, z)

  savePath = 'data/datasets/k22/true_sample'
  jpt.io.save(f'{savePath}/true_sample', {'z': z, 'w': w, 'y': y, 'o': o,
    'll': ll})

  # o.param.moveNames = ['split']
  # o.param.moveProbs = np.array([1.0,])

  plt.figure()
  jpt.viz.plot_points2d_global(y)
  jpt.viz.plot_tracks2d_global(w)

  w_, z_, info = tracker.sample(o, y, w, z, ll)
  print(info['accept'])
  print(info['ll_prop'])

  plt.figure()
  jpt.viz.plot_points2d_global(y)
  jpt.viz.plot_tracks2d_global(w_)
  plt.title('after sample')
  plt.show()


  # ip.embed()




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
