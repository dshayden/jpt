import numpy as np, jpt, matplotlib.pyplot as plt, du, os
import scipy.spatial.distance, scipy.optimize
np.set_printoptions(suppress=True, precision=2)
import IPython as ip, sys

# get last sample in each desired folder
maskFolder = '/Users/dshayden/Downloads/behavior_dets/dets'
# sampleFolder = '/Users/dshayden/Downloads/behavior_dets/dets/tracks'
# sampleFolder = '/Users/dshayden/Research/code/jpt/tmp/k2_test'
sampleFolder = '/Users/dshayden/Research/code/jpt/tmp/k2_test002'
videoFolder = '/Users/dshayden/Downloads/behavior_dets/video'
# outDir = '/Users/dshayden/Downloads/behavior_dets/draw'
# outDir = '/Users/dshayden/Research/code/jpt/tmp/k2_test_draw'
outDir = '/Users/dshayden/Research/code/jpt/tmp/k2_test002_draw_b'

folders = sorted([folder[0] for folder in os.walk(sampleFolder)])
# this is 0:5 in files, but first file is '.' ; todo: set 0:5 as parameter
# hard-coded
# folders = folders[1:6]
# folders = folders[1:3] # just get first two
folders = folders[2:3] # just get first two
sampleFiles = [ du.GetFilePaths(folder, 'gz')[-1] for folder in folders ]
samples = [ jpt.io.load(sampleFile) for sampleFile in sampleFiles ]

# for each sample, get ks that have are observed at least 66% of the time
# get the first and last latent state for each
trackBegins = [ [] for i in range(len(samples)) ]
trackEnds = [ [] for i in range(len(samples)) ]
trackInds = [ [] for i in range(len(samples)) ]

for idx, sample in enumerate(samples):
  T = len(sample['z'].ts)
  for k in sample['z'].ks:
    toK = sample['z'].to(k)
    if (len(toK) / T) < 0.33: continue

    # get first, last track state
    ts_sorted = list(toK.keys())
    t0 = ts_sorted[np.argmin(ts_sorted)]
    tE = ts_sorted[np.argmax(ts_sorted)]
    w = sample['w']
    x0 = w.x[k][1][t0]
    xE = w.x[k][1][tE]

    H = sample['o'].param.H

    trackInds[idx].append(k)
    trackBegins[idx].append( H @ x0 )
    trackEnds[idx].append( H @ xE )

trackBegins = [ np.array(trackBegins[i]) for i in range(len(samples)) ]
trackEnds = [ np.array(trackEnds[i]) for i in range(len(samples)) ]
trackInds = [ np.array(trackInds[i]) for i in range(len(samples)) ]


# establish dictionary of (segment, k, trueIndex)

# all tracks in first segment map to themselves
trackMaps = { }
for k in trackInds[0]: trackMaps[ (0, k) ] = k

for i in range(len(samples)-1):
  cost = scipy.spatial.distance.cdist(trackEnds[i], trackBegins[i+1])
  rows, cols = scipy.optimize.linear_sum_assignment(cost)

  # vertex i (row) matched to vertex j (col)
  #    row[l] is matched to col[l]
  # => trackInds[i][row[l]] matched to trackInds[i][col[l]]
  for l in range(len(rows)):
    print(f'{trackInds[i][rows[l]]} matched to { trackInds[i+1][cols[l]] }')

  for l in range(len(rows)):
    prevMap = trackInds[i][rows[l]]
    nextMap = trackInds[i+1][cols[l]]

    trackMaps[ (i+1, nextMap) ] = trackMaps[ (i, prevMap) ]
    # trackMaps[ (i+1, nextMap) ] = prevMap

# am now able to draw tracks
# get list of images for each sample
videoImgDirNames = [ f'{videoFolder}/{du.fileparts(du.fileparts(sampleFile)[0])[1]}'
  for sampleFile in sampleFiles ]

# these are images
videoImgs = [ jpt.io.imgs_to_obs(vid, 1500*i)
  for i, vid in enumerate(videoImgDirNames) ]

# 0:5 is hard-coded here, corresponds to above folders index
# maskFiles = du.GetFilePaths(maskFolder, 'gz')[0:5]
# maskFiles = du.GetFilePaths(maskFolder, 'gz')[0:2]
maskFiles = du.GetFilePaths(maskFolder, 'gz')[1:2]

masks = [ jpt.io.masks_to_obs(mf, 1500*i)
  for i, mf in enumerate(maskFiles) ]

# I have sampled associations, masks and images; draw them
colors = du.diffcolors(1+len(np.unique(list(trackMaps.values()))), alpha=0.5)


for nS in range(len(samples)):
  for idx, t in enumerate(videoImgs[nS].ts):
    # if t % 10 != 0: continue
    print(nS, t)
    z = samples[nS]['z']

    im = videoImgs[nS][t][0]
    mt, st = masks[nS][t]
    for j in range(z.N[t]):
      if (nS, z[t][j]) not in trackMaps: continue
      mappedK = trackMaps[ (nS, z[t][j]) ]

      mtj = mt[j]
      color = colors[ mappedK ]
      im = du.DrawOnImage(im, np.where(mtj > 0), color)
    du.imwrite(im, f'{outDir}/img-{t:08d}.jpg')
