import argparse, du
import matplotlib.pyplot as plt
import IPython as ip
import numpy as np
import jpt

def work(sampleFile):
  D = jpt.io.load(sampleFile)
  return D['z']

def main(args):
  sampleFiles = du.GetFilePaths(args.sampleDir, 'gz')
  if args.tE < 0: sampleFiles = sampleFiles[args.t0:]
  else: sampleFiles = sampleFiles[args.t0:args.tE]

  z_ = du.Parfor(work, sampleFiles)
  y = jpt.io.load(sampleFiles[0])['y']

  # get all unique target identifiers k
  # ks = np.unique( [ z_[s].ks for s in range(len(z_)) ] )
  ks = np.unique(np.concatenate(( [ z_[s].ks for s in range(len(z_)) ] )))
  nTargets = len(ks)
  colors = du.diffcolors(nTargets)
  nObs = max(y.N.values())

  cnts = np.zeros((len(y.ts), nObs, nTargets))
  for idx, t in enumerate(y.ts):
    for z in z_:
      for k in ks:
        if k not in z.ks: continue
        if t in z.to(k):
          j = z.to(k)[t]
          cnts[idx, j, k] += 1

  for idx, t in enumerate(y.ts):
    for n in range(nObs):
      # get cnts for "observation" j
      s = cnts[idx, n]
      if np.sum(s) == 0: continue
      s = s / np.sum(s)
      ctr = np.array([t, n])
      # ctr = y[t][n]
      plt.pie(s, colors=colors, center=ctr, radius=0.25)
  # plt.xticks(y.ts)
  plt.xticks([]);
  plt.xticks(range(min(y.ts), max(y.ts)+100, 100), fontsize=4, rotation=90)

  plt.xlabel('Time')
  plt.yticks([]);

  # for t in y.ts:
  #   for n in range(y[t].shape[0]):
  #     plt.scatter(*y[t][n].T, s=0.1, color='k')

  # plt.xlim(-1, 51)
  # plt.ylim(-1, 13)

  plt.xlim(-1, len(y.ts)+1)
  plt.ylim(min(ks)-1, max(ks)+1)
  # plt.scatter([0, len(y.ts)], [0, 2], s=0.01)
  plt.gca().set_aspect('equal')

  plt.savefig('test.pdf', bbox_inches='tight')
  # plt.show()



  # plot observations and pie charts
  # ip.embed() 


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('sampleDir', type=str, help='directory with jpt samples')
  # parser.add_argument('outfile', type=str, help='output mat file')
  parser.add_argument('--t0', type=int, default=0, help='first sample index')
  parser.add_argument('--tE', type=int, default=-1, help='last sample index')
  parser.set_defaults(func=main)

  args = parser.parse_args()
  args.func(args)
