import argparse, du
import IPython as ip
import numpy as np
import json
import jpt

def work(sampleFile):
  # return x, z
  D = jpt.io.load(sampleFile)

  # associations z[t][j] == k <=> Object k is associated to y[t][j]
  # z = D['z']._z

  w, y, z, o = ( D['w'], D['y'], D['z'], D['o'] ) 

  ll = D['ll']
  # ########## Resample x | y, z to get log p(x | y, z) ##########
  # # code mostly copied from pointracker/proposals.py
  # editDict_k = {}
  # ll_x = 0.0
  # for k in w.ks:
  #   # build x, y dictionary for kalman.ffbs
  #   zk = z.to(k)
  #   ts_k = zk.keys()
  #
  #   xk = dict([(t, None) for t in ts_k])
  #   # xk = dict([(t, None) for t in y.ts])
  #   yk = dict([(t, y[t][zk[t]]) for t in ts_k])
  #   x0 = ( o.prior.mu0, o.prior.Sigma0 )
  #   theta = w.x[k][0]
  #   okf = jpt.kalman.opts(o.param.dy, o.param.dx, F=o.param.F, H=o.param.H,
  #     Q=theta['Q'], R=theta['R'])
  #   
  #   x_, ll = jpt.kalman.ffbs(okf, xk, yk, x0, return_ll=True)
  #   ll_x += ll
  #
  #   # impute missing values in x (only between beginning and end times of track)
  #   minT = min(ts_k)
  #   maxT = max(ts_k)
  #   x_impute = { t : ( x_[t] if t in x_ else None ) for t in y.ts if t >= minT and t <= maxT }
  #   assert np.setdiff1d(list(x_impute.keys()), range(1500)).size == 0
  #   x_imputed = jpt.kalman.state_estimate(okf, x_impute)
  #
  #   ## add x_imputed keys back to x_impute
  #   for t in x_imputed: x_impute[t] = x_imputed[t][0]
  #   x_ = x_impute
  #   # end impute
  #
  #   editDict_k[k] = ( theta, x_ )
  #
  # ## 3: edit w
  # w = w.edit(editDict_k, kind='k', inplace=False)
  # ########## End Resample x | y, z to get log p(x | y, z) ##########


  # trajectories x
  # x = [ D['w'].x[k][1] for k in D['w'].ks ]
  # x = { k: D['w'].x[k][1] for k in D['w'].ks }
  x = { k: w.x[k][1] for k in w.ks }

  # return x, D['z']._z, ll_x
 
  return x, D['z']._z, ll

def main(args):
  sampleFiles = du.GetFilePaths(args.sampleDir, 'gz')
  if args.tE < 0: sampleFiles = sampleFiles[args.t0:]
  else: sampleFiles = sampleFiles[args.t0:args.tE]

  # x, z, ll_x = zip(*du.Parfor(work, sampleFiles, showProgress=True))
  x, z, ll_x = zip(*du.Parfor(work, sampleFiles))
  # x, z, ll_x = zip(*du.For(work, sampleFiles))

  x, z = ( list(x), list(z) )

  # get y as json
  y = { f't{k:05}' : v.tolist() for k, v in jpt.io.load(sampleFiles[0])['y']._y.items() }
  with open(f'{args.outfile}_y.json', 'w') as fid: json.dump(y, fid)

  # make separate json for each sample x, z
  nSamples = len(x)
  for nS in range(nSamples):
    for k in x[nS].keys():
      x[nS][k] = { f't{k_:05}' : v.tolist() for k_, v in x[nS][k].items() }
    x[nS] = { f'k{k_:03}' : v for k_, v in x[nS].items() }
    z[nS] = { f't{k_:05}' : v.tolist() for k_, v in z[nS].items() }
    
    dct = {'x': x[nS], 'z': z[nS], 'll': ll_x[nS] }
    with open(f'{args.outfile}_sample_{nS:05}.json', 'w') as fid:
      json.dump(dct, fid)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('sampleDir', type=str, help='directory with jpt samples')
  parser.add_argument('outfile', type=str, help='output mat file')
  parser.add_argument('--t0', type=int, default=0, help='first sample index')
  parser.add_argument('--tE', type=int, default=-1, help='last sample index')
  parser.set_defaults(func=main)

  args = parser.parse_args()
  args.func(args)

