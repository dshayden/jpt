import os, du

basePath = '/Users/dshayden/Downloads/behavior_dets/dets'
ifiles = du.GetFilePaths(f'{basePath}/jpt', 'gz')
ofiles = [ f'{basePath}/tracks/{du.fileparts(ifiles[i])[1]}'
  for i in range(len(ifiles)) ]
nSamples = 100

def work(ifile, ofile, nSamples):
  try: os.makedirs(ofile)
  except: pass
  exe = '/Users/dshayden/Research/code/jpt/scripts/jpt_runPointTracker'
  cmd = f'{exe} {ifile} {ofile} {nSamples}'
  os.system(cmd)

args = [ (ifiles[i], ofiles[i], nSamples) for i in range(len(ifiles)) ]
du.Parfor(work, args, nWorkers=4)
