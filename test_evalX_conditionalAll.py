import du, subprocess, numpy as np
import IPython as ip

count = 4
sampleDir = 'data/datasets/k22/samples001/jpt'
filesList = du.GetFilePaths(sampleDir, 'gz')[:count]
allFilesStr = ' '.join(filesList)

# run old script
nums1 = [ ]
for n in range(count):
  cmd1 = f'scripts/jpt_evalX_conditional {filesList[n]} {allFilesStr}'
  output = subprocess.check_output(cmd1, shell=True).decode('utf-8')
  
  nums_1n = [ float(s) for s in output.split('\n') if s ]
  nums1 += nums_1n
nums1 = np.array(nums1)

# get values with new script
K = 2
cmd2 = f'scripts/jpt_evalX_conditionalAll {K} {allFilesStr}'
output = subprocess.check_output(cmd2, shell=True).decode('utf-8')
nums2 = np.array([ float(s) for s in output.split('\n') if s ])

# nums1 is in the order of
#   log p(x_1 | z_1, y)
#   log p(x_1 | z_2, y)
#   ...
#   log p(x_1 | z_1000, y)
#   log p(x_2 | z_1, y)
#   log p(x_2 | z_2, y)
#   ...
#   log p(x_2 | z_1000, y)
#   ...
#   ...
#   log p(x_1000 | z_1000, y)


# nums2 is in the order of
#   log p(x_1 | z_1, y)
#   log p(x_2 | z_1, y)
#   ...
#   log p(x_1000 | z_1, y)
#   log p(x_1 | z_2, y)
#   log p(x_2 | z_2, y)
#   ...
#   log p(x_1000 | z_2, y)
#   ...
#   ...
#   log p(x_1000 | z_1000, y)
ip.embed()
