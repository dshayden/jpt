import jpt, du, numpy as np, matplotlib.pyplot as plt
import IPython as ip, sys

# make small dataset to try appearance modeling on
# can easily add to this by adding image # and +/- x/y for the rule that
# distinguishes which is the colored marmoset (code assumes exactly two
# detections in each frame)
sample = jpt.io.load('data/datasets/marmoset/samples/sample-00399.gz')

colored = [
  (  139, '+x' ),
  (  366, '+y' ),
  (  571, '+y' ),
  (  957, '+x' ),
  ( 1033, '-x' ),
  ( 1174, '+x' ),
  ( 1289, '-y' ),
  ( 1484, '-x' )
]

subImg = [ ]
pixels = [ ]

for (frameNum, rule) in colored:
  y = sample['y'][frameNum-1]
  masks = sample['yMasks'][frameNum-1][0]
  img = sample['yImgs'][frameNum-1][0]

  ind = np.array([0, 1], dtype=np.int)
  if rule == '+x' and y[0,0] < y[1,0]: ind = ind[::-1]
  elif rule == '-x' and y[0,0] > y[1,0]: ind = ind[::-1]
  elif rule == '+y' and y[0,1] < y[1,1]: ind = ind[::-1]
  elif rule == '-y' and y[0,1] > y[1,1]: ind = ind[::-1]

  # highlight colored marmoset in images for verification
  subImg.append( du.DrawOnImage(img, np.where(masks[ind[0]]), [1.0, 0., 0., 0.5]) )

  # get ndarray of pixels of colored and non-colored marmoset
  pixels.append( (img[np.where(masks[ind[0]])], img[np.where(masks[ind[1]])]) )

color, no_color = zip(*pixels)
color = [ c.astype(np.float) / 255. for c in color ]
no_color = [ c.astype(np.float) / 255. for c in no_color ]

du.save('data/datasets/marmoset/color_nocolor', {'color': color, 'no_color': no_color})

## each picture should have colored marmoset shaded
# du.ViewManyImages(subImg)
# plt.show()
