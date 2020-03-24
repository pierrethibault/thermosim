"""
Setting up many small molecules and a big fat one.
"""

import numpy as np
import thermosim
import matplotlib as mpl
from matplotlib import pyplot as plt


N = 1200
l = 300
b = thermosim.Box.generic(N=N, L=[l, l], D=1.1)

img = 1. - plt.imread('text.png')[::-1, :]
ymin = np.nonzero(img.sum(axis=1))[0][0]-1
ymax = np.nonzero(img.sum(axis=1))[0][-1]
xmin = np.nonzero(img.sum(axis=0))[0][0]-1
xmax = np.nonzero(img.sum(axis=0))[0][-1]
xy = np.random.uniform(low=[xmin, ymin], high=[xmax, ymax], size=(20*N, 2))
xyint = np.round(xy).astype(int)
good = img[xyint[:, 1], xyint[:, 0]] > .5
imax = np.nonzero(np.cumsum(good)>N)[0][0]
good[imax:] = False
b.r = xy[good]

b._init()
#b.set_colors('k')
b.fig.set_size_inches((12, 12), forward=True)
#b.run(1, block=True)
b.dt = .02
b.run(600, block=True)
b.v *= -1
b.dt = .04
print("b.run(300)")


