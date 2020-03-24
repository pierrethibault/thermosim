"""
Setting up many small molecules and a big fat one.
"""

import numpy as np
import thermosim
import matplotlib as mpl

N = 101
l = 400
b = thermosim.Box.generic(N=N, L=[l, l])
b.colors = None

# Put all molecules in bottom half and make them motionless
x, y = np.indices((10, 10))

b.r[1:, 1] = l/3. + 10*(x.flatten() - 4.5)
b.r[1:, 0] = l/2. + 10*(y.flatten() - 4.5)
b.v.fill(0.)

b.r[0, 1] = l/2. + 100.
b.r[0, 0] = l/2. + .5
b.d.fill(2.)
b.d[0] = 100.
b.m[0] = N/2.
b.v[0, 1] = -4*np.sqrt(N)/N
b.v[0, 0] = 0.

b._init()
b.set_colors('k')
b.fig.set_size_inches((12, 12), forward=True)
b.run(1500, block=True)
b.v *= -1
input('Hit enter to continue')
b.run(1500)


