"""
Depletion simulation
"""

import numpy as np
import thermosim
import matplotlib as mpl

Ns = 1000
Nl = 10
N = Ns + Nl
l = 400
b = thermosim.Box.generic(N=N, L=[l, l], T=.5)
b.colors = None

b.d.fill(3.)
b.d[Ns:] = 50.
b.m[Ns:] = 150.
b.v[Ns:, :] = 0.

b._init()
colors = Ns*['k'] + Nl*['b']
b.set_colors(colors)
b.fig.set_size_inches((7, 7), forward=True)
b.run(2000)
#b.run(15000, filename='test.mp4')

mpl.pyplot.show(True)


