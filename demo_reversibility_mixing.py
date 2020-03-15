"""
Trying to unmix - but it doesn't work.
"""

import numpy as np
import thermosim
import matplotlib as mpl

b = thermosim.Box.generic(N=100, L=[400, 200])
b.set_fig_position(0, 0, 1286, 1150)
colors = ['blue' if r[0] < b.L[0]/2 else 'red' for r in b.r]
b.colors = None
b._init()
b.set_colors(colors)

b.fig.set_size_inches((12, 12), forward=True)
b.run(300, block=True)
b.v *= -1
raw_input('Hit enter to continue')
b.run(800, block=True)


