import numpy as np
import thermosim
import matplotlib as mpl

b = thermosim.Box.generic(N=500, L=[400,200])
colors = ['blue' if r[0] < b.L[0]/2 else 'red' for r in b.r]
b.colors = None
b._init()
b.set_colors(colors)
print colors
b.run(100000)
mpl.pyplot.show(True)
