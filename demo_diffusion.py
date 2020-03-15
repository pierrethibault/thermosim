import thermosim
import numpy as np

# Start with 600 blue molecules
L = 400
b = thermosim.Box.generic(N=1000, L=[L, L])

R = 50
colors = ['red' if ((r-L/2)**2).sum()<R**2 else 'skyblue' for r in b.r]
b.colors = None
b._init()
b.set_colors(colors)
b.set_fig_position(0, 0, 1286, 1150)
b.v *=.5
b.run(100000, blit=True)
