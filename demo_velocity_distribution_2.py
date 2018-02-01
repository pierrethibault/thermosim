import numpy as np
import thermosim
import matplotlib as mpl

b = thermosim.Box.generic(N=500)
b.show('velocities')
b.dt = 1.
b.v[:,1] = 0.
b.v[:,0] = 1.
b._init()
b.run(100000)
mpl.pyplot.show(True)
