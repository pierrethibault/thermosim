import thermosim
import numpy as np

b = thermosim.Box.generic(N=500)
b.panel = thermosim.VelocityPanel(b)
cb = b.cbox
cb.dt = 1.
cb.v[:, 1] = 0.
cb.v[:, 0] = 1.
b._init()
b.set_fig_position(0, 0, 1286, 1150)
b.run(100000)
