import thermosim

b = thermosim.Box.generic(N=500)
thermosim.SpeedPanel(b)
b.dt = 1.
b.v[:, 1] = 0.
b.v[:, 0] = 1.
b._init()
b.set_fig_position(0, 0, 1286, 1150)
b.run(100000)
