import thermosim

b = thermosim.Box.generic(D=2., N=2000)
thermosim.SpeedPanel(b)
b.colors = None
b.m[:400] = 40.
b._init()
b.set_colors(400*['blue'] + 1600*['red'])
b.set_fig_position(0, 0, 1286, 1150)
b.run(100000)
