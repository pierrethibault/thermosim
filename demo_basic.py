import thermosim

b = thermosim.Box.generic(D=2., N=1000)
b.set_fig_position(0, 0, 1286, 1150)
b.run(100000)
