import thermosim

b = thermosim.Box.generic(N=1000)
b.show('velocities')
b.set_fig_position(0, 0, 1286, 1150)
b.run(100000)
