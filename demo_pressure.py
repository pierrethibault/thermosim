import thermosim

b = thermosim.Box.generic(N=1000)
b.side_panel = 'pressure'
b._init()
b.set_fig_position(0, 0, 1286, 1150)
b.run(1000)
