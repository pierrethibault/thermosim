import thermosim

b = thermosim.Box.generic(N=20, D=5)
thermosim.PressurePanel(b)
b._init()
b.set_fig_position(0, 0, 1286, 1150)
b.run(100000)
