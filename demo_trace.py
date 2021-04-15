import thermosim

b = thermosim.Box.generic(N=200)
b.show_trace = 0
thermosim.MFPPanel(b)
print('Mean free path: %f' % b.mfp)
b._init()
b.set_fig_position(0, 0, 1286, 1150)
b.run(100000)
