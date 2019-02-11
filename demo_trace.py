import thermosim

b = thermosim.Box.generic(N=200)
b.toshow['trace'] = 0
print('Mean free path: %f' % b.mfp)
b._init()
b.set_fig_position(0, 0, 1286, 1150)
b.run(10000)
