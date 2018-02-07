import thermosim

b = thermosim.Box.generic(N=500, L=200)
b.toshow['quiver'] = True
b._init()
b.run(10000)
