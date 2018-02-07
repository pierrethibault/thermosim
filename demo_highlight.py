import thermosim

b = thermosim.Box.generic(N=500, L=200, D=2)
b.toshow['quiver'] = True
b._init()
b.highlight_rule = '(v[:,1] > -.2) & (v[:,1] < .2) & (v[:,0] > .5) & (v[:,0] < .9)'
b.run(10000)
