import thermosim
import numpy as np

L = 200.
w = 30.
b = thermosim.Box.generic(N=300, L=[400, 200])
b.r = 200*np.random.uniform(size=(300,2))
b.add_obstacle(np.array([[L-1,(L+w)/2], [L-1, L],[L+1,L], [L+1,(L+w)/2]]))
b.add_obstacle(np.array([[L+1,(L-w)/2], [L+1, 0],[L-1,0], [L-1,(L-w)/2]]))
b._init()
b.run(10000)
