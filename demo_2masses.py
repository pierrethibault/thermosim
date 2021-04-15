import numpy as np
import thermosim

def anim_collision(m1, m2, r1, r2, v1, v2, d1=1, d2=1):
    L = 40.
    r = np.asarray([r1, r2], dtype=float)
    r += L/2
    b = thermosim.Box.generic(N=2, D=2., L=L)
    cb = b.cbox
    cb.r[:] = r
    cb.v[:] = np.array([v1,v2])
    cb.m[:] = [m1, m2]
    cb.dt = .08
    b.quiver_scale = 15.
    b.show_quiver = True
    b._init()
    # for some reason better to increase interval in jupyter
    # b.interval = 50.
    b.run(200)
    return b

m1 = 1.
m2 = 1.
r1 = [-10, 0]
r2 = [10, 0]
v1i = np.array([1, 0.])
v2i = np.array([-5, 0.])
vcm = (v1i*m1 + v2i*m2)/(m1+m2)
anim_collision(m1, m2, r1, r2, v1i, v2i)