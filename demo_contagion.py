from thermosim import Box
import numpy as np

"""
class CBox(Box):
    def __init__(self,  L, r, v, d, m, **kwargs):
        super(Box, self).__init__(self, L, r, v, d, m, **kwargs)
        self.infected_color = None
        self.healthy_color = None

    def _init(self):
        super(Box, self)._init(self)
        self.N_infected = sum(c == 'infected_color' for c in self.colors)
"""
# Start with 1000 blue molecules
L = 400
b = Box.generic(N=1000, L=[L, L])
infected_color = 'red'
healthy_color = 'skyblue'
ci = np.argmin(((b.r-L/2)**2).sum(axis=1))
colors = [healthy_color for r in b.r]
colors[ci] = infected_color
b.colors = None
b._init()
b.set_colors(colors)

def infection_callback(box, i, j):
    if (box._colors[i] == healthy_color) ^ (box._colors[j] == healthy_color):
        box._colors[i] = infected_color
        box._colors[j] = infected_color
        box.set_colors()
    return

b._collision_callback = infection_callback
b.set_fig_position(0, 0, 1286, 1150)
b.run(100000, blit=True)
