import numpy as np
import thermosim
import matplotlib as mpl
from matplotlib import pyplot as plt

b = thermosim.Box.generic(N=1000)
b.run(100000, blit=True)
mpl.pyplot.show(True)
