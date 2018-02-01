import numpy as np
import thermosim
import matplotlib as mpl
from matplotlib import pyplot as plt

b = thermosim.Box.generic(N=1000)
b.show('velocities')

if __name__ == "__main__":
    b.run(100000)
    m = plt.get_current_fig_manager()
    m.window.setGeometry(0, 0, 1286, 1150)
    mpl.pyplot.show(True)
