from thermosim import Box
import numpy as np

class Infection(object):
    infected_color = 'red'
    healthy_color = 'skyblue'
    recovered_color = 'green'
    incubating_color = 'orange'
    status_colors = ['skyblue', 'orange', 'red', 'green']

    def __init__(self, box, incubation_time = 10., infection_time = 30.):
        self.box = box
        cb = box.cbox
        self.cb = cb

        self.incubation_time = incubation_time
        self.infection_time = infection_time

        self.status = np.zeros((cb.N,), dtype=int)
        self.incubation = -np.inf * np.ones((cb.N,))
        self.infection = -np.inf * np.ones((cb.N,))

        self.time = [0.]
        self.N_healthy = [cb.N-1]
        self.N_incubating = [1]
        self.N_infected = [0]
        self.N_recovered = [0]

        ci = np.argmin(((cb.r-cb.L[0]/2)**2).sum(axis=1))

        self._infect(ci)

        box.colors = None
        box._init()
        self._update_colors()

        cb._collision_callback = self.infection_callback
        box._update_callback = self.update

        self.nc = 0

    def infection_callback(self, box, i, j):
        self.nc += 1
        if self.status[i] == 2:
            self._infect(j)
        elif self.status[j] == 2:
            self._infect(i)

        self._update_colors()

        return

    def update(self, i):
        self.incubation += self.cb.dt
        done1 = self.incubation > 0
        self.incubation[done1] = -np.inf
        self.status[done1] += 1
        self.infection[done1] = -self.infection_time

        self.infection += self.cb.dt
        done2 = self.infection > 0
        self.infection[done2] = -np.inf
        self.status[done2] += 1
        if any(done1) or any(done2):
            self._update_colors()

        self.time.append(self.cb.t)
        self.N_healthy.append((self.status==0).sum())
        self.N_incubating.append((self.status==1).sum())
        self.N_infected.append((self.status==2).sum())
        self.N_recovered.append((self.status==3).sum())

    def _infect(self, i):
        if self.status[i] > 0:
            return
        self.status[i] = 1
        self.incubation[i] = -self.incubation_time

    def _update_colors(self):
        colors = [self.status_colors[s] for s in self.status]
        self.box.set_colors(colors)

class ContagionPanel(object):
    """
    A display panel in addition to the box. Here, showing contagion status as a function of time
    """

    def __init__(self, box, ic):
        self.box = box
        self.cbox = box.cbox
        self.ic = ic
        box.panel = self

    def init(self, ax):
        self.ax = ax
        ax.clear()
        self.cbox._pressure = 0
        self.plot_healthy = ax.plot([0], [0], '-', color=self.ic.healthy_color, label='Susceptible')[0]
        self.plot_infected = ax.plot([0], [0], '-', color=self.ic.infected_color, label='Infected')[0]
        self.plot_recovered = ax.plot([0], [0], '-', color=self.ic.recovered_color, label='Recovered')[0]
        to_show = (self.plot_healthy, self.plot_infected, self.plot_recovered)
        if self.ic.incubation_time > 0:
            self.plot_incubating = ax.plot([0], [0], '-', color=self.ic.incubating_color, label='Incubating')[0]
            to_show = to_show + (self.plot_incubating,)
        else:
            self.plot_incubating = None
        ax.legend(loc='upper left')
        ax.set_xlabel('time')
        ax.set_ylabel('population')
        ax.set_ylim(0, self.cbox.N)

        p0 = ax.get_position()
        p1 = ax.get_position()
        ax.set_position([p1.x0, p0.y0, p0.width, p0.height])
        return to_show

    def update(self, i):
        ic = self.ic

        # Update plot data
        self.plot_healthy.set_data(ic.time, ic.N_healthy)
        self.plot_infected.set_data(ic.time, ic.N_infected)
        self.plot_recovered.set_data(ic.time, ic.N_recovered)
        to_show = (self.plot_healthy, self.plot_infected, self.plot_recovered)
        if self.plot_incubating:
            self.plot_incubating.set_data(ic.time, ic.N_incubating)
            to_show = to_show + (self.plot_incubating,)
        self.ax.set_xlim(0, ic.time[-1])
        return to_show


L = 400
b = Box.generic(N=800, L=[L, L])

ic = Infection(b, 0., 1000.)
cp = ContagionPanel(b, ic)
b._init()
collision_rate = np.sqrt((b.cbox.v**2).sum(axis=1)).mean() / b.cbox.mfp

b.set_fig_position(0, 0, 1286, 600)
#b.set_fig_position(0, 0, 1286, 1150)
b.run(100000, blit=True)
