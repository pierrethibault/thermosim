import numpy as np
import sys
from matplotlib.collections import EllipseCollection
from matplotlib.patches import Polygon
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.spatial.distance import squareform, pdist


def norm(x):
    return np.sqrt(np.dot(x, x))


def sqrt(x):
    """Safe square root"""
    return np.sqrt(np.clip(x, 0, np.inf))


class VelocityPanel(object):
    """
    A display panel in addition to the box. Here, showing velocity histogram
    """
    def __init__(self, box):
        self.box = box
        self.cbox = box.cbox
        box.panel = self

    def init(self, ax):
        self.ax = ax
        self.ax.clear()
        b = self.box
        cb = self.cbox
        ax.set_xlabel('velocity along x')
        ax.get_yaxis().set_visible(False)
        f, bins = np.histogram(cb.v[:, 0], int(np.sqrt(cb.N)), range=(cb.vxmin, cb.vxmax), density=True)
        binwidth = bins[1] - bins[0]

        # Create bar graph
        self.vhist = ax.bar(bins[:-1], f, width=binwidth, align='edge')

        p0 = ax.get_position()
        p1 = ax.get_position()
        ax.set_position([p1.x0, p0.y0, p0.width, p0.height])
        return self.vhist,

    def update(self, i):
        b = self.box
        cb = self.cbox

        # Recompute histogram
        nbins = len(self.vhist)
        f, bins = np.histogram(cb.v[:, 0], nbins, range=(cb.vxmin, cb.vxmax), density=True)
        self.ax.set_xlim(cb.vxmin, cb.vxmax)
        binwidth = bins[1] - bins[0]

        # Update histogram
        for i in range(nbins):
            self.vhist[i].set_height(f[i])
            self.vhist[i].set_width(binwidth)
            self.vhist[i].set_facecolor(b.cm((bins[i] + .5 * binwidth) ** 2 / cb.v2max))
            self.vhist[i].set_x(bins[i])
        # Adjust vertical extent.
        ylim = self.ax.get_ylim()[1]
        dylim = 1.2 * f.max() - ylim
        if abs(dylim) > .1 * ylim:
            new_ylim = ylim + .1 * (dylim)
            self.ax.set_ylim(ymax=new_ylim)

        return self.vhist,

class SpeedPanel(object):
    """
    A display panel in addition to the box. Here, showing velocity histogram
    """
    def __init__(self, box):
        self.box = box
        self.cbox = box.cbox
        box.panel = self

    def init(self, ax):
        self.ax = ax
        self.ax.clear()
        b = self.box
        cb = self.cbox

        ax.set_xlabel('velocity along x')
        ax.get_yaxis().set_visible(False)
        f, bins = np.histogram(np.sqrt((cb.v ** 2).sum(axis=1)), int(np.sqrt(cb.N)),
                                range=(0., cb.v2max), density=True)
        binwidth = bins[1] - bins[0]

        # Create bar graph
        self.vhist = ax.bar(bins[:-1], f, width=binwidth, align='edge')

        p0 = ax.get_position()
        p1 = ax.get_position()
        ax.set_position([p1.x0, p0.y0, p0.width, p0.height])
        return self.vhist,

    def update(self, i):
        b = self.box
        cb = self.cbox

        # Recompute histogram
        nbins = len(self.vhist)
        f, bins = np.histogram(np.sqrt((cb.v**2).sum(axis=1)), nbins, range=(0, cb.v2max), density=True)
        self.ax.set_xlim(0, cb.v2max)
        binwidth = bins[1] - bins[0]

        # Update histogram
        for i in range(nbins):
            self.vhist[i].set_height(f[i])
            self.vhist[i].set_width(binwidth)
            self.vhist[i].set_facecolor(b.cm((bins[i] + .5 * binwidth) ** 2 / cb.v2max))
            self.vhist[i].set_x(bins[i])
        # Adjust vertical extent.
        ylim = self.ax.get_ylim()[1]
        dylim = 1.2 * f.max() - ylim
        if abs(dylim) > .1 * ylim:
            new_ylim = ylim + .1 * (dylim)
            self.ax.set_ylim(ymax=new_ylim)

        return self.vhist,

class MFPPanel(object):
    """
    A display panel in addition to the box. Here, showing mean free path plot
    """

    def __init__(self, box):
        self.box = box
        self.cbox = box.cbox
        self.trace_length = 1
        self.nbins = 30
        box.panel = self

    def init(self, ax):
        self.ax = ax
        ax.clear()
        b = self.box
        cb = self.cbox

        ax.set_xlabel('free path')
        ax.get_yaxis().set_visible(False)
        ax.set_ylim(ymin=0.)
        f, bins = np.histogram([], self.nbins, range=(0, 3*cb.mfp), density=True)
        binwidth = bins[1] - bins[0]

        # Create bar graph
        self.vhist = ax.bar(bins[:-1], f, width=binwidth, align='edge')

        p0 = ax.get_position()
        p1 = ax.get_position()
        ax.set_position([p1.x0, p0.y0, p0.width, p0.height])
        return self.vhist,

    def update(self, i):
        b = self.box
        cb = self.cbox

        # Recompute histogram
        x, y = cb.trace.get_data()
        if len(x) <= self.trace_length:
            # Nothing to do
            return self.vhist,
        self.trace_length = len(x)
        paths = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        xlim = max(3*cb.mfp, paths.max())
        f, bins = np.histogram(paths, self.nbins, range=(0, xlim), density=True)
        self.ax.set_xlim(0, xlim)
        binwidth = bins[1] - bins[0]

        # Update histogram
        for i in range(self.nbins):
            self.vhist[i].set_height(f[i])
            self.vhist[i].set_width(binwidth)
            self.vhist[i].set_x(bins[i])

        # Adjust vertical extent
        self.ax.set_ylim(ymax=1.2*f.max())

        return self.vhist,

class PressurePanel(object):
    """
    A display panel in addition to the box. Here, showing cumulative momentum on one of the walls
    """

    def __init__(self, box):
        self.box = box
        self.cbox = box.cbox
        self.trace_length = 1
        self.nbins = 30
        box.panel = self

    def init(self, ax):
        self.ax = ax
        ax.clear()
        self.cbox._pressure = 0
        self.plot_pressure = ax.plot([0], [0], 'b-', label='Total momentum')[0]
        self.plot_pressure_theory = ax.plot([0, 0], [0, 0], 'k-', label='Total momentum (theory)')[0]
        ax.legend(loc='upper left')
        ax.set_xlabel('time')
        ax.set_ylabel('momentum')
        ax.set_ylim(ymin=0.)

        p0 = ax.get_position()
        p1 = ax.get_position()
        ax.set_position([p1.x0, p0.y0, p0.width, p0.height])
        return self.plot_pressure, self.plot_pressure_theory

    def update(self, i):
        b = self.box
        cb = self.cbox

        # Update plot data
        self.plot_pressure.set_data(cb._pressure_t, cb._wall_momentum)
        self.plot_pressure_theory.set_data([0, cb._pressure_t[-1]], [0, cb._wall_momentum_theory])
        self.ax.set_xlim(0, cb._pressure_t[-1])
        self.ax.set_ylim(np.min(cb._wall_momentum), np.max(cb._wall_momentum))
        return self.plot_pressure, self.plot_pressure_theory

class TimePanel(object):
    """
    A display panel in addition to the box. Here, showing mean free path plot
    """

    def __init__(self, box):
        self.box = box
        box.panel = self

    def init(self, ax):
        self.ax = ax
        ax.clear()
        b = self.box

    def update(self, i):
        b = self.box


class Box(object):

    def __init__(self, cbox, **kwargs):
        """
        Wrapper for a CBox implementing visualisation.
        """
        self.cbox = cbox

        self.fig = plt.figure()

        self.panel = None

        self.show_trace = None
        self.show_quiver = None

        # For molecule trace
        self._vtrace = None

        # Default coloring
        self.colors = 'velocities'

        self._colors = None

        self.cids = []

        # Create display
        self._init()

        self._i = None
        self.animobj = None

        self.highlight_rule = None

        self._update_callback = None

    @classmethod
    def generic(cls, N=150, L=200., D=3., T=1., ndim=2):
        """
        Create a generic box with a given number of particles, given diameter, box dimensions and reduced temperature.
        self.N = N  # number of particles
        """
        if np.isscalar(L):
            L = L*np.ones((ndim,))

        # Positions
        r = np.array([L[i]*np.random.uniform(size=(N,)) for i in range(ndim)]).T

        # Masses
        m = np.ones((N,))

        # Diameters
        d = D*np.ones((N,))

        # Velocities
        v = np.random.normal(size=(N, ndim))
        v /= np.sqrt((v**2).sum()/N)
        v *= np.sqrt(2*T/m[:, np.newaxis])

        cbox = CBox(L, r, v, d, m)

        return cls(cbox)

    def run(self, nsteps=100000, filename=None, blit=False, block=None):
        """Start animation

        nsteps: number of steps
        filename: if not None, movie file to save to (work in progress)
        blit: False by default, does not work always
        block: if True, return only after run is done, if False return immediately
               if None, returns only in interactive mode
        """
        self._i = 0
        self.nsteps = nsteps
        self.animobj = animation.FuncAnimation(self.fig, self._update, frames=nsteps, interval=5., repeat=False, blit=blit)
        if filename is not None:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=150, bitrate=500, extra_args=['-filter', 'tblend', '-r', '25'])
            self.animobj._start()
            self.animobj.save(filename, writer=writer)
        else:
            self.animobj._start()
            plt.show(block=False)
            if block or ((block is None) and not hasattr(sys, 'ps1')):
                plt.pause(.1)
                while self.animobj.event_source and self.animobj.event_source.callbacks:
                    plt.pause(.1)

    def stop(self):
        try:
            self.animobj._stop()
        except:
            pass

    def _init(self):
        """
        Initialise display
        """
        cb = self.cbox

        # No update during setup
        plt.ioff()
        self.fig.clear()

        if self.panel is not None:
            # Create two side-by-side axes, with a histogram of the x-component of the velodicities in the second one.
            axes = [self.fig.add_subplot(121, aspect='equal', adjustable='box'),
                    self.fig.add_subplot(122)]
        else:
            # Create just one axis - the particle box
            axes = [self.fig.add_subplot(111, aspect='equal', adjustable='box')]

        # Set box size and axis properties
        axes[0].axis([0, cb.L[0], 0, cb.L[1]])
        axes[0].get_xaxis().set_visible(False)
        axes[0].get_yaxis().set_visible(False)

        # Draw particles
        circles = EllipseCollection(widths=cb.d, heights=cb.d, angles=0, units='xy',
                                    facecolors='k', offsets=cb.r, transOffset=axes[0].transData)
        axes[0].add_collection(circles)

        # Create colormap
        self.cm = plt.get_cmap('plasma')

        self.fig.tight_layout()
        self.circles = circles

        to_return = (circles,)

        # Option to show the trace of one particle (to illustrate random walk)
        if self.show_trace is not None:
            i = self.show_trace
            self.trace = axes[0].plot([cb.r[i, 0]], [cb.r[i,1]], 'k-')[0]
            self._vtrace = cb.v[i].copy()
            to_return += (self.trace,)

        # Option to show velocity arrows
        if self.show_quiver:
            quiver = plt.quiver(cb.r[:, 0], cb.r[:, 1], cb.v[:, 0], cb.v[:, 1], units='xy', scale=35.*cb.vRMS/cb.L.mean())
            self.quiver = quiver
            to_return += (quiver,)

        self.axes = axes

        if self.panel is not None:
            to_return += self.panel.init(axes[1])

        # Process all obstacles (polygons)
        if cb.obstacles:
            for obs in cb.obstacles:
                vc = obs['vertices']
                axes[0].add_patch(Polygon(vc, facecolor='black'))

        # (re)connect events
        self._connect()

        return to_return

    def set_fig_position(self, x, y, dx, dy):
        """Set figure windoe position (might work only with QT backend)"""
        plt.get_current_fig_manager().window.setGeometry(x, y, dx, dy)

    def set_colors(self, colors=None):
        if colors is None:
            colors = self._colors
        else:
            self._colors = colors
        self.circles.set_facecolors(colors)

    def _connect(self):
        """
        Manage event connections
        FIXME: This does not work.
        """
        canvas = self.fig.canvas
        # Disconnect eventual connections
        for cid in self.cids:
            canvas.mpl_disconnect(cid)
        # Reconnect
        self.cids.append(canvas.mpl_connect('button_press_event', self.onpress))
        self.cids.append(canvas.mpl_connect('key_press_event', self.onkeypress))
        self.cids.append(canvas.mpl_connect('close_event', self.onclose))
        self.cids.append(canvas.mpl_connect('scroll_event', self.onscroll))
        return

    def onpress(self, event):
        pass

    def onkeypress(self, event):
        if event.key in ['space']:
            print('blip!')
            #self.animobj._stop()

    def onclose(self, event):
        pass

    def onscroll(self, event):
        pass

    def _update(self, i):
        """Update plot"""
        cb = self.cbox

        self.i = i

        # Compute move
        cb.step()

        # Move molecules
        self.circles.set_offsets(cb.r)

        # Change colours
        if self.colors == 'velocities':
            vmag = np.sqrt((cb.v**2).sum(axis=1))
            self.set_colors(self.cm(vmag/cb.v2max))

        to_return = (self.circles,)

        if self.panel is not None:
            to_return += self.panel.update(i)

        if self.show_trace is not None:
            i = self.show_trace
            newv = cb.v[i]
            x, y = self.trace.get_data()
            if not np.allclose(self._vtrace, newv):
                x = np.append(x, cb.r[i,0])
                y = np.append(y, cb.r[i,1])
            else:
                x[-1] = cb.r[i,0]
                y[-1] = cb.r[i,1]
            self.trace.set_data(x,y)
            self._vtrace = newv.copy()
            to_return += (self.trace,)

        if self.show_quiver:
            self.quiver.set_offsets(cb.r)
            self.quiver.set_UVC(cb.v[:, 0], cb.v[:, 1])
            to_return += (self.quiver,)

        if self.highlight_rule:
            highlighted = eval(self.highlight_rule, cb.__dict__)
            lw = 5.
            self.circles.set_lw([lw if h else 0. for h in highlighted])
            self.circles.set_edgecolors(['yellow' if h else 'black' for h in highlighted])
            if self.show_quiver:
                self.quiver.set_UVC(highlighted*cb.v[:, 0], highlighted*cb.v[:, 1])

        if self._update_callback is not None:
            self._update_callback(i)

        return to_return

    def show(self):
        plt.ion()


class CBox(object):

    def __init__(self, L, r, v, d, m, **kwargs):
        """
        Create a box with particles inside.
        """
        N, ndim = r.shape
        if np.isscalar(L):
            L = L*np.ones((ndim,))

        self.N = N
        self.ndim = ndim
        self.L = L
        self.r = r
        self.v = v
        self.d = d
        self.m = m

        self.bounds = [[0, L[i]] for i in range(ndim)]

        # Initialise time
        self.t = 0.

        # Number of steps (defined in run)
        self.nsteps = None

        # Total time and time step (computed in init)
        self.dt = None
        self.t = 0.
        self.i = 0

        # Mean free path
        self.mfp = None

        # Initialise other necessary attributes
        self.r0 = None  # This will store the previous positions
        self.v0 = None  # For previous velocities

        # Gravity
        self.g = 0.

        # Velocity statistics
        self.v2max = None
        self.vxmin, self.vxmax = None, None
        self.vymin, self.vymax = None, None

        # For pressure calculation
        self._pressure = None
        self._wall_momentum = [0.]
        self._wall_momentum_theory = 0.
        self._pressure_t = [0.]

        # Real volume
        self.real_volume = None

        # For obstacle collisions
        self.obstacles = []

        # Callbacks
        self._collision_callback = None
        self._walls_callback = None
        self._obs_callback = None

        self.init()

    @classmethod
    def generic(cls, N=150, L=200., D=3., T=1., ndim=2):
        """
        Create a generic box with a given number of particles, given diameter, box dimensions and reduced temperature.
        self.N = N  # number of particles
        """
        if np.isscalar(L):
            L = L*np.ones((ndim,))

        # Positions
        r = np.array([L[i]*np.random.uniform(size=(N,)) for i in range(ndim)]).T

        # Masses
        m = np.ones((N,))

        # Diameters
        d = D*np.ones((N,))

        # Velocities
        v = np.random.normal(size=(N, ndim))
        v /= np.sqrt((v**2).sum()/N)
        v *= np.sqrt(2*T/m[:, np.newaxis])

        return cls(L, r, v, d, m)

    @property
    def T(self):
        """Temperature"""
        return (.5*self.m * (self.v**2).sum(axis=1)).mean()

    @property
    def vRMS(self):
        """RMS velocity"""
        return np.sqrt((self.v**2).sum(axis=1).mean())

    @property
    def P(self):
        """Pressure - in 2D, P = U/A"""
        return (.5*self.m * (self.v**2).sum(axis=1)).sum() / self.real_volume

    def init(self):
        """
        Reinitialise all quantities.
        """
        # Velocity statistics
        self.v2max = np.sqrt((self.v**2).sum(axis=1)).max()
        self.vxmin, self.vxmax = self.v[:, 0].min(), self.v[:, 0].max()
        self.vymin, self.vymax = self.v[:, 1].min(), self.v[:, 1].max()

        # Optimal time step ~ .25 * (D/v_RMS)
        self.dt = .25*self.d.mean()/np.sqrt(2*(self.v**2).mean())

        # Mean free path
        self.mfp = np.prod(self.L)/(4.*self.N*self.d.mean())

        # Volume
        self.real_volume = np.prod(self.L-self.d.mean()) - .5*np.pi*sum(self.d**2)

        # For pressure calculation
        self._pressure = None
        self._wall_momentum = [0.]
        self._wall_momentum_theory = 0.
        self._pressure_t = [0.]

    def add_obstacle(self, vc):
        """
        Add an obstacle (convex polygon defined by vertices vc).
        """
        # Construct edges info
        edges = []
        for i in range(len(vc)):
            a = vc[i] - vc[i-1]
            n = np.array([-a[1], a[0]])
            n /= norm(n)
            edges.append((n, np.dot(n, vc[i])))

        # Store
        self.obstacles.append({'vertices': vc, 'edges': edges})

    def update_stats(self):
        """
        Update statistics
        """
        vmag = np.sqrt((self.v**2).sum(axis=1))
        self.v2max = max(self.v2max, vmag.max())
        self.vxmax = max(self.vxmax, self.v[:, 0].max())
        self.vxmin = min(self.vxmin, self.v[:, 0].min())
        self.vymax = max(self.vymax, self.v[:, 1].max())
        self.vymin = min(self.vymin, self.v[:, 1].min())

    def step(self):
        """
        Move by one step
        """
        # Increment
        self.i += 1
        self.r += self.dt * self.v
        self.t += self.dt

        # Process collisions
        self.walls(self._walls_callback)
        self.collide(callback=self._collision_callback)
        self.obs_collide(self._obs_callback)

        # Update statistics
        self.update_stats()

    def walls(self, callback=None):
        """
        Process wall collisions.

        TODO: implement wall callback
        """
        for dim in range(self.ndim):
            # "Negative" wall
            d0 = self.r[:, dim] - .5*self.d - self.bounds[dim][0]
            self.r[d0 < 0, dim] -= 2*d0[d0 < 0]
            self.v[d0 < 0, dim] *= -1
            # "Positive" wall
            d1 = self.r[:, dim] + .5*self.d - self.bounds[dim][1]
            self.r[d1 > 0, dim] -= 2*d1[d1 > 0]
            self.v[d1 > 0, dim] *= -1

            if self._pressure == dim:
                #self._pressure_t.append(self._pressure_t[-1] + self.dt)
                #self._wall_momentum.append(self._wall_momentum[-1] - 2 * sum(self.m[d1 > 0] * self.v[d1 > 0, dim]))
                #self._wall_momentum_theory.append(self._wall_momentum_theory[-1] + self.P*self.L[dim]*self.dt)
                self._pressure_t.append(self.t)
                self._wall_momentum.append(self._wall_momentum[-1] - 2 * sum(self.m[d1 > 0] * self.v[d1 > 0, dim]))
                self._wall_momentum_theory = self.P*self.L[dim]*self.t

    def obs_collide(self, callback=None):
        """
        Process collisions with additional rectangular obstacles

        TODO: implement obstacle callback
        """
        for obs in self.obstacles:
            vc = obs['vertices']
            ec = obs['edges']
            Nn = len(ec)

            nn = np.array([n for n,e in ec])

            # Find molecules that collided
            dw = np.array([np.dot(self.r, n) - .5*self.d - e for n, e in ec]).T
            hit = (dw < 0).all(axis=1)

            if not hit.any():
                # No particle collided
                continue

            Nh = hit.sum()

            # Work on subset
            r = self.r[hit]
            v = self.v[hit]
            d = self.d[hit]
            dw = dw[hit]

            # Find which facet was hit
            dta = dw / np.dot(v, nn.T)
            dta[dta < 0] = 1e12

            # Which wall was hit?
            wi = np.argmin(dta, axis=1)

            # How long ago?
            dt = np.min(dta, axis=1)

            # Backtrack
            rc = r - dt[:, None]*v

            # Process individually
            for i in range(Nh):
                rcc = rc[i]
                rr = r[i]
                vv = v[i]
                dd = d[i]

                # Check if we hit a corner
                vc0, vc1 = vc[wi[i]-1], vc[wi[i]]
                vcc = None
                if np.dot(rcc-vc0, vc1 - vc0) < 0:
                    vcc = vc0
                elif np.dot(rcc-vc1, vc0 - vc1) < 0:
                    vcc = vc1

                if vcc is not None:
                    # Corner collision
                    dr = rr - vcc

                    ndv = norm(vv)
                    ru = np.dot(vv, dr) / ndv
                    b2 = ru**2 + .25*dd**2 - np.dot(dr, dr)
                    if b2 < 0:
                        # No collision - this should not happen
                        continue

                    ds = ru + sqrt(b2)
                    dtc = ds / ndv
                    drc = dr - vv * dtc

                    # Store new values
                    v[i] = vv - 2. * drc * np.dot(vv, drc) / np.dot(drc, drc)
                    r[i] = rr + (v[i] - vv) * dtc

                else:
                    # Edge collision
                    v[i] -= 2 * np.dot(v[i], nn[wi[i]]) * nn[wi[i]]
                    r[i] = rcc + dt[i]*v[i]

            # Put everything back in
            self.v[hit] = v
            self.r[hit] = r

    def collide(self, callback=None):
        """
        Process eventual collisions

        callback, if not None, is a function with signature callback(self, particle_index1, particle_index2)
        and is called after updating positions and velocities.
        """

        # Find colliding particles
        D = squareform(pdist(self.r))
        ind1, ind2 = np.where(D < .5*np.add.outer(self.d, self.d))
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        # Process collisions
        for p1, p2 in zip(ind1, ind2):

            # Initial parameters
            v1, v2 = self.v[p1], self.v[p2]
            r1, r2 = self.r[p1], self.r[p2]
            d1, d2 = self.d[p1], self.d[p2]
            m1, m2 = self.m[p1], self.m[p2]

            # Relative positions and velocities
            dv = v2-v1
            dr = r2-r1

            # Backtrack
            ndv = norm(dv)
            if ndv == 0:
                # Special case: overlapping particles with same velocities
                ndr = norm(dr)
                offset = .5*dr*(.5*(d1+d2)/ndr - 1.)
                self.r[p1] -= offset
                self.r[p2] += offset
                continue
            ru = np.dot(dv, dr)/ndv
            ds = ru + sqrt(ru**2 + .25*(d1+d2)**2 - np.dot(dr, dr))
            if np.isnan(ds):
                1/0

            # Time since collision
            dtc = ds/ndv

            # New collision parameter
            drc = dr - dv*dtc

            # Center of mass velocity
            vcm = (m1*v1 + m2*v2)/(m1+m2)

            # Velocities after collision
            dvf = dv - 2.*drc * np.dot(dv, drc)/np.dot(drc, drc)
            v1f = vcm - dvf * m2/(m1+m2)
            v2f = vcm + dvf * m1/(m1+m2)

            # Backtracked positions
            r1f = r1 + (v1f-v1)*dtc
            r2f = r2 + (v2f-v2)*dtc

            # Update values
            self.r[p1] = r1f
            self.r[p2] = r2f
            self.v[p1] = v1f
            self.v[p2] = v2f

            if callback is not None:
                callback(self, p1, p2)
