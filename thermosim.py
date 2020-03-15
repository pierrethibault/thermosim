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

    def init(self, ax):
        self.ax = ax
        b = self.box
        ax.set_xlabel('velocity along x')
        ax.get_yaxis().set_visible(False)
        f, bins = np.histogram(b.v[:, 0], int(np.sqrt(b.N)), range=(b.vxmin, b.vxmax), density=True)
        binwidth = bins[1] - bins[0]

        # Create bar graph
        self.vhist = ax.bar(bins[:-1], f, width=binwidth, align='edge')

        p0 = ax.get_position()
        p1 = ax.get_position()
        ax.set_position([p1.x0, p0.y0, p0.width, p0.height])
        return self.vhist,

    def update(self, i):
        b = self.box

        # Recompute histogram
        nbins = len(self.vhist)
        f, bins = np.histogram(b.v[:, 0], nbins, range=(b.vxmin, b.vxmax), density=True)
        self.ax.set_xlim(b.vxmin, b.vxmax)
        binwidth = bins[1] - bins[0]

        # Update histogram
        for i in range(nbins):
            self.vhist[i].set_height(f[i])
            self.vhist[i].set_width(binwidth)
            self.vhist[i].set_facecolor(b.cm((bins[i] + .5 * binwidth) ** 2 / b.v2max))
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
        self.trace_length = 1
        self.nbins = 30

    def init(self, ax):
        self.ax = ax
        b = self.box
        ax.set_xlabel('free path')
        ax.get_yaxis().set_visible(False)
        ax.set_ylim(ymin=0.)
        f, bins = np.histogram([], self.nbins, range=(0, 3*b.mfp), density=True)
        binwidth = bins[1] - bins[0]

        # Create bar graph
        self.vhist = ax.bar(bins[:-1], f, width=binwidth, align='edge')

        p0 = ax.get_position()
        p1 = ax.get_position()
        ax.set_position([p1.x0, p0.y0, p0.width, p0.height])
        return self.vhist,

    def update(self, i):
        b = self.box

        # Recompute histogram
        x, y = b.trace.get_data()
        if len(x) <= self.trace_length:
            # Nothing to do
            return self.vhist,
        self.trace_length = len(x)
        paths = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        xlim = max(3*b.mfp, paths.max())
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


class Box(object):

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

        # Optimal time step ~ .25 * (D/v_RMS)
        self.dt = .25*d.mean()/np.sqrt(2*(self.v**2).mean())

        # Mean free path
        self.mfp = np.prod(L)/(4.*N*d.mean())

        # Initialise other necessary attributes
        self.r0 = None  # This will store the previous positions
        self.v0 = None  # For previous velocities

        # Gravity
        self.g = -.1
        self.t = 0.

        self.v2max = np.sqrt((self.v**2).sum(axis=1)).max()
        self.vxmin, self.vxmax = self.v[:, 0].min(), self.v[:, 0].max()
        self.vymin, self.vymax = self.v[:, 1].min(), self.v[:, 1].max()

        self.fig = plt.figure()

        #self.toshow = {'velocities': False, 'quiver': False, 'trace': None, 'speeds': False}
        self.toshow = {'velocities': False, 'quiver': False, 'trace': None, 'speeds': False, 'pressure': False}

        # For molecule trace
        self._vtrace = None

        # For pressure calculation
        self._pressure = None
        self._wall_momentum = [0.]
        self._wall_momentum_theory = 0.
        self._pressure_t = [0.]

        # Real volume (will be computed in _init)
        self.real_volume = 0.

        # For obstacle collisions
        self.obstacles = []

        # Default coloring
        self.colors = 'velocities'

        # Create display
        self._init()

        self._i = None
        self.animobj = None

        self.highlight_rule = None


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

    def run(self, nsteps=100000, filename=None, blit=False, block=None):
        """Start animation

        nsteps: number of steps
        filename: if not None, movie file to save to (work in progress)
        blit: False by default, does not work always
        block: if True, return only after run is done, if False return immediately
               if None, returns only in interactive mode
        """
        self._i = 0
        self.animobj = animation.FuncAnimation(self.fig, self._update, frames=nsteps, interval=5., repeat=False, blit=blit)
        if filename is not None:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=150, bitrate=500, extra_args=['-filter', 'tblend', '-r', '25'])
            self.animobj._start()
            self.animobj.save(filename, writer=writer)
        else:
            self.animobj._start()
            plt.show(False)
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

        # Recompute volume
        self.real_volume = np.prod(self.L-self.d.mean()) - .5*np.pi*sum(self.d**2)

        # No update during setup
        plt.ioff()
        self.fig.clear()

        show_v = self.toshow['velocities']
        show_s = self.toshow['speeds']
        show_p = self.toshow['pressure']
        show_t = self.toshow['trace'] is not None
        if sum([show_v, show_s, show_p, show_t]) > 1:
            raise RuntimeError('Can show only velocities OR speeds OR pressure')

        if show_v or show_s or show_p or show_t:
            # Create two side-by-side axes, with a histogram of the x-component of the velodicities in the second one. 
            axes = [self.fig.add_subplot(121, aspect='equal', adjustable='box'),
                    self.fig.add_subplot(122)]
            if show_v:
                axes[1].set_xlabel('velocity along x')
            elif show_s:
                axes[1].set_xlabel('speeds')
            elif show_p:
                axes[1].set_xlabel('time')
            elif show_t:
                self.panel = MFPPanel(self)

            if show_p:
                axes[1].set_ylabel('momentum')
            elif show_v or show_s:
                # Y axis does not mean much
                axes[1].get_yaxis().set_visible(False)

            if show_p:
                self._pressure = 0
                self.plot_pressure = axes[1].plot([0], [0], 'b-', label='Total momentum')[0]
                self.plot_pressure_theory = axes[1].plot([0, 0], [0, 0], 'k-', label='Total momentum (theory)')[0]
                axes[1].legend(loc='upper left')
            elif show_v or show_s:
                # Generate initial histogram
                if show_v:
                    f, bins = np.histogram(self.v[:, 0], int(np.sqrt(self.N)), range=(self.vxmin, self.vxmax), density=True)
                else:
                    f, bins = np.histogram(np.sqrt((self.v ** 2).sum(axis=1)), int(np.sqrt(self.N)),
                                           range=(0., self.v2max), density=True)
                binwidth = bins[1] - bins[0]
                # Create bar graph
                self.vhist = axes[1].bar(bins[:-1], f, width=binwidth, align='edge')

        else:
            # Create just one axis - the particle box
            axes = [self.fig.add_subplot(111, aspect='equal', adjustable='box')]

        # Set box size and axis properties
        axes[0].axis([0, self.L[0], 0, self.L[1]])
        axes[0].get_xaxis().set_visible(False)
        axes[0].get_yaxis().set_visible(False)

        # Draw particles
        circles = EllipseCollection(widths=self.d, heights=self.d, angles=0, units='xy',
                                    facecolors='k', offsets=self.r, transOffset=axes[0].transData)
        axes[0].add_collection(circles)

        # Create colormap
        self.cm = plt.get_cmap('plasma')

        self.fig.tight_layout()
        self.circles = circles

        to_return = (circles,)

        # Option to show the trace of one particle (to illustrate random walk)
        if self.toshow['trace'] is not None:
            i = self.toshow['trace']
            self.trace = axes[0].plot([self.r[i, 0]], [self.r[i,1]], 'k-')[0]
            self._vtrace = self.v[i].copy()
            to_return += (self.trace,)

        # Option to show velocity arrows
        if self.toshow['quiver']:
            quiver = plt.quiver(self.r[:, 0], self.r[:, 1], self.v[:, 0], self.v[:, 1], units='xy', scale=35.*self.vRMS/self.L.mean())
            self.quiver = quiver
            to_return += (quiver,)

        self.axes = axes

        if show_s or show_v:
            p0 = axes[0].get_position()
            p1 = axes[1].get_position()
            axes[1].set_position([p1.x0, p0.y0, p0.width, p0.height])
            to_return += (self.vhist,)

        if show_p:
            p0 = axes[0].get_position()
            p1 = axes[1].get_position()
            axes[1].set_position([p1.x0, p0.y0, p0.width, p0.height])
            to_return += (self.plot_pressure, self.plot_pressure_theory)

        if show_t:
            to_return += self.panel.init(axes[1])

        # Process all obstacles (polygons)
        if self.obstacles:
            for obs in self.obstacles:
                vc = obs['vertices']
                axes[0].add_patch(Polygon(vc, facecolor='black'))

        return to_return

    def set_fig_position(self, x, y, dx, dy):
        """Set figure windoe position (might work only with QT backend)"""
        plt.get_current_fig_manager().window.setGeometry(x, y, dx, dy)

    def set_colors(self, colors):
        self.circles.set_facecolors(colors)

    def _update(self, i):
        """Update plot"""
        self.i = i

        # Compute move
        self._step()

        # Velocity statistics
        vmag = np.sqrt((self.v**2).sum(axis=1))
        self.v2max = max(self.v2max, vmag.max())
        self.vxmax = max(self.vxmax, self.v[:, 0].max())
        self.vxmin = min(self.vxmin, self.v[:, 0].min())
        self.vymax = max(self.vymax, self.v[:, 1].max())
        self.vymin = min(self.vymin, self.v[:, 1].min())

        # Move molecules
        self.circles.set_offsets(self.r)

        # Change colours
        if self.colors == 'velocities':
            self.set_colors(self.cm(vmag/self.v2max))

        to_return = (self.circles,)

        show_v = self.toshow['velocities']
        show_s = self.toshow['speeds']
        show_p = self.toshow['pressure']

        if show_v or show_s:
            # Recompute histogram
            nbins = len(self.vhist)
            if show_v:
                f, bins = np.histogram(self.v[:, 0], nbins, range=(self.vxmin, self.vxmax), density=True)
                self.axes[1].set_xlim(self.vxmin, self.vxmax)
            else:
                f, bins = np.histogram(np.sqrt((self.v**2).sum(axis=1)), nbins, range=(0, self.v2max), density=True)
                self.axes[1].set_xlim(0, self.v2max)

            binwidth = bins[1]-bins[0]
            # Update histogram
            for i in range(nbins):
                self.vhist[i].set_height(f[i])
                self.vhist[i].set_width(binwidth)
                self.vhist[i].set_facecolor(self.cm((bins[i]+.5*binwidth)**2/self.v2max))
                self.vhist[i].set_x(bins[i])
            # Adjust vertical extent.
            ylim = self.axes[1].get_ylim()[1]
            dylim = 1.2*f.max() - ylim
            if abs(dylim) > .1*ylim:
                new_ylim = ylim +.1*(dylim)
                self.axes[1].set_ylim(ymax=new_ylim)

            to_return += (self.vhist,)

        if show_p:
            # Update plot data
            self.plot_pressure.set_data(self._pressure_t, self._wall_momentum)
            self.plot_pressure_theory.set_data([0, self._pressure_t[-1]], [0, self._wall_momentum_theory])
            self.axes[1].set_xlim(0, self._pressure_t[-1])
            self.axes[1].set_ylim(np.min(self._wall_momentum), np.max(self._wall_momentum))
            to_return += (self.plot_pressure, self.plot_pressure_theory)

        if self.toshow['trace'] is not None:
            i = self.toshow['trace']
            newv = self.v[i]
            x, y = self.trace.get_data()
            if not np.allclose(self._vtrace, newv):
                x = np.append(x, self.r[i,0])
                y = np.append(y, self.r[i,1])
            else:
                x[-1] = self.r[i,0]
                y[-1] = self.r[i,1]
            self.trace.set_data(x,y)
            self._vtrace = newv.copy()
            to_return += (self.trace,)
            to_return += self.panel.update(i)

        if self.toshow['quiver']:
            self.quiver.set_offsets(self.r)
            self.quiver.set_UVC(self.v[:, 0], self.v[:, 1])
            to_return += (self.quiver,)

        if self.highlight_rule:
            highlighted = eval(self.highlight_rule, self.__dict__)
            lw = 5.
            self.circles.set_lw([lw if h else 0. for h in highlighted])
            self.circles.set_edgecolors(['yellow' if h else 'black' for h in highlighted])
            if self.toshow['quiver']:
                self.quiver.set_UVC(highlighted*self.v[:, 0], highlighted*self.v[:, 1])

        return to_return

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

    def _step(self):
        """Move molecules"""
        self.r += self.dt * self.v
        self.t += self.dt

        self.walls()
        self.collide()
        self.obs_collide()

    def walls(self):
        """
        Process wall collisions.
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

    def obs_collide(self):
        """
        Process collisions with additional rectangular obstacles
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

    def collide(self):
        """
        Process eventual collisions
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

    @staticmethod
    def _disc_collide(r1, v1, d1, m1, r2, v2=None, d2=0., m2=None):
        """
        Collision between two discs.
        """
        if v2 is None:
            v2 = v1*0.

        # Relative positions and velocities
        dv = v2 - v1
        dr = r2 - r1

        # Backtrack
        ndv = norm(dv)
        if ndv == 0:
            # Special case: overlapping particles with same velocities
            ndr = norm(dr)
            offset = .5 * dr * (.5 * (d1 + d2) / ndr - 1.)
            r1out = r1 - offset
            r2out = r2 + offset
            return r1out, v1, r2out, v2

        ru = np.dot(dv, dr) / ndv
        b2 = ru ** 2 + .25 * (d1 + d2) ** 2 - np.dot(dr, dr)
        if b2 < 0:
            # No collision
            return None, None, None, None
        ds = ru + sqrt(b2)

        # Time since collision
        dtc = ds / ndv

        # New collision parameter
        drc = dr - dv * dtc

        # Center of mass velocity
        if m2 is None:
            m1r = 0,
            m2r = 1.
        else:
            m1r = m1/(m1+m2)
            m2r = m2/(m1+m2)

        vcm = m1r * v1 + m2r * v2

        # Velocities after collision
        dvf = dv - 2. * drc * np.dot(dv, drc) / np.dot(drc, drc)
        v1f = vcm - dvf * m2r
        v2f = vcm + dvf * m1r

        # Backtracked positions
        r1f = r1 + (v1f - v1) * dtc
        r2f = r2 + (v2f - v2) * dtc

        return r1f, v1f, r2f, v2f

    def show(self, what=None):
        if what is None:
            plt.ion()
        elif what not in self.toshow:
            raise RuntimeError('Can show only' + str(self.toshow.keys()))
        self.toshow[what] = True
        self._init()
