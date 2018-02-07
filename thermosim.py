import numpy as np
import sys
from matplotlib.collections import EllipseCollection
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.spatial.distance import squareform, pdist


def norm(x):
    return np.sqrt(np.dot(x, x))


def sqrt(x):
    """Safe square root"""
    return np.sqrt(np.clip(x, 0, np.inf))


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

        self.toshow = {'velocities': False, 'quiver': False}

        self.colors = 'velocities'
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
        """Initialise display"""
        plt.ioff()
        self.fig.clear()
        if self.toshow['velocities']:
            axes = [self.fig.add_subplot(121, aspect='equal', adjustable='box'),
                    self.fig.add_subplot(122)]
            axes[1].set_xlabel('velocity along x')
            axes[1].get_yaxis().set_visible(False)
            f, bins = np.histogram(self.v[:, 0], int(np.sqrt(self.N)), range=(self.vxmin, self.vxmax), density=True)
            binwidth = bins[1]-bins[0]
            self.vhist = axes[1].bar(bins[:-1], f, width=binwidth, align='edge')
        else:
            axes = [self.fig.add_subplot(111, aspect='equal', adjustable='box')]


        axes[0].axis([0, self.L[0], 0, self.L[1]])
        axes[0].get_xaxis().set_visible(False)
        axes[0].get_yaxis().set_visible(False)
        circles = EllipseCollection(widths=self.d, heights=self.d, angles=0, units='xy',
                                    facecolors='k', offsets=self.r, transOffset=axes[0].transData)
        axes[0].add_collection(circles)

        self.cm = plt.get_cmap('plasma')

        self.fig.tight_layout()
        self.circles = circles

        to_return = (circles,)

        if self.toshow['quiver']:
            quiver = plt.quiver(self.r[:, 0], self.r[:, 1], self.v[:, 0], self.v[:, 1], units='xy', scale=35.*self.vRMS/self.L.mean())
            self.quiver = quiver
            to_return += (quiver,)

        self.axes = axes

        if self.toshow['velocities']:
            p0 = axes[0].get_position()
            p1 = axes[1].get_position()
            axes[1].set_position([p1.x0, p0.y0, p0.width, p0.height])
            to_return += (self.vhist,)

        return to_return

    def set_fig_position(self, x, y, dx, dy):
        """Set figure windoe position (might work only with QT backend)"""
        plt.get_current_fig_manager().window.setGeometry(x, y, dx, dy)

    def set_colors(self, colors):
        self.circles.set_facecolors(colors)

    def _update(self, i):
        """Update plot"""
        self.i = i
        self._step()
        vmag = np.sqrt((self.v**2).sum(axis=1))
        self.v2max = max(self.v2max, vmag.max())
        self.vxmax = max(self.vxmax, self.v[:, 0].max())
        self.vxmin = min(self.vxmin, self.v[:, 0].min())
        self.vymax = max(self.vymax, self.v[:, 1].max())
        self.vymin = min(self.vymin, self.v[:, 1].min())

        self.circles.set_offsets(self.r)
        if self.colors == 'velocities':
            self.set_colors(self.cm(vmag/self.v2max))

        to_return = (self.circles,)


        if self.toshow['velocities']:
            nbins = len(self.vhist)
            f, bins = np.histogram(self.v[:, 0], nbins, range=(self.vxmin, self.vxmax), density=True)
            binwidth = bins[1]-bins[0]
            for i in range(nbins):
                self.vhist[i].set_height(f[i])
                self.vhist[i].set_width(binwidth)
                self.vhist[i].set_facecolor(self.cm((bins[i]+.5*binwidth)**2/self.v2max))
                self.vhist[i].set_x(bins[i])
            self.axes[1].set_xlim(self.vxmin, self.vxmax)
            to_return += (self.vhist,)

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

    def _step(self):
        """Move molecules"""
        self.r += self.dt * self.v
        self.t += self.dt

        self.walls()
        self.collide()

    def walls(self):
        """
        Process wall collisions.
        """
        for dim in range(self.ndim):
            d0 = self.r[:, dim] - .5*self.d - self.bounds[dim][0]
            self.r[d0 < 0, dim] -= 2*d0[d0 < 0]
            self.v[d0 < 0, dim] *= -1
            d1 = self.r[:, dim] + .5*self.d - self.bounds[dim][1]
            self.r[d1 > 0, dim] -= 2*d1[d1 > 0]
            self.v[d1 > 0, dim] *= -1

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

    def show(self, what=None):
        if what is None:
            plt.ion()
        elif what not in self.toshow:
            raise RuntimeError('Can show only' + str(self.toshow.keys()))
        self.toshow[what] = True
        self._init()
