from typing import IO, Optional, Union

import numpy as np
from time import time

from ase import Atoms
from ase.optimize.optimize import Optimizer
from ase.utils.linesearch import LineSearch


class LBFGS(Optimizer):
    """Limited memory BFGS optimizer.

    A limited memory version of the bfgs algorithm. Unlike the bfgs algorithm
    used in bfgs.py, the inverse of Hessian matrix is updated.  The inverse
    Hessian is represented only as a diagonal matrix to save memory

    """

    def __init__(
        self,
        atoms: Atoms,
        restart: Optional[str] = None,
        logfile: Union[IO, str] = '-',
        trajectory: Optional[str] = None,
        maxstep: Optional[float] = None,
        memory: int = 100,
        damping: float = 1.0,
        alpha: float = 70.0,
        use_line_search: bool = False,
        **kwargs,
    ):
        """

        Parameters
        ----------
        atoms: :class:`~ase.Atoms`
            The Atoms object to relax.

        restart: str
            JSON file used to store vectors for updating the inverse of
            Hessian matrix. If set, file with such a name will be searched
            and information stored will be used, if the file exists.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory: string
            Trajectory file used to store optimisation path.

        maxstep: float
            How far is a single atom allowed to move. This is useful for DFT
            calculations where wavefunctions can be reused if steps are small.
            Default is 0.2 Angstrom.

        memory: int
            Number of steps to be stored. Default value is 100. Three numpy
            arrays of this length containing floats are stored.

        damping: float
            The calculated step is multiplied with this number before added to
            the positions.

        alpha: float
            Initial guess for the Hessian (curvature of energy surface). A
            conservative value of 70.0 is the default, but number of needed
            steps to converge might be less if a lower value is used. However,
            a lower value also means risk of instability.

        kwargs : dict, optional
            Extra arguments passed to
            :class:`~ase.optimize.optimize.Optimizer`.

        """
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, **kwargs)

        if maxstep is not None:
            self.maxstep = maxstep
        else:
            self.maxstep = self.defaults['maxstep']

        if self.maxstep > 1.0:
            raise ValueError('You are using a much too large value for ' +
                             'the maximum step size: %.1f Angstrom' %
                             self.maxstep)

        self.memory = memory
        # Initial approximation of inverse Hessian 1./70. is to emulate the
        # behaviour of BFGS. Note that this is never changed!
        self.H0 = 1. / alpha
        self.damping = damping
        self.use_line_search = use_line_search
        self.p = None
        self.function_calls = 0
        self.force_calls = 0
        self.optimizable = atoms

    def initialize(self):
        """Initialize everything so no checks have to be done in step"""
        self.iteration = 0
        self.s = []
        self.y = []
        # Store also rho, to avoid calculating the dot product again and
        # again.
        self.rho = []

        self.r0 = None
        self.f0 = None
        self.e0 = None
        self.task = 'START'
        self.load_restart = False
        
        self.position_history = []
        self.score_history = []
        self.time_history = []
        self.time0 = time()

    def read(self):
        """Load saved arrays to reconstruct the Hessian"""
        self.iteration, self.s, self.y, self.rho, \
            self.r0, self.f0, self.e0, self.task = self.load()
        self.load_restart = True

    def step(self, forces=None):
        """Take a single step

        Use the given forces, update the history and calculate the next step --
        then take it"""

        if forces is None:
            forces = self.optimizable.get_forces()

        pos = self.optimizable.get_positions()

        self.update(pos, forces, self.r0, self.f0)

        s = self.s
        y = self.y
        rho = self.rho
        H0 = self.H0

        loopmax = np.min([self.memory, self.iteration])
        a = np.empty((loopmax,), dtype=np.float64)

        # ## The algorithm itself:
        q = -forces.reshape(-1)
        for i in range(loopmax - 1, -1, -1):
            a[i] = rho[i] * np.dot(s[i], q)
            q -= a[i] * y[i]
        z = H0 * q

        for i in range(loopmax):
            b = rho[i] * np.dot(y[i], z)
            z += s[i] * (a[i] - b)

        self.p = - z.reshape((-1, 3))
        # ##

        g = -forces
        if self.use_line_search is True:
            e = self.func(pos)
            self.line_search(pos, g, e)
            dr = (self.alpha_k * self.p).reshape(len(self.optimizable), -1)
        else:
            self.force_calls += 1
            self.function_calls += 1
            dr = self.determine_step(self.p) * self.damping
        self.optimizable.set_positions(pos + dr)

        self.iteration += 1
        self.r0 = pos
        self.f0 = -g
        self.dump((self.iteration, self.s, self.y,
                   self.rho, self.r0, self.f0, self.e0, self.task))
        
        self.position_history.append(self.optimizable.get_positions())
        self.score_history.append({"LBFGS metric": self.optimizable.get_potential_energy()})
        self.time_history.append(time() - self.time0)

    def determine_step(self, dr):
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """
        steplengths = (dr**2).sum(1)**0.5
        longest_step = np.max(steplengths)
        if longest_step >= self.maxstep:
            dr *= self.maxstep / longest_step

        return dr

    def update(self, pos, forces, r0, f0):
        """Update everything that is kept in memory

        This function is mostly here to allow for replay_trajectory.
        """
        if self.iteration > 0:
            s0 = pos.reshape(-1) - r0.reshape(-1)
            self.s.append(s0)

            # We use the gradient which is minus the force!
            y0 = f0.reshape(-1) - forces.reshape(-1)
            self.y.append(y0)

            rho0 = 1.0 / np.dot(y0, s0)
            self.rho.append(rho0)

        if self.iteration > self.memory:
            self.s.pop(0)
            self.y.pop(0)
            self.rho.pop(0)

    def replay_trajectory(self, traj):
        """Initialize history from old trajectory."""
        if isinstance(traj, str):
            from ase.io.trajectory import Trajectory
            traj = Trajectory(traj, 'r')
        r0 = None
        f0 = None
        # The last element is not added, as we get that for free when taking
        # the first qn-step after the replay
        for i in range(len(traj) - 1):
            pos = traj[i].get_positions()
            forces = traj[i].get_forces()
            self.update(pos, forces, r0, f0)
            r0 = pos.copy()
            f0 = forces.copy()
            self.iteration += 1
        self.r0 = r0
        self.f0 = f0

    def func(self, x):
        """Objective function for use of the optimizers"""
        self.optimizable.set_positions(x.reshape(-1, 3))
        self.function_calls += 1
        return self.optimizable.get_potential_energy()

    def fprime(self, x):
        """Gradient of the objective function for use of the optimizers"""
        self.optimizable.set_positions(x.reshape(-1, 3))
        self.force_calls += 1
        # Remember that forces are minus the gradient!
        return - self.optimizable.get_forces().reshape(-1)

    def line_search(self, r, g, e):
        self.p = self.p.ravel()
        p_size = np.sqrt((self.p**2).sum())
        if p_size <= np.sqrt(len(self.optimizable) * 1e-10):
            self.p /= (p_size / np.sqrt(len(self.optimizable) * 1e-10))
        g = g.ravel()
        r = r.ravel()
        ls = LineSearch()
        self.alpha_k, e, self.e0, self.no_update = \
            ls._line_search(self.func, self.fprime, r, self.p, g, e, self.e0,
                            maxstep=self.maxstep, c1=.23,
                            c2=.46, stpmax=50.)
        if self.alpha_k is None:
            raise RuntimeError('LineSearch failed!')