import trep
import multiprocessing
import _trep
from _trep import _MidpointVI
import numpy as np
import numpy.linalg

from util import dynamics_indexing_decorator

Py_DEBUG = False
try:
    import sysconfig
    if sysconfig.get_config_var('Py_DEBUG'):
        Py_DEBUG = True
except ImportError:
    pass


class MidpointVI(_MidpointVI):
    def __init__(self, system, tolerance=1e-10, num_threads=None):
        _MidpointVI.__init__(self)

        assert isinstance(system, trep.System)
        self._system = system
        self.system.add_structure_changed_func(self._structure_updated)
        self.tolerance = tolerance
        self._structure_updated()

        if num_threads is None and Py_DEBUG:
            print """
            GMVI: Detected debug build, disabling multithreading by
            default.  Use gmvi._set_num_threads(num_threads) to enable
            multithreading.
            """
        else:
            if num_threads is None:
                num_threads = multiprocessing.cpu_count()
            self._set_num_threads(num_threads)

    def _structure_updated(self):
        self._q1 = np.zeros(self.nd+self.nk, np.double, 'C')
        self._q2 = np.zeros(self.nd+self.nk, np.double, 'C')
        self._p1 = np.zeros(self.nd, np.double, 'C')
        self._p2 = np.zeros(self.nd, np.double, 'C')
        self._u1 = np.zeros(self.nu, np.double, 'C')
        self._lambda1 = np.zeros(self.nc, np.double, 'C')

        self._Dh1T = np.zeros((self.nq, self.nc), np.double, 'C')
        self._Dh2 = np.zeros((self.nc, self.nq), np.double, 'C')
        self._f = np.zeros((self.nd+self.nc), np.double, 'C')
        self._Df = np.zeros((self.nd+self.nc, self.nd+self.nc), np.double, 'C')
        self._Df_index = np.zeros((self.nd+self.nc,), np.int, 'C')

        self._DDh1T = np.zeros((self.nq, self.nq, self.nc), np.double, 'C')
        self._M2_lu = np.zeros((self.nd, self.nd), np.double, 'C')
        self._M2_lu_index = np.zeros((self.nd,), np.int, 'C')
        self._proj_lu = np.zeros((self.nc, self.nc), np.double, 'C')
        self._proj_lu_index = np.zeros((self.nc,), np.int, 'C')

        self._q2_dq1 = np.zeros((self.nq, self.nd), np.double, 'C')
        self._q2_dp1 = np.zeros((self.nd, self.nd), np.double, 'C')
        self._q2_du1 = np.zeros((self.nu, self.nd), np.double, 'C')
        self._q2_dk2 = np.zeros((self.nk, self.nd), np.double, 'C')
        self._p2_dq1 = np.zeros((self.nq, self.nd), np.double, 'C')
        self._p2_dp1 = np.zeros((self.nd, self.nd), np.double, 'C')
        self._p2_du1 = np.zeros((self.nu, self.nd), np.double, 'C')
        self._p2_dk2 = np.zeros((self.nk, self.nd), np.double, 'C')
        self._l1_dq1 = np.zeros((self.nq, self.nc), np.double, 'C')
        self._l1_dp1 = np.zeros((self.nd, self.nc), np.double, 'C')
        self._l1_du1 = np.zeros((self.nu, self.nc), np.double, 'C')
        self._l1_dk2 = np.zeros((self.nk, self.nc), np.double, 'C')

        self._q2_dq1dq1 = np.zeros((self.nq, self.nq, self.nd), np.double, 'C')
        self._q2_dq1dp1 = np.zeros((self.nq, self.nd, self.nd), np.double, 'C')
        self._q2_dq1du1 = np.zeros((self.nq, self.nu, self.nd), np.double, 'C')
        self._q2_dq1dk2 = np.zeros((self.nq, self.nk, self.nd), np.double, 'C')
        self._q2_dp1dp1 = np.zeros((self.nd, self.nd, self.nd), np.double, 'C')
        self._q2_dp1du1 = np.zeros((self.nd, self.nu, self.nd), np.double, 'C')
        self._q2_dp1dk2 = np.zeros((self.nd, self.nk, self.nd), np.double, 'C')
        self._q2_du1du1 = np.zeros((self.nu, self.nu, self.nd), np.double, 'C')
        self._q2_du1dk2 = np.zeros((self.nu, self.nk, self.nd), np.double, 'C')
        self._q2_dk2dk2 = np.zeros((self.nk, self.nk, self.nd), np.double, 'C')

        self._p2_dq1dq1 = np.zeros((self.nq, self.nq, self.nd), np.double, 'C')
        self._p2_dq1dp1 = np.zeros((self.nq, self.nd, self.nd), np.double, 'C')
        self._p2_dq1du1 = np.zeros((self.nq, self.nu, self.nd), np.double, 'C')
        self._p2_dq1dk2 = np.zeros((self.nq, self.nk, self.nd), np.double, 'C')
        self._p2_dp1dp1 = np.zeros((self.nd, self.nd, self.nd), np.double, 'C')
        self._p2_dp1du1 = np.zeros((self.nd, self.nu, self.nd), np.double, 'C')
        self._p2_dp1dk2 = np.zeros((self.nd, self.nk, self.nd), np.double, 'C')
        self._p2_du1du1 = np.zeros((self.nu, self.nu, self.nd), np.double, 'C')
        self._p2_du1dk2 = np.zeros((self.nu, self.nk, self.nd), np.double, 'C')
        self._p2_dk2dk2 = np.zeros((self.nk, self.nk, self.nd), np.double, 'C')

        self._l1_dq1dq1 = np.zeros((self.nq, self.nq, self.nc), np.double, 'C')
        self._l1_dq1dp1 = np.zeros((self.nq, self.nd, self.nc), np.double, 'C')
        self._l1_dq1du1 = np.zeros((self.nq, self.nu, self.nc), np.double, 'C')
        self._l1_dq1dk2 = np.zeros((self.nq, self.nk, self.nc), np.double, 'C')
        self._l1_dp1dp1 = np.zeros((self.nd, self.nd, self.nc), np.double, 'C')
        self._l1_dp1du1 = np.zeros((self.nd, self.nu, self.nc), np.double, 'C')
        self._l1_dp1dk2 = np.zeros((self.nd, self.nk, self.nc), np.double, 'C')
        self._l1_du1du1 = np.zeros((self.nu, self.nu, self.nc), np.double, 'C')
        self._l1_du1dk2 = np.zeros((self.nu, self.nk, self.nc), np.double, 'C')
        self._l1_dk2dk2 = np.zeros((self.nk, self.nk, self.nc), np.double, 'C')

        self._DDDh1T = np.zeros((self.nq, self.nq, self.nq, self.nc), np.double, 'C')
        self._DDh2 = np.zeros((self.nc, self.nq, self.nq), np.double, 'C')
        self._temp_ndnc = np.zeros((self.nd, self.nc), np.double, 'C')

        self._D1D1L2_D1fm2 = np.zeros((self.nq, self.nd), np.double, 'C')
        self._D2D1L2_D2fm2 = np.zeros((self.nq, self.nd), np.double, 'C')
        self._D1D2L2 = np.zeros((self.nq, self.nd), np.double, 'C')
        self._D2D2L2 = np.zeros((self.nq, self.nd), np.double, 'C')
        self._D3fm2 = np.zeros((self.nu, self.nd), np.double, 'C')
        self._D1D1D1L2_D1D1fm2 = np.zeros((self.nq, self.nq, self.nd), np.double, 'C')
        self._D1D2D1L2_D1D2fm2 = np.zeros((self.nq, self.nq, self.nd), np.double, 'C')
        self._D2D2D1L2_D2D2fm2 = np.zeros((self.nd, self.nq, self.nq), np.double, 'C')
        self._D1D1D2L2 = np.zeros((self.nq, self.nq, self.nd), np.double, 'C')
        self._D1D2D2L2 = np.zeros((self.nq, self.nq, self.nd), np.double, 'C')
        self._D2D2D2L2 = np.zeros((self.nd, self.nq, self.nq), np.double, 'C')
        self._D1D3fm2 = np.zeros((self.nq, self.nu, self.nd), np.double, 'C')
        self._D2D3fm2 = np.zeros((self.nq, self.nu, self.nd), np.double, 'C')
        self._D3D3fm2 = np.zeros((self.nu, self.nu, self.nd), np.double, 'C')

        self._dp2_dq1_op = np.zeros((self.nd, self.nq, self.nd), np.double, 'C')
        self._dl1_dq1_op = np.zeros((self.nc, self.nq, self.nd), np.double, 'C')
        self._dq2_dq1_op = np.zeros((self.nd, self.nq, self.nd), np.double, 'C')
        self._dq2_dp1_op = np.zeros((self.nd, self.nd, self.nd), np.double, 'C')
        self._dl1_dp1_op = np.zeros((self.nc, self.nd, self.nd), np.double, 'C')
        self._dp2_dp1_op = np.zeros((self.nd, self.nd, self.nd), np.double, 'C')
        self._dq2_du1_op = np.zeros((self.nd, self.nu, self.nd), np.double, 'C')
        self._dl1_du1_op = np.zeros((self.nc, self.nu, self.nd), np.double, 'C')
        self._dp2_du1_op = np.zeros((self.nd, self.nu, self.nd), np.double, 'C')
        self._dq2_dk2_op = np.zeros((self.nd, self.nk, self.nd), np.double, 'C')
        self._dl1_dk2_op = np.zeros((self.nc, self.nk, self.nd), np.double, 'C')
        self._dp2_dk2_op = np.zeros((self.nd, self.nk, self.nd), np.double, 'C')

    def initialize_from_state(self, t1, q1, p1, lambda1=None):
        """
        Initialize the integrator from a known state (time,
        configuration, and momentum.)

        lambda1 can optionally be specified.
        """
        self.t1 = t1
        self.q1 = q1
        self.p1 = p1
        self.t2 = t1
        self.q2 = q1
        self.p2 = p1
        if lambda1 is None:
            lambda1 = np.zeros((self.system.nc,))
        self.lambda1 = lambda1

    def initialize_from_configs(self, t0, q0, t1, q1, lambda1=None):
        """
        Initialize the integrator from two consecutive time and
        configuration pairs.

        This calculates p1 from the two pairs and initializes the
        integrator with the state (t1, q1, p1).

        lambda1 can optionall be specified.
        """
        self.t1 = t0
        self.q1 = q0
        self.t2 = t1
        self.q2 = q1
        self.calc_p2()
        if lambda1 is None:
            lambda1 = np.zeros((self.system.nc,))
        self.lambda1 = lambda1

    def step(self, t2, u1=tuple(), k2=tuple(), max_iterations=200,
             q2_hint=None,
             lambda1_hint=None):
        """
        Step the integrator forward to time t2 .  This solves the DEL
        equation.  The result will be available in gmvi.t2, gmvi.q2,
        gmvi.p2.
        """
        u1 = np.array(u1)
        k2 = np.array(k2)
        assert u1.shape == (self.system.nu,)
        assert k2.shape == (self.nk,)

        # Advance the integrator
        self.q1 = self.q2
        self.p1 = self.p2
        self.u1 = u1
        self._q2[self.nd:] = k2
        self.t1 = self.t2
        self.t2 = t2
        if q2_hint is not None:
            self._q2[:self.nd] = q2_hint[:self.nd]
        if lambda1_hint is not None:
            self.lambda1 = lambda1_hint
        try:
            return self._solve_DEL(max_iterations)
        except ValueError:  # Catch singular derivatives.
            raise trep.ConvergenceError("Singular derivative of DEL at t=%s" % t2)

    def calc_f(self):
        """
        Evaluate the DEL equation at the current
        states.  For dynamically consistent states, this should be
        zero.  Otherwise it is the remainder of the DEL.
        """
        self._calc_f()
        return self._f.copy()

    @property
    def system(self):
        return self._system

    def __repr__(self):
        return "<MidpointVI t1=%f t2=%f nd=%d nk=%d nc=%d nu=%d>" % (
            self.t1, self.t2, self.nd, self.nk, self.nc, self.nu)

    @property
    def nq(self):
        """Number of configuration variables in the system."""
        return self._system.nQ

    @property
    def nd(self):
        """Number of dynamic configuration variables in the system."""
        return self._system.nQd

    @property
    def nk(self):
        """Number of kinematic configuration variables in the system."""
        return self._system.nQk

    @property
    def nu(self):
        """Number of input variables in the system."""
        return self._system.nu

    @property
    def nc(self):
        """Number of constraints in the system."""
        return self._system.nc

    @property
    def q1(self):
        """Configuration at k=1  (ie, 'previous' configuration)"""
        return self._q1.copy()

    @q1.setter
    def q1(self, value):
        self._cache = 0
        self._q1[:] = value[:]

    @property
    def q2(self):
        """Configuration at k=2  (ie, 'current' configuration)"""
        return self._q2.copy()

    @q2.setter
    def q2(self, value):
        self._cache = 0
        self._q2[:] = value[:]

    @property
    def p1(self):
        """Discrete momentum at k=1  (ie, 'previous' momentum)"""
        return self._p1.copy()

    @p1.setter
    def p1(self, value):
        self._cache = 0
        self._p1[:] = value[:]

    @property
    def p2(self):
        """Discrete momentum at k=2  (ie, 'current' momentum)"""
        return self._p2.copy()

    @p2.setter
    def p2(self, value):
        self._cache = 0
        self._p2[:] = value[:]

    @property
    def u1(self):
        """Input at k=1."""
        return self._u1.copy()

    @u1.setter
    def u1(self, value):
        self._cache = 0
        self._u1[:] = value[:]

    @property
    def lambda1(self):
        """Constraint force vector at k=1"""
        return self._lambda1.copy()

    @lambda1.setter
    def lambda1(self, value):
        self._cache = 0
        self._lambda1[:] = value[:]

    @property
    def t1(self):
        """Time for k=1 (ie, 'previous' time)"""
        return self._t1

    @t1.setter
    def t1(self, t):
        self._cache = 0
        self._t1 = t

    @property
    def t2(self):
        """Time for k=2 (ie, 'current' time)"""
        return self._t2

    @t2.setter
    def t2(self, t):
        self._cache = 0
        self._t2 = t

    @property
    def v2(self):
        """Discrete kinematic velocity for k=2 (q2k-q1k/t2-t1)"""
        if self.t2 != self.t1:
            v2 = (self.q2 - self.q1)/(self.t2 - self.t1)
            return v2[self.nd:]
        else:
            return None


    # Derivatives of q2

    @dynamics_indexing_decorator('dq')
    def q2_dq1(self, q=None, q1=None):
        """
        Calculate the derivative of (the dynamic part of) q2 with
        respect to q1.
        """
        self._calc_deriv1()
        return self._q2_dq1[q1, q].T.copy()

    @dynamics_indexing_decorator('dq')
    def q2_dp1(self, q=None, p1=None):
        """
        Calculate the derivative of (the dynamic part of) q2 with
        respect to p1.
        """
        self._calc_deriv1()
        return self._q2_dp1[p1, q].T.copy()

    @dynamics_indexing_decorator('du')
    def q2_du1(self, q=None, u1=None):
        """
        Calculate the derivative of (the dynamic part of) q2 with
        respect to u1.
        """
        self._calc_deriv1()
        return self._q2_du1[u1, q].T.copy()

    @dynamics_indexing_decorator('dk')
    def q2_dk2(self, q=None, k2=None):
        """
        Calculate the derivative of (the dynamic part of) q2 with
        respect to k2 (the kinematic configuration at k=2).
        """
        self._calc_deriv1()
        return self._q2_dk2[k2, q].T.copy()

    @dynamics_indexing_decorator('dqq')
    def q2_dq1dq1(self, q=None, q1_1=None, q1_2=None):
        """
        Calculate the second derivative of (the dynamic part of) q2
        with respect to q1.
        """
        self._calc_deriv2()
        return self._q2_dq1dq1[q1_1, q1_2, q].copy()

    @dynamics_indexing_decorator('dqd')
    def q2_dq1dp1(self, q=None, q1=None, p1=None):
        """
        Calculate the second derivative of (the dynamic part of) q2
        with respect to q1 and p1.
        """
        self._calc_deriv2()
        return self._q2_dq1dp1[q1, p1, q].copy()

    @dynamics_indexing_decorator('ddd')
    def q2_dp1dp1(self, q=None, p1_1=None, p1_2=None):
        """
        Calculate the second derivative of (the dynamic part of) q2
        with respect to p1.
        """
        self._calc_deriv2()
        return self._q2_dp1dp1[p1_1, p1_2, q].copy()

    @dynamics_indexing_decorator('dqu')
    def q2_dq1du1(self, q=None, q1=None, u1=None):
        """
        Calculate the second derivative of (the dynamic part of) q2
        with respect to q1 and u1.
        """
        self._calc_deriv2()
        return self._q2_dq1du1[q1, u1, q].copy()

    @dynamics_indexing_decorator('ddu')
    def q2_dp1du1(self, q=None, p1=None, u1=None):
        """
        Calculate the second derivative of (the dynamic part of) q2
        with respect to u1 and p1.
        """
        self._calc_deriv2()
        return self._q2_dp1du1[p1, u1, q].copy()

    @dynamics_indexing_decorator('duu')
    def q2_du1du1(self, q=None, u1_1=None, u1_2=None):
        """
        Calculate the second derivative of (the dynamic part of) q2
        with respect to u1.
        """
        self._calc_deriv2()
        return self._q2_du1du1[u1_1, u1_2, q].copy()

    @dynamics_indexing_decorator('dqk')
    def q2_dq1dk2(self, q=None, q1=None, k2=None):
        """
        Calculate the second derivative of (the dynamic part of) q2
        with respect to k2 and q1.

        (k2 is the kinematic part of the configuration at k=2)
        """
        self._calc_deriv2()
        return self._q2_dq1dk2[q1, k2, q].copy()

    @dynamics_indexing_decorator('ddk')
    def q2_dp1dk2(self, q=None, p1=None, k2=None):
        """
        Calculate the second derivative of (the dynamic part of) q2
        with respect to k2 and p1.

        (k2 is the kinematic part of the configuration at k=2)
        """
        self._calc_deriv2()
        return self._q2_dp1dk2[p1, k2, q].copy()

    @dynamics_indexing_decorator('duk')
    def q2_du1dk2(self, q=None, u1=None, k2=None):
        """
        Calculate the second derivative of (the dynamic part of) q2
        with respect to k2 and u1.

        (k2 is the kinematic part of the configuration at k=2)
        """
        self._calc_deriv2()
        return self._q2_du1dk2[u1, k2, q].copy()

    @dynamics_indexing_decorator('dkk')
    def q2_dk2dk2(self, q=None, k2_1=None, k2_2=None):
        """
        Calculate the second derivative of (the dynamic part of) q2
        with respect to k21.

        (k2 is the kinematic part of the configuration at k=2)
        """
        self._calc_deriv2()
        return self._q2_dk2dk2[k2_1, k2_2, q].copy()


    # Derivatives of p2

    @dynamics_indexing_decorator('dq')
    def p2_dq1(self, p=None, q1=None):
        """
        Calculate the derivative of p2 with respect to q1.
        """
        self._calc_deriv1()
        return self._p2_dq1[q1, p].T.copy()

    @dynamics_indexing_decorator('dd')
    def p2_dp1(self, p=None, p1=None):
        """
        Calculate the derivative of p2 with respect to p1.
        """
        self._calc_deriv1()
        return self._p2_dp1[p1, p].T.copy()

    @dynamics_indexing_decorator('du')
    def p2_du1(self, p=None, u1=None):
        """
        Calculate the derivative of p2 with respect to u1.
        """
        self._calc_deriv1()
        return self._p2_du1[u1, p].T.copy()

    @dynamics_indexing_decorator('dk')
    def p2_dk2(self, p=None, k2=None):
        """
        Calculate the derivative of p2 with respect to k.

        (k2 is the kinematic part of the configuration at k=2)
        """
        self._calc_deriv1()
        return self._p2_dk2[k2, p].T.copy()

    @dynamics_indexing_decorator('dqq')
    def p2_dq1dq1(self, p=None, q1_1=None, q1_2=None):
        """
        Calculate the second derivative of p2 with respect to q1.
        """
        self._calc_deriv2()
        return self._p2_dq1dq1[q1_1, q1_2, p].copy()

    @dynamics_indexing_decorator('dqd')
    def p2_dq1dp1(self, p=None, q1=None, p1=None):
        """
        Calculate the second derivative of p2 with respect to q1 and
        p1.
        """
        self._calc_deriv2()
        return self._p2_dq1dp1[q1, p1, p].copy()

    @dynamics_indexing_decorator('ddd')
    def p2_dp1dp1(self, p=None, p1_1=None, p1_2=None):
        """
        Calculate the second derivative of p2 with respect to p1.
        """
        self._calc_deriv2()
        return self._p2_dp1dp1[p1_1, p1_2, p].copy()

    @dynamics_indexing_decorator('dqu')
    def p2_dq1du1(self, p=None, q1=None, u1=None):
        """
        Calculate the second derivative of p2 with respect to q1 and
        u1.
        """
        self._calc_deriv2()
        return self._p2_dq1du1[q1, u1, p].copy()

    @dynamics_indexing_decorator('ddu')
    def p2_dp1du1(self, p=None, p1=None, u1=None):
        """
        Calculate the second derivative of p2 with respect to p1 and
        u1.
        """
        self._calc_deriv2()
        return self._p2_dp1du1[p1, u1, p].copy()

    @dynamics_indexing_decorator('duu')
    def p2_du1du1(self, p=None, u1_1=None, u1_2=None):
        """
        Calculate the second derivative of p2 with respect to u1.
        """
        self._calc_deriv2()
        return self._p2_du1du1[u1_1, u1_2, p].copy()

    @dynamics_indexing_decorator('dqk')
    def p2_dq1dk2(self, p=None, q1=None, k2=None):
        """
        Calculate the second derivative of p2 with respect to k2 and
        q1.

        (k2 is the kinematic part of the configuration at k=2)
        """
        self._calc_deriv2()
        return self._p2_dq1dk2[q1, k2, p].copy()

    @dynamics_indexing_decorator('ddk')
    def p2_dp1dk2(self, p=None, p1=None, k2=None):
        """
        Calculate the second derivative of p2 with respect to k2 and
        p1.

        (k2 is the kinematic part of the configuration at k=2)
        """
        self._calc_deriv2()
        return self._p2_dp1dk2[p1, k2, p].copy()

    @dynamics_indexing_decorator('duk')
    def p2_du1dk2(self, p=None, u1=None, k2=None):
        """
        Calculate the second derivative of p2 with respect to k2 and
        u1.

        (k2 is the kinematic part of the configuration at k=2)
        """
        self._calc_deriv2()
        return self._p2_du1dk2[u1, k2, p].copy()

    @dynamics_indexing_decorator('dkk')
    def p2_dk2dk2(self, p=None, k2_1=None, k2_2=None):
        """
        Calculate the second derivative of p2 with respect to k2.

        (k2 is the kinematic part of the configuration at k=2)
        """
        self._calc_deriv2()
        return self._p2_dk2dk2[k2_1, k2_2, p].copy()


    # Derivatives of lambda1

    @dynamics_indexing_decorator('cq')
    def lambda1_dq1(self, constraint=None, q1=None):
        """
        Calculate the derivative of lambda1 with respect to q1.
        """
        self._calc_deriv1()
        return self._l1_dq1[q1, constraint].T.copy()

    @dynamics_indexing_decorator('cd')
    def lambda1_dp1(self, constraint=None, p1=None):
        """
        Calculate the derivative of lambda1 with respect to p1.
        """
        self._calc_deriv1()
        return self._l1_dp1[p1, constraint].T.copy()

    @dynamics_indexing_decorator('cu')
    def lambda1_du1(self, constraint=None, u1=None):
        """
        Calculate the derivative of lambda1 with respect to u1.
        """
        self._calc_deriv1()
        return self._l1_du1[u1, constraint].T.copy()

    @dynamics_indexing_decorator('ck')
    def lambda1_dk2(self, constraint=None, k2=None):
        """
        Calculate the derivative of lambda1 with respect to k2.

        (k2 is the kinematic part of the configuration at k=2)
        """
        self._calc_deriv1()
        return self._l1_dk2[k2, constraint].T.copy()

    @dynamics_indexing_decorator('cqq')
    def lambda1_dq1dq1(self, constraint=None, q1_1=None, q1_2=None):
        """
        Calculate the second derivative of lambda1 with respect to q1.
        """
        self._calc_deriv2()
        return self._l1_dq1dq1[q1_1, q1_2, constraint].copy()

    @dynamics_indexing_decorator('cqd')
    def lambda1_dq1dp1(self, constraint=None, q1=None, p1=None):
        """
        Calculate the second derivative of lambda1 with respect to q1
        and p1.
        """
        self._calc_deriv2()
        return self._l1_dq1dp1[q1, p1, constraint].copy()

    @dynamics_indexing_decorator('cdd')
    def lambda1_dp1dp1(self, constraint=None, p1_1=None, p1_2=None):
        """
        Calculate the second derivative of lambda1 with respect to p1.
        """
        self._calc_deriv2()
        return self._l1_dp1dp1[p1_1, p1_2, constraint].copy()

    @dynamics_indexing_decorator('cqu')
    def lambda1_dq1du1(self, constraint=None, q1=None, u1=None):
        """
        Calculate the second derivative of lambda1 with respect to q1
        and u1.
        """
        self._calc_deriv2()
        return self._l1_dq1du1[q1, u1, constraint].copy()

    @dynamics_indexing_decorator('cdu')
    def lambda1_dp1du1(self, constraint=None, p1=None, u1=None):
        """
        Calculate the second derivative of lambda1 with respect to p1
        and u1.
        """
        self._calc_deriv2()
        return self._l1_dp1du1[p1, u1, constraint].copy()

    @dynamics_indexing_decorator('cuu')
    def lambda1_du1du1(self, constraint=None, u1_1=None, u1_2=None):
        """
        Calculate the second derivative of lambda1 with respect to u1.
        """
        self._calc_deriv2()
        return self._l1_du1du1[u1_1, u1_2, constraint].copy()

    @dynamics_indexing_decorator('cqk')
    def lambda1_dq1dk2(self, constraint=None, q1=None, k2=None):
        """
        Calculate the second derivative of lambda1 with respect to q1
        and k2.

        (k2 is the kinematic part of the configuration at k=2)
        """
        self._calc_deriv2()
        return self._l1_dq1dk2[q1, k2, constraint].copy()

    @dynamics_indexing_decorator('cdk')
    def lambda1_dp1dk2(self, constraint=None, p1=None, k2=None):
        """
        Calculate the second derivative of lambda1 with respect to p1
        and k2.

        (k2 is the kinematic part of the configuration at k=2)
        """
        self._calc_deriv2()
        return self._l1_dp1dk2[p1, k2, constraint].copy()

    @dynamics_indexing_decorator('cuk')
    def lambda1_du1dk2(self, constraint=None, u1=None, k2=None):
        """
        Calculate the second derivative of lambda1 with respect to u1
        and k2.

        (k2 is the kinematic part of the configuration at k=2)
        """
        self._calc_deriv2()
        return self._l1_du1dk2[u1, k2, constraint].copy()

    @dynamics_indexing_decorator('ckk')
    def lambda1_dk2dk2(self, constraint=None, k2_1=None, k2_2=None):
        """
        Calculate the second derivative of lambda1 with respect to k2.

        (k2 is the kinematic part of the configuration at k=2)
        """
        self._calc_deriv2()
        return self._l1_dk2dk2[k2_1, k2_2, constraint].copy()
