import trep
import numpy as np
import dlqr
from numpy import dot
from collections import namedtuple

# There is a lot of input validation here because I've dealt with a
# lot of headaches from mismatching array dimensions and shapes in the
# past.

class DSystem(object):
    """
    Wrapper to treat a trep variational integrator as an arbitrary
    discrete system with
      X[k+1] = f(X[k], U[k], k)
    where
      X[k] = [ Q[k] ; p[k] ; v[k] ]
         v[k] = (rho[k] - rho[k-1]) / (t[k]-t[k-1])         
      U[k] = [ u[k] ; rho[k+1] ]
    """
    # State is X[k] = [ Q[k] ; p[k] ; v[k] ]
    # Input is U[k] = [ u[k] ; rho[k+1] ]

    # You have to provide the time vector to the discrete system so it
    # can correctly calculate the vk[k] part of the state and provide
    # the time to the variational integrator while presenting the
    # f(x[k], u[k], k) interface.
    
    def __init__(self, varint, t):
        self.varint = varint

        self._xk = None
        self._uk = None
        self._time = np.array(t).squeeze()
        self._k = None

        self._nQ = len(self.system.configs)
        self._np = len(self.system.dyn_configs)
        self._nv = len(self.system.kin_configs)
        self._nu = len(self.system.inputs)
        self._nrho = len(self.system.kin_configs)

        self._nX = self._nQ + self._np + self._nv
        self._nU = self._nu + self._nrho

        # Slices of the components in the appropriate vector
        self._slice_Q = slice(0, self._nQ)
        self._slice_Qd = slice(0, self._np)
        self._slice_Qk = slice(self._np, self._nQ)
        self._slice_p = slice(self._nQ, self._nQ + self._np)
        self._slice_v = slice(self._nQ + self._np, self._nX)
        self._slice_u = slice(0, self._nu)
        self._slice_rho = slice(self._nu, self._nU)

        # Named tuples types for function returns
        self.trajectory_return = namedtuple('trajectory', 'X U')
        self.tangent_trajectory_return = namedtuple('tangent_trajectory', 'dX dU')
        self.split_state_return = namedtuple('split_state', 'Q p v')
        self.split_input_return = namedtuple('split_input', 'u rho')
        self.split_trajectory_return = namedtuple('split_trajectory', 'Q p v u rho')
        self.linearization_return = namedtuple('linearization', 'A B')
        self.error_return = namedtuple('errors', 'error exact_norm approx_norm')
        self.feedback_return = namedtuple('feedback', 'Kproj A B')

    @property
    def nX(self):
        """Number of states to the discrete system."""
        return self._nX
    
    @property
    def nU(self):
        """Number of inputs to the discrete system."""
        return self._nU

    @property
    def system(self):
        """The mechanical system modeled by the variational integrator."""
        return self.varint.system

    @property
    def time(self):
        """The time of the discrete steps."""
        return self._time

    @time.setter
    def time(self, t):
        t = np.array(t).squeeze()
        assert t.ndim == 1
        self._time = np.array(t).squeeze()

    @property
    def xk(self):
        """Current state of the system."""
        return self._xk.copy()

    @property
    def uk(self):
        """Current input of the system."""
        return self._uk.copy()

    @property
    def k(self):
        """Current discrete time of the system."""
        return self._k
    

    def kf(self):
        """
        Return the last available state that the system can be set to.
        This is one less than len(self.time).
        """
        return len(self._time)-1


    def build_state(self, Q=None, p=None, v=None):
        """
        Build a state vector from components.  Unspecified components
        are set to zero.
        """
        X = np.zeros((self._nX, ))

        if Q is not None: X[self._slice_Q] = Q
        if p is not None: X[self._slice_p] = p
        if v is not None: X[self._slice_v] = v
        return X         

    def build_input(self, u=None, rho=None):
        """
        Build an input vector from components.  Unspecified components
        are set to zero.
        """
        U = np.zeros((self._nU, ))

        if u is not None: U[self._slice_u] = u
        if rho is not None: U[self._slice_rho] = rho
        return U

    def build_trajectory(self, Q=None, p=None, v=None, u=None, rho=None):
        """
        Combine component trajectories into a state and input
        trajectories.  The state length is the same as the time base,
        the input length is one less than the time base.  Unspecified
        components are set to zero.

        dsys.build_trajectory() -> all zero state and input trajectories
        """

        def bad_arguments(reason):
            string = reason + "\n"
            string += "Arguments: \n"
            for name, value in [ ('Q',Q), ('p',p), ('v',v), ('u',u), ('rho', rho) ]:
                if value is None:
                    string += "  %s: None\n" % name
                else:
                    string += "  %s: %r, shape=%r\n" % (name, type(Q), np.array(value, copy=False).shape)
            raise StandardError(string)

        # Check the lengths for consistency
        if Q is not None and len(Q) != len(self._time):
            bad_arguments("Invalid length for Q (expected %d)" % len(self._time))
        if p is not None and len(p) != len(self._time):
            bad_arguments("Invalid length for p (expected %d)" % len(self._time))
        if v is not None and len(v) != len(self._time):
            bad_arguments("Invalid length for v (expected %d)" % len(self._time))
        if u is not None and len(u) != len(self._time)-1:
            bad_arguments("Invalid length for u (expected %d)" % (len(self._time)-1))
        if rho is not None and len(rho) != len(self._time)-1:
            bad_arguments("Invalid length for rho (expected %d)" % (len(self._time)-1))

        X = np.zeros((len(self._time), self._nX))
        U = np.zeros((len(self._time)-1, self._nU))
        if Q is not None: X[:,self._slice_Q] = Q
        if p is not None: X[:,self._slice_p] = p
        if v is not None: X[:,self._slice_v] = v
        if u is not None:   U[:,self._slice_u] = u
        if rho is not None: U[:,self._slice_rho] = rho
        
        return self.trajectory_return(X,U)
        
    def split_state(self, X=None):
        """
        Split a state vector into its configuration, moementum, and
        kinematic velocity parts.  If X is None, returns zero arrays
        for each component.

        Returns (Q,p,v)
        """
        if X is None:
            X = nd.zeros(self.nX)
        Q = X[self._slice_Q]
        p = X[self._slice_p]
        v = X[self._slice_v]
        return self.split_state_return(Q,p,v)

    def split_input(self, U=None):
        """
        Split a state input vector U into its force and kinematic
        input parts, (u, rho).  If U is empty, returns zero arrays of
        the appropriate size.
        """
        if U is None:
            U = np.zeros(self.nU)
        u = U[self._slice_u]
        rho = U[self._slice_rho]
        return self.split_input_return(u, rho)
        
    def split_trajectory(self, X=None, U=None):
        """
        Split an X,U state trajectory into its Q,p,v,u,rho components.
        If X or U are None, the corresponding components are arrays of
        zero.
        """
        if X is None and U is None:
            X = np.zeros((len(self._time), self.nX))
            U = np.zeros((len(self._time)-1, self.nU))            
        elif X is None:
            X = np.zeros((U.shape[0]+1, self.nX))
        elif U is None:
            U = np.zeros((X.shape[0]+1, self.nU))
            
        Q = X[:,self._slice_Q]
        p = X[:,self._slice_p]
        v = X[:,self._slice_v]
        u = U[:,self._slice_u]
        rho = U[:,self._slice_rho]
        return self.split_trajectory_return(Q,p,v,u,rho)

        
    def set(self, xk, uk, k, xk_hint=None, lambda_hint=None):
        """
        Set the current state, input, and time of the discrete system.
        """
        self._k = k
        self._xk = xk.copy()
        self._uk = uk.copy()
        (q1, p1, v1) = self.split_state(xk)
        (u1, rho2) = self.split_input(uk)
        t1 = self._time[self._k+0]
        t2 = self._time[self._k+1]

        if xk_hint is not None:
            q2_hint = self.split_state(xk_hint)[0]
            q2_hint = q2_hint[:self.varint.nd]
        else:
            q2_hint = None

        self.varint.initialize_from_state(t1, q1, p1)
        self.varint.step(t2, u1, rho2,
                         q2_hint=q2_hint,
                         lambda1_hint=lambda_hint)

        
    def step(self, uk,
             xk_hint=None,
             lambda_hint=None):
        """
        Advance the system to the next discrete time using the given
        values for the input.  
        """
        self._xk = self.f()
        self._uk = uk.copy()
        self._k += 1        
        (u1, rho2) = self.split_input(uk)
        t2 = self._time[self._k+1]

        if xk_hint is not None:
            q2_hint = self.split_state(xk_hint)[0]
            q2_hint = q2_hint[:self.varint.nd]
        else:
            q2_hint = None

        self.varint.step(t2, u1, rho2,
                         q2_hint=q2_hint,
                         lambda1_hint=lambda_hint)


    def f(self):
        """
        Get the next state of the system.
        """
        return self.build_state(self.varint.q2, self.varint.p2, self.varint.v2)


    def fdx(self):
        """
        Get the derivative of the f() with respect to the state.
        Returns a numpy array with derivatives across the rows.
        """
        
        # Initialize with diagonal matrix of -1/dt to get dv/dv block.
        # The other diagonal terms will be overwritten.
        dt = self._time[self._k+1] - self._time[self._k+0]
        fdx = np.zeros((self._nX, self._nX))
        fdx[self._slice_Qd, self._slice_Q] = self.varint.q2_dq1()
        fdx[self._slice_Qd, self._slice_p] = self.varint.q2_dp1()
        # Qk derivatives are all zero
        fdx[self._slice_p, self._slice_Q] = self.varint.p2_dq1()
        fdx[self._slice_p, self._slice_p] = self.varint.p2_dp1()
        fdx[self._slice_v, self._slice_Qk] = np.diag(np.ones(self._nv) * -1.0/dt)
        return fdx


    def fdu(self):
        """
        Get the derivative of the f() with respect to the input.
        Returns a numpy array with derivatives across the rows.
        """
        
        dt = self._time[self._k+1] - self._time[self._k+0]
        fdu = np.zeros((self._nX, self._nU))
        fdu[self._slice_Qd, self._slice_u] = self.varint.q2_du1()
        fdu[self._slice_Qd, self._slice_rho] = self.varint.q2_dk2()
        fdu[self._slice_Qk, self._slice_rho] = np.eye(self._nrho)
        fdu[self._slice_p, self._slice_u] = self.varint.p2_du1()
        fdu[self._slice_p, self._slice_rho] = self.varint.p2_dk2()
        fdu[self._slice_v, self._slice_rho] = np.diag(np.ones(self._nrho) * 1.0/dt)
        return fdu


    def fdxdx(self, z):
        """
        Get the second derivative of f with respect to the state, with
        the outputs multiplied by vector z.  Returns a [nX x nX] numpy
        array.
        """

        zQd = z[self._slice_Qd]
        zp = z[self._slice_p]
        # Don't care about zv or zQk because second derivative is always zero.

        fdxdx = np.zeros((self._nX, self._nX))
        fdxdx[self._slice_Q, self._slice_Q] = (np.inner(zQd, self.varint.q2_dq1dq1()) +
                                               np.inner( zp, self.varint.p2_dq1dq1()))            
        fdxdx[self._slice_Q, self._slice_p] = (np.inner(zQd, self.varint.q2_dq1dp1()) +
                                               np.inner( zp, self.varint.p2_dq1dp1()))
        fdxdx[self._slice_p, self._slice_Q] = fdxdx[self._slice_Q, self._slice_p].T

        fdxdx[self._slice_p, self._slice_p] = (np.inner(zQd, self.varint.q2_dp1dp1()) +
                                               np.inner( zp, self.varint.p2_dp1dp1()))
        return fdxdx

        
    def fdxdu(self, z):
        """
        Get the second derivative of f with respect to the state and
        input, with the outputs multiplied by vector z. Returns a [nX
        x nU] numpy array.
        """        

        zQd = z[self._slice_Qd]
        zp = z[self._slice_p]
        # Don't care about zv or zQk because second derivative is always
        # zero.

        fdxdu = np.zeros((self._nX, self._nU))
        fdxdu[self._slice_Q, self._slice_u]   = (np.inner(zQd, self.varint.q2_dq1du1()) +
                                                 np.inner( zp, self.varint.p2_dq1du1()))
        fdxdu[self._slice_Q, self._slice_rho] = (np.inner(zQd, self.varint.q2_dq1dk2()) +
                                                 np.inner( zp, self.varint.p2_dq1dk2()))
        fdxdu[self._slice_p, self._slice_u]   = (np.inner(zQd, self.varint.q2_dp1du1()) +
                                                 np.inner( zp, self.varint.p2_dp1du1()))
        fdxdu[self._slice_p, self._slice_rho] = (np.inner(zQd, self.varint.q2_dp1dk2()) +
                                                 np.inner( zp, self.varint.p2_dp1dk2()))
        return fdxdu


    def fdudu(self, z):
        """
        Get the second derivative of f with respect to the input, with
        the outputs multiplied by vector z.  Returns a [nU x nU] numpy
        array.
        """

        zQd = z[self._slice_Qd]
        zp = z[self._slice_p]
        # Don't care about zv or zQk because second derivative is always zero.

        fdudu = np.zeros((self._nU, self._nU))
        fdudu[self._slice_u, self._slice_u]   = (np.inner(zQd, self.varint.q2_du1du1()) +
                                                 np.inner( zp, self.varint.p2_du1du1()))
        fdudu[self._slice_u, self._slice_rho] = (np.inner(zQd, self.varint.q2_du1dk2()) +
                                                 np.inner( zp, self.varint.p2_du1dk2()))
        fdudu[self._slice_rho, self._slice_u]   = fdudu[self._slice_u, self._slice_rho].T
        fdudu[self._slice_rho, self._slice_rho] = (np.inner(zQd, self.varint.q2_dk2dk2()) +
                                                   np.inner( zp, self.varint.p2_dk2dk2()))
        return fdudu
    

    def save_state_trajectory(self, filename, X=None, U=None):
        """
        Save a trajectory to a file.
        """
        (Q,p,v,u,rho) = self.split_trajectory(X,U)
        trep.save_trajectory(filename, self.system, self._time, Q, p, v, u, rho)


    def load_state_trajectory(self, filename):
        """
        Load a trajectory from a file.
        """
        (t,Q,p,v,u,rho) = trep.load_trajectory(filename, self.system)
        self.time = t
        return self.build_trajectory(Q,p,v,u,rho)


    def linearize_trajectory(self, X, U):
        """
        Calculate the linearization about a trajectory.  X and U do
        not have to be an exact trajectory of the system.

        Returns (A, B) 
        """
        # Setting the state at every timestep instead of just
        # simulating from the initial condition is more stable and
        # versatile.
        A = np.zeros((len(X)-1, self.nX, self.nX))
        B = np.zeros((len(X)-1, self.nX, self.nU))
        for k in xrange(len(X)-1):
            self.set(X[k], U[k], k,
                     xk_hint=X[k+1])
            A[k] = self.fdx()
            B[k] = self.fdu()
        return self.linearization_return(A,B)


    def project(self, bX, bU, Kproj=None):
        """
        Project bX and bU into a nearby trajectory for the system
        using a linear feedback law:

        X[0] = bX[0]
        U[k] = bU[k] - Kproj * (X[k] - bU[k])
        X[k+1] = f(X[k], U[k], k)

        If no feedback law is specified, one will be created from the
        LQR solution to the linearization of the system about bX and
        bU.  This is typically a bad idea if bX and bU are not very
        close to an actual trajectory for the system.
        """        
        if Kproj is None:
            # Not necessarily a good idea!
            Kproj = self.calc_feedback_controller(bX, bU)
        
        nX = np.zeros(bX.shape)
        nU = np.zeros(bU.shape)

        nX[0] = bX[0]
        for k in range(len(bX)-1):
            nU[k] = bU[k] - dot(Kproj[k], nX[k] - bX[k])
            if k == 0:
                self.set(nX[k], nU[k], k,
                         xk_hint=bX[k+1])
            else:
                self.step(nU[k], xk_hint=bX[k+1])
            nX[k+1] = self.f()
            
        return self.trajectory_return(nX, nU)


    def dproject(self, A, B, bdX, bdU, K):
        """
        Project bdX and bdU into the tangent trajectory space for the
        system about the linearization A,B.
        """
        dX = np.zeros(bdX.shape)
        dU = np.zeros(bdU.shape)
        dX[0] = bdX[0]
        for k in xrange(len(bdX)-1):
            dU[k] = bdU[k] - dot(K[k], dX[k] - bdX[k])
            dX[k+1] = dot(A[k],dX[k]) + dot(B[k],dU[k])
        return self.tangent_trajectory_return(dX, dU)


    def calc_feedback_controller(self, X, U, Q=None, R=None, return_linearization=False):
        """
        Calculate a stabilizing feedback controller for the system
        about a trajectory X and U by solving the discrete LQR problem
        about the linearized system along X and U.  If the LQR weights
        Q and R are not specified, identity matrices are used.
        """
        (A, B) = self.linearize_trajectory(X, U)

        if Q is None:
            _Q = np.eye(self.nX)
            Q = lambda k: _Q
        if R is None:            
            _R = np.eye(self.nU)
            R = lambda k: _R
        
        Kproj = dlqr.solve_tv_lqr(A, B, Q, R)[0]
        if return_linearization:
            return self.feedback_return(Kproj, A, B)
        else:
            return Kproj
        

    def convert_trajectory(self, dsys_a, X, U):
        """
        dsys_b = self

        Maps a trajectory X,U for dsys_a to a trajectory nX, nY for
        dsys_b.
        """

        nX = np.zeros((len(X), self.nX))
        nU = np.zeros((len(U), self.nU))

        (q_a, p_a, v_a, mu_a, rho_a) = dsys_a.split_trajectory(X, U)
        (q_b, p_b, v_b, mu_b, rho_b) = self.split_trajectory(nX, nU)

        def build_map(list_a, list_b):
            a_names = [item.name for item in list_a]
            b_names = [item.name for item in list_b]
            b_to_a_map = {}
            for i, b_name in enumerate(b_names):
                if b_name in a_names:
                    b_to_a_map[i] = a_names.index(b_name)
            return b_to_a_map

        q_map = build_map(dsys_a.system.configs, self.system.configs)
        d_map = build_map(dsys_a.system.dyn_configs, self.system.dyn_configs)
        k_map = build_map(dsys_a.system.kin_configs, self.system.kin_configs)
        u_map = build_map(dsys_a.system.inputs, self.system.inputs)

        if len(q_map) != 0:
            q_b[:, q_map.keys()] = q_a[:, q_map.values()]
        if len(d_map) != 0:
            p_b[:, d_map.keys()] = p_a[:, d_map.values()]
        if len(u_map) != 0:
            mu_b[:, u_map.keys()] = mu_a[:, u_map.values()]
        if len(k_map) != 0:
            v_b[:, k_map.keys()] = v_a[:, k_map.values()]
            rho_b[:, k_map.keys()] = rho_a[:, k_map.values()]

        return self.build_trajectory(q_b, p_b, v_b, mu_b, rho_b)        

        
    def check_fdx(self, xk, uk, k, delta=1e-5):
        """
        Check the first derivative f_dx of the discrete system
        dynamics against a numeric approximation from f().
        """
        
        self.set(xk, uk, k)
        f0 = self.f()
        fdx_exact = self.fdx()

        # Build approximation for fdx 
        fdx_approx = np.zeros((self.nX, self.nX))
        for i1 in range(self.nX):
            dxk = xk.copy()
            dxk[i1] += delta
            self.set(dxk, uk, k)
            dfp = self.f()
            dxk = xk.copy()
            dxk[i1] -= delta
            self.set(dxk, uk, k)
            dfm = self.f()
            fdx_approx[:, i1] = (dfp-dfm)/(2*delta)

        error = np.linalg.norm(fdx_exact - fdx_approx) #/np.linalg.norm(fdx_exact)
        exact_norm = np.linalg.norm(fdx_exact)
        approx_norm = np.linalg.norm(fdx_approx)        
        return self.error_return(error, exact_norm, approx_norm)


    def check_fdu(self, xk, uk, k, delta=1e-5):
        """
        Check the first derivative f_du of the discrete system
        dynamics against a numeric approximation from f().
        """

        self.set(xk, uk, k)
        f0 = self.f()
        fdu_exact = self.fdu()

        # Build approximation for fdu
        fdu_approx = np.zeros((self.nX, self.nU))
        for i1 in range(self.nU):
            duk = uk.copy()
            duk[i1] += delta
            self.set(xk, duk, k)
            dfp = self.f()
            duk = uk.copy()
            duk[i1] -= delta
            self.set(xk, duk, k)
            dfm = self.f()
            fdu_approx[:, i1] = (dfp-dfm)/(2*delta)
            
        error = np.linalg.norm(fdu_exact - fdu_approx) #/np.linalg.norm(fdu_exact)
        exact_norm = np.linalg.norm(fdu_exact)
        approx_norm = np.linalg.norm(fdu_approx)        
        return self.error_return(error, exact_norm, approx_norm)


    def check_fdxdx(self, xk, uk, k, delta=1e-5):
        """
        Check the second derivative f_dxdx of the discrete system
        dynamics against a numeric approximation from f_dx().
        """

        # Build fdxdx_exact
        self.set(xk, uk, k)
        fdxdx_exact = np.zeros((self.nX, self.nX, self.nX))
        for i in range(self.nX):
            z = np.zeros(self.nX)
            z[i] = 1.0
            fdxdx_exact[i,:,:] = self.fdxdx(z)

        # Build fdxdx_approx
        dx0 = np.zeros((self.nX, self.nX, self.nX))
        dx1 = np.zeros((self.nX, self.nX, self.nX))        
        for i in range(self.nX):
            dxk = xk.copy()
            dxk[i] -= delta        
            self.set(dxk, uk, k)            
            dx0[:,:,i] = self.fdx()

            dxk = xk.copy()
            dxk[i] += delta        
            self.set(dxk, uk, k)            
            dx1[:,:,i] = self.fdx()
            
        fdxdx_approx = (dx1 - dx0)/(2*delta)
            
        error = np.linalg.norm(fdxdx_exact - fdxdx_approx)
        exact_norm = np.linalg.norm(fdxdx_exact)
        approx_norm = np.linalg.norm(fdxdx_approx)
        return self.error_return(error, exact_norm, approx_norm)

        
    def check_fdxdu(self, xk, uk, k, delta=1e-5):
        """
        Check the second derivative f_dxdu of the discrete system
        dynamics against a numeric approximation from f_dx().
        """

        # Build fdxdu_exact
        self.set(xk, uk, k)
        fdxdu_exact = np.zeros((self.nX, self.nX, self.nU))
        for i in range(self.nX):
            z = np.zeros(self.nX)
            z[i] = 1.0
            fdxdu_exact[i,:,:] = self.fdxdu(z)

        # Build fdxdu_approx
        dx0 = np.zeros((self.nX, self.nX, self.nU))
        dx1 = np.zeros((self.nX, self.nX, self.nU))        
        for i in range(self.nU):
            duk = uk.copy()
            duk[i] -= delta        
            self.set(xk, duk, k)            
            dx0[:,:,i] = self.fdx()

            duk = uk.copy()
            duk[i] += delta        
            self.set(xk, duk, k)            
            dx1[:,:,i] = self.fdx()
            
        fdxdu_approx = (dx1 - dx0)/(2*delta)
        
        error = np.linalg.norm(fdxdu_exact - fdxdu_approx)
        exact_norm = np.linalg.norm(fdxdu_exact)
        approx_norm = np.linalg.norm(fdxdu_approx)
        return self.error_return(error, exact_norm, approx_norm)
        
        
    def check_fdudu(self, xk, uk, k, delta=1e-5):
        """
        Check the second derivative f_dudu of the discrete system
        dynamics against a numeric approximation from f_du().
        """

        # Build fdudu_exact
        self.set(xk, uk, k)
        fdudu_exact = np.zeros((self.nX, self.nU, self.nU))
        for i in range(self.nX):
            z = np.zeros(self.nX)
            z[i] = 1.0
            fdudu_exact[i,:,:] = self.fdudu(z)

        # Build fdudu_approx
        dx0 = np.zeros((self.nX, self.nU, self.nU))
        dx1 = np.zeros((self.nX, self.nU, self.nU))        
        for i in range(self.nU):
            duk = uk.copy()
            duk[i] -= delta        
            self.set(xk, duk, k)            
            dx0[:,:,i] = self.fdu()

            duk = uk.copy()
            duk[i] += delta        
            self.set(xk, duk, k)            
            dx1[:,:,i] = self.fdu()
            
        fdudu_approx = (dx1 - dx0)/(2*delta)
            
        error = np.linalg.norm(fdudu_exact - fdudu_approx)
        exact_norm = np.linalg.norm(fdudu_exact)
        approx_norm = np.linalg.norm(fdudu_approx)
        return self.error_return(error, exact_norm, approx_norm)
