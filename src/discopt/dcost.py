import numpy as np

from numpy import dot

class DCost(object):
    """
    Define the cost for a discrete trajectory optimization for
    tracking a desired trajectory.

    l(x,u,k) = (x - x_d[k]).T * Q * (x - x_d[k]) +
               (u - u_d[k]).T * R * (u - u_d[k])

    m(xf) = (xf - x_d[kf]).T * Qf * (xf - x_d[kf])
    """
    
    def __init__(self, xd, ud, Q, R, Qf=None):
        """ Create a new DCost instance. """
        
        self.xd = xd.copy()
        self.ud = ud.copy()
        self.Q = Q
        if Qf is None:
            self.Qf = self.Q
        else:
            self.Qf = Qf
        self.R = R
        nX = Q.shape[0]
        nU = R.shape[0]
        # This should not be modified as it will not be correctly
        # included in the cost.  It is only saved so that
        # DCost.l_dxdu() can easily return a zero matrix of the
        # correct size.
        self._S = np.zeros((nX, nU))

    def l(self, xk, uk, k):
        """
        Calculate the incremental cost for the given state, input, and
        discrete time.
        """
        dx = xk - self.xd[k]
        du = uk - self.ud[k]
        return 0.5 * (dot(dot(dx, self.Q), dx) +
                      dot(dot(du, self.R), du))
        
    def m(self, xkf):
        """Calculate the terminal cost for the given state."""
        dx = xkf - self.xd[-1]
        return 0.5 * dot(dot(dx, self.Qf), dx)

    def l_dx(self, xk, uk, k):
        """
        Calculate the derivative of the incremental cost with respect
        to the state.
        """
        dx = xk - self.xd[k]
        return dot(dx, self.Q)

    def l_du(self, xk, uk, k):
        """
        Calculate the derivative of the incremental cost with respect
        to the input.
        """
        du = uk - self.ud[k]
        return dot(du, self.R)

    def m_dx(self, xkf):
        """Calculate the derivative of the terminal cost."""
        dx = xkf - self.xd[-1]
        return dot(dx, self.Qf)

    def l_dxdx(self, xk, uk, k):
        """
        Calculate the second derivative of the incremental cost with
        respect to the state.
        """
        return self.Q.copy()

    def l_dxdu(self, xk, uk, k):
        """
        Calculate the second derivative of the incremental cost with
        respect to the state and input.
        """
        return self._S.copy()

    def l_dudu(self, xk, uk, k):
        """
        Calculate the second derivative of the incremental cost with
        respect to the input.
        """
        return self.R.copy()
    
    def m_dxdx(self, xkf):
        """Calculate the second derivative of the terminal cost."""
        return self.Qf.copy()
        
