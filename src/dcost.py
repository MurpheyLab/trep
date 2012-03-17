import numpy as np

from numpy import dot

class DCost(object):
    def __init__(self, xd, ud, Q, R):
        self.xd = xd.copy()
        self.ud = ud.copy()
        self.Q = Q
        self.R = R
        nX = Q.shape[0]
        nU = R.shape[0]
        self.S = np.zeros((nX, nU))

    def l(self, xk, uk, k):
        dx = xk - self.xd[k]
        du = uk - self.ud[k]
        return 0.5 * (dot(dot(dx, self.Q), dx) +
                      dot(dot(du, self.R), du))
        
    def m(self, xkf):
        dx = xkf - self.xd[-1]
        return 0.5 * dot(dot(dx, self.Q), dx)

    def l_dx(self, xk, uk, k):
        dx = xk - self.xd[k]
        return dot(dx, self.Q)

    def l_du(self, xk, uk, k):
        du = uk - self.ud[k]
        return dot(du, self.R)

    def m_dx(self, xkf):
        dx = xkf - self.xd[-1]
        return dot(dx, self.Q)

    def l_dxdx(self, xk, uk, k):
        return self.Q.copy()

    def l_dxdu(self, xk, uk, k):
        return self.S.copy()

    def l_dudu(self, xk, uk, k):
        return self.R.copy()
    
    def m_dxdx(self, xkf):
        return self.Q.copy()
        
