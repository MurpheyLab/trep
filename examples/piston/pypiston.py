import math
import trep
from OpenGL.GL import *
import numpy as np


class PyPistonForce(trep.Force):
    def __init__(self, system, crank_angle, offset, piston, curve):
        trep.Force.__init__(self, system, name='%s-force' % piston)

        # The combustion is modeled as a simple spline that is
        # dependent on the angle of the crank shaft.
        self.crank_angle = system.get_config(crank_angle)
        # An offset is added to the crank_angle so that each piston
        # can determine the crank shaft's angle relative to its
        # orientation.
        self.offset = offset

        # The combustion force is applied to the Z-axis of the piston.
        self.piston = system.get_frame(piston)
        # The force_curve spline is our model of the combustion force.
        self.force_curve = curve
        # A magnitude to scale the combustion force.
        self.magnitude = 10.0

    def f(self, q):
        # We can improve performance by skipping the work when we know
        # the force will be zero.
        if not self.piston.uses_config(q):
            return 0.0

        # The force is the combustion force applied to the Z axis of
        # the piston's body frame:
        #   F = unhat(g^{-1} dg/dq)  dot (0,0,f,0,0,0)
        dz = np.dot(self.piston.g_inv(),
                    self.piston.g_dq(q))[2,3]
        f = self.force_curve.y((self.crank_angle.q - self.offset) % (2*math.pi))
        return -self.magnitude*dz*f

    def f_dq(self, q, dq1):
        # We really improve performance by doing this check in the
        # first derivative.
        if not self.piston.uses_config(q):
            return 0.0
        if not self.piston.uses_config(dq1) and dq1 != self.crank_angle:
            return 0.0

        # The derivative is found directly by applying the
        # chain/product rule.
        dz = np.dot(self.piston.g_inv(),
                    self.piston.g_dq(q))[2,3]

        dz_dq = (np.dot(self.piston.g_inv_dq(dq1),
                       self.piston.g_dq(q))[2,3]
                 +
                 np.dot(self.piston.g_inv(),
                        self.piston.g_dqdq(q,dq1))[2,3])

        # If this is the derivative with respect to the crank shaft
        # angle, we also have to apply the chain rule to the force.
        angle = (self.crank_angle.q - self.offset) % (2*math.pi)
        f = self.force_curve.y(angle)
        if dq1 == self.crank_angle:
            f_dq = self.force_curve.dy(angle)
        else:
            f_dq = 0.0

        return -self.magnitude*(dz_dq*f + dz*f_dq)

    def f_ddq(self, q, ddq1):
        return 0.0

    def f_du(self, q, du1):
        return 0.0

    def opengl_draw(self):
        glPushMatrix()        

        myangle = (self.crank_angle.q - self.offset) % (2*math.pi)

        f = self.force_curve.y(myangle)
        p1 = self.piston.p()
        p2 = np.dot(self.piston.g(), [0,0,f,1])

        glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT | GL_LINE_BIT)
        glColor3f(1.0, 0.0, 0.0)
        glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        glVertex3f(*p1[:3])
        glVertex3f(*p2[:3])
        glEnd()
        glPopAttrib()
        glPopMatrix()
