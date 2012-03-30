import trep
import math
import numpy as np
from OpenGL.GL import *
import _piston


class CPistonForce(_piston._PistonForce, trep.Force):
    def __init__(self, system, crank_angle, offset, piston, curve):
        trep.Force.__init__(self, system, name='%s-force' % piston)
        _piston._PistonForce.__init__(self)

        # These are the same as the pure python implementation.  
        self._crank_angle = system.get_config(crank_angle)
        self._offset = offset
        self._piston = system.get_frame(piston)
        self._combustion_model = curve
        self._magnitude = 10.0


    def opengl_draw(self):
        glPushMatrix()        

        myangle = (self._crank_angle.q - self._offset) % (2*math.pi)

        f = self._combustion_model.y(myangle)
        p1 = self._piston.p()
        p2 = np.dot(self._piston.g(), [0,0,f,1])

        glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT | GL_LINE_BIT)
        glColor3f(1.0, 0.0, 0.0)
        glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        glVertex3f(*p1[:3])
        glVertex3f(*p2[:3])
        glEnd()
        glPopAttrib()
        glPopMatrix()

