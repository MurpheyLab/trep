import math
import trep
from trep import Force
from trep._trep import _LinearDamperForce
import numpy as np

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    _opengl = True
except:
    _opengl = False

class LinearDamper(_LinearDamperForce, Force):
    def __init__(self, system, frame1, frame2, c, name=None):
        """
        Create a new viscous Damper between frame1 and frame2.  The damper is
        automatically added to the system.
        """
        Force.__init__(self, system, name)
        _LinearDamperForce.__init__(self)
        
        self._coefficient = c

        if not system.get_frame(frame1):
            raise ValueError("Could not find frame %r" % frame1)
        self._frame1 = system.get_frame(frame1)
        
        if not system.get_frame(frame2):
            raise ValueError("Could not find frame %r" % frame2)
        self._frame2 = system.get_frame(frame2)

        self._path = trep.TapeMeasure(system, (self._frame1, self._frame2))

        if _opengl:
            self.measure = trep.TapeMeasure(system, (self._frame1, self._frame2))

    @property
    def c(self):
        return self._coefficient

    @c.setter
    def c(self, value):
        self._coefficient = float(value)

    @property 
    def frame1(self):
        return self._frame1

    @property 
    def frame2(self):
        return self._frame2

    if _opengl:
        def opengl_draw(self):
            """
            Dampers are represented as blue lines whose thickness depends
            on how much force the damper is applying.
            """
            frame1 = self.frame1.g()
            frame2 = self.frame2.g()
            dx = self.measure.velocity()

            glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT | GL_LINE_BIT)
            glColor3f(0.0, 0.0, 1.0)
            glDisable(GL_LIGHTING)
            glLineWidth(4*math.exp(-0.1*dx))
            glBegin(GL_LINES)
            glVertex3f(frame1[0][3], frame1[1][3], frame1[2][3])
            glVertex3f(frame2[0][3], frame2[1][3], frame2[2][3])    
            glEnd()
            glPopAttrib()
