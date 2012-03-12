import math
import trep
from trep import Potential
from trep._trep import _LinearSpringPotential
import numpy as np
import numpy.linalg

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    _opengl = True
except:
    _opengl = False

class LinearSpring(_LinearSpringPotential, Potential):
    """
    Spring implements a fixed-length, fixed-spring-constant linear
    spring between two coordinate frames:

    V(q) = 0.5*k*(distance(g1,g2) - x0)**2

    Spring potentials can be created with the s-expression:
    (spring frame1 frame2 x0 k [name])
    where frame1/2 are the frames' names.
    """
    def __init__(self, system, frame1, frame2, k, x0=0, name=None):
        """
        Create a new Spring between frame1 and frame2.  The spring is
        automatically added to the system.
        """
        Potential.__init__(self, system, name)
        _LinearSpringPotential.__init__(self)

        if not system.get_frame(frame1):
            raise ValueError("Could not find frame %r" % frame1)
        self._frame1 = system.get_frame(frame1)
        
        if not system.get_frame(frame2):
            raise ValueError("Could not find frame %r" % frame2)
        self._frame2 = system.get_frame(frame2)

        self._k = k
        self._x0 = x0

    @property
    def frame1(self):
        return self._frame1

    @property
    def frame2(self):
        return self._frame2

    @property
    def x0(self):
        return self._x0

    @x0.setter
    def x0(self, x0):
        self._x0 = x0

    @property
    def k(self):
        return self._k
    
    @k.setter
    def k(self, k):
        self._k = k
            
    if _opengl:
        def opengl_draw(self):
            """
            Springs are represented as red lines whose thickness depends
            on how stretched/compressed the spring is.
            """
            frame1 = self.frame1.g()
            frame2 = self.frame2.g()

            dx = np.linalg.norm(self.frame2.p() - self.frame1.p()) / max(self.x0, 1)

            glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT | GL_LINE_BIT)
            glColor3f(1.0, 0.0, 0.0)
            glDisable(GL_LIGHTING)
            glLineWidth(4*math.exp(-0.1*dx))
            glBegin(GL_LINES)
            glVertex3f(frame1[0][3], frame1[1][3], frame1[2][3])
            glVertex3f(frame2[0][3], frame2[1][3], frame2[2][3])    
            glEnd()
            glPopAttrib()
