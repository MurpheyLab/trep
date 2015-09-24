import trep
from trep import Constraint
from trep._trep import _DistanceConstraint

try:
    from OpenGL.GLUT import *
    from OpenGL.GLU import *
    from OpenGL.GL import *
    _opengl = True
except:
    _opengl = False


class Distance(_DistanceConstraint, Constraint):
    def __init__(self, system, frame1, frame2, distance, name=None):
        Constraint.__init__(self, system, name)
        _DistanceConstraint.__init__(self)

        assert frame1 is not None
        self._frame1 = system.get_frame(frame1)

        assert frame2 is not None
        self._frame2 = system.get_frame(frame2)

        if isinstance(distance, str):
            self._config = trep.Config(system=self.system, name=distance, kinematic=True)
        else:
            self._config = None
            self._distance = distance

        
    def __repr__(self):
        if self.config:
            return "<DistanceConstraint '%s' '%s' '%s'>" % (
                self.frame1.name,
                self.config.name,
                self.frame2.name)
        else:
            return "<DistanceConstraint '%s' %f '%s'>" % (
                self.frame1.name,
                self.distance,
                self.frame2.name)        

    @property
    def config(self):
        return self._config

    @property
    def frame1(self):
        return self._frame1

    @property
    def frame2(self):
        return self._frame2

    @property 
    def distance(self):
        """Returns the constraint's distance."""
        if self.config:
            return self.config.q
        else:
            return self._distance

    @distance.setter
    def distance(self, value):
        """Set the constraint's distance."""
        if self.config:
            self.config.q = value
        else:
            self._distance = value


    def get_actual_distance(self):
        """Return the actual distance between frame1 and frame2."""
        p1 = self.frame1.p()
        p2 = self.frame2.p()

        distance = ((p1[0]-p2[0])**2.0 + 
                    (p1[1]-p2[1])**2.0 + 
                    (p1[2]-p2[2])**2.0)**0.5
        return distance

    if _opengl:
        def opengl_draw(self):
            frame1 = self.frame1.g()
            frame2 = self.frame2.g()

            glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT )
            glColor3f(0.0, 0.0, 1.0)
            glDisable(GL_LIGHTING)
            glBegin(GL_LINES)
            glVertex3f(frame1[0][3], frame1[1][3], frame1[2][3])
            glVertex3f(frame2[0][3], frame2[1][3], frame2[2][3])    
            glEnd()
            glPopAttrib()
            

