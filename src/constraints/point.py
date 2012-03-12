import trep
from trep import Constraint
from trep._trep import _PointConstraint
import numpy as np

class Point(_PointConstraint, Constraint):
    """
    The poorly named 'Point' constraint enforces the constraint:

    g_1 dot n dot (p1 - p2) = 0

    where p1 and p2 are the position of two coordinate frames and g_1
    is coordinate frame of p_1 and n is a constant vector.  This is
    equivalent to constraining frame2 to be in the plane with normal n
    that goes through the origin of frame1.

    Three point constraints with linearly independent vectors force
    two frames to have the same position.

    Point constraints can be created with the s-expression:
    (point-constraint frame1 n1 n2 n3 frame2 [name])
    where n1,n2,n3 are the components of the axis and frame1/2 are the
    names of the frames.
    """    
    def __init__(self, system, frame1, axis, frame2, name=None):
        """
        Create a new Point constraint between frame1 and frame2 along
        the axis.  The axis should be a sequence of three numbers and
        is defined with respect to the frame1 coordinate system.
        """
        Constraint.__init__(self, system, name)
        _PointConstraint.__init__(self)

        if not system.get_frame(frame1):
            raise ValueError("Could not find frame %r" % frame1)
        self._frame1 = system.get_frame(frame1)
        
        if not system.get_frame(frame2):
            raise ValueError("Could not find frame %r" % frame2)
        self._frame2 = system.get_frame(frame2)

        self.axis = axis
        
    def __repr__(self):
        return "<Point constraint g1='%s' axis=(%f %f %f) g2='%s'>" % (
               self.frame1.name,
               self.axis[0],
               self.axis[1],
               self.axis[2],
               self.frame2.name)

    def get_frame1(self): return self._frame1
    frame1 = property(get_frame1)

    def get_frame2(self): return self._frame2
    frame2 = property(get_frame2)

    def get_axis(self):
        return np.array([self._axis0, self._axis1, self._axis2])
    def set_axis(self, axis):
        self._axis0 = axis[0]
        self._axis1 = axis[1]
        self._axis2 = axis[2]
    axis = property(get_axis, set_axis)
        
