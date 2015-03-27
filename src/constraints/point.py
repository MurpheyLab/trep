import trep
from trep import Constraint
from trep._trep import _PointToPointConstraint

class PointToPoint3D():
    def __init__(self, system, frame1, frame2, name=None):
        assert frame1 is not None
        assert frame2 is not None

        self.frame1 = system.get_frame(frame1)
        self.frame2 = system.get_frame(frame2)
        
        PointToPoint1D(system, 'x', frame1, frame2, name)
        PointToPoint1D(system, 'y', frame1, frame2, name)
        PointToPoint1D(system, 'z', frame1, frame2, name)

    def get_actual_distance(self):
        """Return the actual distance between frame1 and frame2."""
        p1 = self.frame1.p()
        p2 = self.frame2.p()

        distance = ((p1[0]-p2[0])**2.0 + 
                    (p1[1]-p2[1])**2.0 + 
                    (p1[2]-p2[2])**2.0)**0.5
        return distance


class PointToPoint2D():
    def __init__(self, system, plane, frame1, frame2, name=None):
        assert frame1 is not None
        assert frame2 is not None

        self.frame1 = system.get_frame(frame1)
        self.frame2 = system.get_frame(frame2)
        
        plane_map = {'YZ':0, 'ZY':0, 'XZ':1, 'ZX':1, 'XY':2, 'YX':2, 
                     'yz':0, 'zy':0, 'xz':1, 'zx':1, 'xy':2, 'yx':2}

        if plane_map[plane] == 0:
            PointToPoint1D(system, 'y', frame1, frame2, name)
            PointToPoint1D(system, 'z', frame1, frame2, name)
        elif plane_map[plane] == 1:
            PointToPoint1D(system, 'x', frame1, frame2, name)
            PointToPoint1D(system, 'z', frame1, frame2, name)
        elif plane_map[plane] == 2:
            PointToPoint1D(system, 'x', frame1, frame2, name)
            PointToPoint1D(system, 'y', frame1, frame2, name)

    def get_actual_distance(self):
        """Return the actual distance between frame1 and frame2."""
        p1 = self.frame1.p()
        p2 = self.frame2.p()

        distance = ((p1[0]-p2[0])**2.0 + 
                    (p1[1]-p2[1])**2.0 + 
                    (p1[2]-p2[2])**2.0)**0.5
        return distance


class PointToPoint1D(_PointToPointConstraint, Constraint):
    def __init__(self, system, axis, frame1, frame2, name=None):
        Constraint.__init__(self, system, name)
        _PointToPointConstraint.__init__(self)

        self._frame1 = system.get_frame(frame1)
        self._frame2 = system.get_frame(frame2)
        self.axis = axis

        axis_map = {'x' : 0, 'y' : 1, 'z' : 2, 'X' : 0, 'Y' : 1, 'Z' : 2}
        self._component = axis_map[self.axis]

        
    def __repr__(self):
        return "<PointToPointConstraint %s-axis '%s' '%s'>" % (
            self.axis,
            self.frame1.name,
            self.frame2.name)   

    @property
    def frame1(self):
        return self._frame1

    @property
    def frame2(self):
        return self._frame2

    def get_actual_distance(self):
        """Return the actual distance between frame1 and frame2."""
        p1 = self.frame1.p()
        p2 = self.frame2.p()

        distance = ((p1[0]-p2[0])**2.0 + 
                    (p1[1]-p2[1])**2.0 + 
                    (p1[2]-p2[2])**2.0)**0.5
        return distance

