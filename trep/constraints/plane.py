import trep
from trep import Constraint
from trep._trep import _PointOnPlaneConstraint
import numpy as np

class PointOnPlane(_PointOnPlaneConstraint, Constraint):
    def __init__(self, system, plane_frame, plane_normal, point_frame, name=None):
        Constraint.__init__(self, system, name)
        _PointOnPlaneConstraint.__init__(self)

        self._plane_frame = system.get_frame(plane_frame)
        self._point_frame = system.get_frame(point_frame)

        self.normal = plane_normal
        
    def __repr__(self):
        return "<PointOnPlane plane_frame='%s' normal=(%f %f %f) point_frame='%s'>" % (
               self.plane_frame.name,
               self.normal[0],
               self.normal[1],
               self.normal[2],
               self.point_frame.name)

    @property
    def plane_frame(self):
        return self._plane_frame

    @property
    def point_frame(self):
        return self._point_frame

    @property
    def normal(self):
        return np.array([self._normal0, self._normal1, self._normal2])

    @normal.setter
    def normal(self, normal):
        self._normal0 = normal[0]
        self._normal1 = normal[1]
        self._normal2 = normal[2]
        
