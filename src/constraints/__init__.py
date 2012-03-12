"""
This package contains the built-in constraint types.

trep includes some common holonomic constraints.  New constraints are
created by creating instances of the constraint type.  The initializer
argument varies between constraint types, so see the documentation for
the specific constraint you want to create.

The constraints included are:

Distance - A distance constraint maintains a specific distance between two
  coordinate frames.  The distance can be a constant or a kinematic
  configuration variable.

Point - A point constraint keeps the distance between two frames at
  zero along some axis.  Multiple point constraints are combined to
  make pin or spherical joints.

New constraint types are created by subclassing trep.Constraint.  See
the documentation trep.Constraint for more information.
"""

from distance import Distance
from point import Point


