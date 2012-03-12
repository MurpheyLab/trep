import trep
from trep import Potential
from trep._trep import _GravityPotential
import numpy as np

class Gravity(_GravityPotential, Potential):
    """
    """    
    def __init__(self, system, gravity=(0.0, 0.0, -9.8), name=None):
        Potential.__init__(self, system, name)
        _GravityPotential.__init__(self)
        
        # Set gravity
        self.gravity = gravity

    def __repr__(self):
        return "<Gravity %f %f %f>" % tuple(self.gravity)

    @property
    def gravity(self):
        """Gravity vector."""
        return np.array([self._gravity0, self._gravity1, self._gravity2])
    
    @gravity.setter
    def gravity(self, g):
        self._gravity0 = g[0]
        self._gravity1 = g[1]
        self._gravity2 = g[2]

