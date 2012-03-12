import math
import trep
from trep import Potential
from trep._trep import _ConfigSpringPotential
import numpy as np

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    _opengl = True
except:
    _opengl = False

class ConfigSpring(_ConfigSpringPotential, Potential):
    """
    Spring implements a fixed-length, fixed-spring-constant
    spring against a configuratoin variable:

    V(q) = 0.5*k*(q - q0)**2
    """
    def __init__(self, system, config, k, q0=0.0, name=None):
        Potential.__init__(self, system, name)
        _ConfigSpringPotential.__init__(self)
        self._q0 = 0.0
        self._k = 0.0

        if not system.get_config(config):
            raise ValueError("Could not find config %r" % config)
        self._config = system.get_config(config)
        
        self._q0 = q0
        self._k = k

    @property
    def config(self):
        return self._config

    @property
    def q0(self):
        return self._q0

    @q0.setter
    def q0(self, q0):
        self._q0 = q0

    @property 
    def k(self):
        return self._k

    @k.setter
    def k(self, k):
        self._k = k
            
