import math
import trep
from trep import Potential
from trep._trep import _NonlinearConfigSpring
import numpy as np

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    _opengl = True
except:
    _opengl = False

class NonlinearConfigSpring(_NonlinearConfigSpring, Potential):
    """
    NonlinearConfigSpring implements a non-linear spring on a
    configuration variable.  The spring force is defined a spline y = f(x).

    dV/dq = -f(m*q - b)

    """
    def __init__(self, system, config, spline, m=1.0, b=0.0, name=None):
        Potential.__init__(self, system, name)
        _NonlinearConfigSpring.__init__(self)

        if not system.get_config(config):
            raise ValueError("Could not find config %r" % config)
        self._config = system.get_config(config)

        assert isinstance(spline, trep.Spline)
        self._spline = spline.copy()
        self._m = m
        self._b = b

    def get_config(self): return self._config
    config = property(get_config)

    def get_spline(self): return self._spline
    spline = property(get_spline)
