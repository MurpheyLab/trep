
# Replace "Template" with your potential name

import math
import trep
from trep import Potential
from trep._trep import _TemplatePotential
import numpy as np

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    _opengl = True
except:
    _opengl = False

class Template(_TemplatePotential, Potential):
    """
    Document your potential here
    """
    def __init__(self, system, parameter=0.0, name=None):
        Potential.__init__(self, system, name)
        _TemplatePotential.__init__(self)
        self._parameter = 0.0 # Immediately initialize your Python
                               # parameters with constants so there
                               # are no unassigned variables if an
                               # exception is raised.

        # After the internals are initialized, fill in the correct
        # values
        self._parameter = parameter

    def get_parameter(self):
        "Document each parameter"
        return self._parameter
    parameter = property(get_parameter)

    if _opengl:
        # You can add OpenGL code for the auto drawing visualizations
        # to draw your potential
        def opengl_draw(self):
            glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT | GL_LINE_BIT)
            glColor3f(1.0, 0.0, 0.0)
            glDisable(GL_LIGHTING)
            glBegin(GL_LINES)
            #glVertex3f(frame1[0][3], frame1[1][3], frame1[2][3])
            #glVertex3f(frame2[0][3], frame2[1][3], frame2[2][3])    
            glEnd()
            glPopAttrib()
