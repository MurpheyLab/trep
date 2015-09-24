import trep
from trep import Force
from trep._trep import _SpatialWrenchForce
import numpy as np

try:
    import trep.visual
    from OpenGL.GL import *
    _opengl = True
except:
    _opengl = False


class SpatialWrench(_SpatialWrenchForce, Force):
    def __init__(self, system, frame, wrench=tuple(), name=None):
        Force.__init__(self, system, name)
        _SpatialWrenchForce.__init__(self)
        self._frame = None
        self._wrench_vars = (None,)*6
        self._wrench_cons = (0.0, )*6
        
        if not system.get_frame(frame):
            raise ValueError("Could not find frame %r" % frame)
        self._frame = system.get_frame(frame)

        # Pad wrench to make sure we have a list with 6 entries
        wrench = (list(wrench) + [0.0]*6)[:6]

        wrench_var = [None]*6
        wrench_con = [0.0]*6
        for i in range(6):
            if isinstance(wrench[i], str):
                wrench_var[i] = self._create_input(wrench[i])
            else:
                wrench_con[i] = float(wrench[i])
        self._wrench_vars = wrench_var
        self._wrench_cons = wrench_con

    def _get_wrench_vars(self):
        return (self._wrench_var0,
                self._wrench_var1,
                self._wrench_var2,
                self._wrench_var3,
                self._wrench_var4,
                self._wrench_var5)
    def _set_wrench_vars(self, vars):
        self._wrench_var0 = vars[0]
        self._wrench_var1 = vars[1]
        self._wrench_var2 = vars[2]
        self._wrench_var3 = vars[3]
        self._wrench_var4 = vars[4]
        self._wrench_var5 = vars[5]
    _wrench_vars = property(_get_wrench_vars, _set_wrench_vars)
        
    def _get_wrench_cons(self):
        return (self._wrench_con0,
                self._wrench_con1,
                self._wrench_con2,
                self._wrench_con3,
                self._wrench_con4,
                self._wrench_con5)
    def _set_wrench_cons(self, cons):
        self._wrench_con0 = cons[0]
        self._wrench_con1 = cons[1]
        self._wrench_con2 = cons[2]
        self._wrench_con3 = cons[3]
        self._wrench_con4 = cons[4]
        self._wrench_con5 = cons[5]
    _wrench_cons = property(_get_wrench_cons, _set_wrench_cons)
                    
    def get_wrench(self):
        return [V if V else C for (V,C) in zip(self._wrench_vars, self._wrench_cons)]
    wrench = property(get_wrench)

    def get_wrench_val(self):
        return [V.q if V else C for (V,C) in zip(self._wrench_vars, self._wrench_cons)]
    def set_wrench_val(self, wrench):
        for i,v in enumerate(wrench[:6]):
            if self._wrench_vars[i]:
                self._wrench_vars[i].q = v
            else:
                self._wrench_cons[i] = v
    wrench_val = property(get_wrench_val, set_wrench_val)

    def get_frame(self): return self._frame
    frame = property(get_frame)

    if _opengl:
        def opengl_draw(self):
            glPushMatrix()        
            mat = np.zeros((4,4))
            mat[0,0] = 1.0
            mat[1,1] = 1.0
            mat[2,2] = 1.0
            mat[3,3] = 1.0
            mat[0,3] = self.frame.g()[0,3]
            mat[1,3] = self.frame.g()[1,3]
            mat[2,3] = self.frame.g()[2,3]
            glMultMatrixf(mat.flatten('F'))
            glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT | GL_LINE_BIT)
            glColor3f(1.0, 0.0, 0.0)
            glDisable(GL_LIGHTING)
            glBegin(GL_LINES)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(self.wrench[0], self.wrench[1], self.wrench[2])
            glEnd()
            glPopAttrib()
            glPopMatrix()


