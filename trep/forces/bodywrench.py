import trep
from trep import Force
from trep._trep import _BodyWrenchForce

try:
    import trep.visual
    from OpenGL.GL import *
    _opengl = True
except:
    _opengl = False


class BodyWrench(_BodyWrenchForce, Force):
    def __init__(self, system, frame, wrench=tuple(), name=None):
        Force.__init__(self, system, name)
        _BodyWrenchForce.__init__(self)
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

    @property
    def _wrench_vars(self):
        return (self._wrench_var0,
                self._wrench_var1,
                self._wrench_var2,
                self._wrench_var3,
                self._wrench_var4,
                self._wrench_var5)

    @_wrench_vars.setter
    def _wrench_vars(self, vars):
        self._wrench_var0 = vars[0]
        self._wrench_var1 = vars[1]
        self._wrench_var2 = vars[2]
        self._wrench_var3 = vars[3]
        self._wrench_var4 = vars[4]
        self._wrench_var5 = vars[5]

    @property
    def _wrench_cons(self):
        return (self._wrench_con0,
                self._wrench_con1,
                self._wrench_con2,
                self._wrench_con3,
                self._wrench_con4,
                self._wrench_con5)

    @_wrench_cons.setter
    def _wrench_cons(self, cons):
        self._wrench_con0 = cons[0]
        self._wrench_con1 = cons[1]
        self._wrench_con2 = cons[2]
        self._wrench_con3 = cons[3]
        self._wrench_con4 = cons[4]
        self._wrench_con5 = cons[5]

    @property
    def wrench(self):
        return [V if V else C for (V,C) in zip(self._wrench_vars, self._wrench_cons)]

    @property
    def wrench_val(self):
        return [V.q if V else C for (V,C) in zip(self._wrench_vars, self._wrench_cons)]

    @wrench_val.setter
    def wrench_val(self, wrench):
        for i,v in enumerate(wrench[:6]):
            if self._wrench_vars[i]:
                self._wrench_vars[i].q = v
            else:
                self._wrench_cons[i] = v

    @property 
    def frame(self):
        return self._frame

    if _opengl:
        def opengl_draw(self):
            glPushMatrix()
            glMultMatrixf(self.frame.g().flatten('F'))
            glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT | GL_LINE_BIT)
            glColor3f(1.0, 0.0, 0.0)
            glDisable(GL_LIGHTING)
            glBegin(GL_LINES)
            glVertex3f(0.0, 0.0, 0.0)
            glVertex3f(float(self.wrench[0]), float(self.wrench[1]), float(self.wrench[2]))
            glEnd()
            glPopAttrib()
            glPopMatrix()
