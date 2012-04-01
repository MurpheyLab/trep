from itertools import product
import trep
from _trep import _TapeMeasure
import numpy as np

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    _opengl = True
except ImportError:
    _opengl = False

class TapeMeasure(_TapeMeasure):
    def __init__(self, system, frames):
        self._system = system
        self._frames = tuple([system.get_frame(f) for f in frames])
        # Update the seg table.
        self._system.add_structure_changed_func(self._update_structure)
        # Trigger a system change so our update gets called.  Don't
        # call it directly in case there is a hold in progres right now.
        self._system._structure_changed()

    @property
    def system(self):
        return self._system

    @property
    def frames(self):
        return self._frames

    def length(self):
        return self._length()

    def length_dq(self, q1):
        assert isinstance(q1, trep.Config)
        return self._length_dq(q1)

    def length_dqdq(self, q1, q2):
        assert isinstance(q1, trep.Config)
        assert isinstance(q2, trep.Config)
        return self._length_dqdq(q1, q2)

    def length_dqdqdq(self, q1, q2, q3):
        assert isinstance(q1, trep.Config)
        assert isinstance(q2, trep.Config)
        assert isinstance(q3, trep.Config)
        return self._length_dqdqdq(q1, q2, q3)

    def velocity(self):
        return self._velocity()

    def velocity_dq(self, q1):
        assert isinstance(q1, trep.Config)
        return self._velocity_dq(q1)

    def velocity_dqdq(self, q1, q2):
        assert isinstance(q1, trep.Config)
        assert isinstance(q2, trep.Config)
        return self._velocity_dqdq(q1, q2)

    def velocity_ddq(self, dq1):
        assert isinstance(dq1, trep.Config)
        return self._velocity_ddq(dq1)

    def velocity_ddqdq(self, dq1, q2):
        assert isinstance(dq1, trep.Config)
        assert isinstance(q2, trep.Config)
        return self._velocity_ddqdq(dq1, q2)

    if _opengl:
        def opengl_draw(self, width=1, color=(1,1,1)):
            glPushAttrib(GL_CURRENT_BIT | GL_LINE_BIT | GL_LIGHTING_BIT)
            glDisable(GL_LIGHTING)
            glColor3f(*color)
            glLineWidth(width)
            glBegin(GL_LINE_STRIP)
            for f in self._frames:
                glVertex3f(*f.p()[:3])
            glEnd()
            glPopAttrib()

    def _update_structure(self):

        ## We are going to create a table that maps configuration
        ## variables with a list of segments where only one frame
        ## depends on that configuration variable.  We could do this
        ## with a simple list of lists, but to avoid writing any more
        ## C than we have to, we'll save it to a 2D numpy array of
        ## integers.  It will be [nQ x num_segments] initialized to
        ## -1.  Each row corresponds to one config.  Then we fill the
        ## columns from left to right with indices of the correct
        ## segments.

        # This has to be one longer than the length of segments so we
        # can have -1 to indicate the end even if all segments depend on q.        
        table = np.ones( (self.system.nQ, len(self._frames)), dtype=np.int32) * -1

        for qi,q in enumerate(self.system.configs):
            segments = []

            for i in range(len(self._frames)-1):
                frame1 = self._frames[i]
                frame2 = self._frames[i+1]

                # Looking for segments where only one frame depends on q
                if not frame1.uses_config(q) and not frame2.uses_config(q):
                    continue
                if frame1.uses_config(q) and frame2.uses_config(q):
                    continue
                segments.append(i)                
            # Save the list to the table
            for i,seg in enumerate(segments):
                table[qi, i] = seg
        # Save it for the C code to use.  Make a copy to ensure
        # correct ordering and type (this bit me earlier when numpy
        # decided to change it between np.ones() and here.
        self._seg_table = np.array(table, dtype=np.int32, order='C')
            
        

    ############################################################################
    # Functions to test derivatives (may move to unit tests)

    def validate_length_dq(self, delta=1e-6, tolerance=1e-6, verbose=False):
        return self._system.test_derivative_dq(self.length,
                                               self.length_dq,
                                               delta, tolerance,
                                               verbose=verbose,
                                               test_name='TapeMeasure.length_dq()' )
    
    def validate_length_dqdq(self, delta=1e-6, tolerance=1e-6, verbose=False):
        def test(q1):
            return self._system.test_derivative_dq(
                lambda : self.length_dq(q1),
                lambda q2: self.length_dqdq(q1, q2),
                delta, tolerance,
                verbose=verbose,
                test_name='TapeMeasure.length_dqdq()')

        result = [test(q1) for q1 in self._system.configs]
        return all(result)

    def validate_length_dqdqdq(self, delta=1e-6, tolerance=1e-6, verbose=False):
        def test(q1, q2):
            return self._system.test_derivative_dq(
                lambda : self.length_dqdq(q1, q2),
                lambda q3: self.length_dqdqdq(q1, q2, q3),
                delta, tolerance,
                verbose=verbose,
                test_name='TapeMeasure.length_dqdqdq()')

        result = [test(q1,q2) for q1,q2 in product(self._system.configs, repeat=2)]
        return all(result)

    def validate_velocity_dq(self, delta=1e-6, tolerance=1e-6, verbose=False):
        return self._system.test_derivative_dq(self.velocity,
                                               self.velocity_dq,
                                               delta, tolerance,
                                               verbose=verbose,
                                               test_name='TapeMeasure.velocity_dq()' )
    
    def validate_velocity_dqdq(self, delta=1e-6, tolerance=1e-6, verbose=False):
        def test(q1):
            return self._system.test_derivative_dq(
                lambda : self.velocity_dq(q1),
                lambda q2: self.velocity_dqdq(q1, q2),
                delta, tolerance,
                verbose=verbose,
                test_name='TapeMeasure.velocity_dqdq()')

        result = [test(q1) for q1 in self._system.configs]
        return all(result)

    def validate_velocity_ddq(self, delta=1e-6, tolerance=1e-6, verbose=False):
        return self._system.test_derivative_ddq(self.velocity,
                                                self.velocity_ddq,
                                                delta, tolerance,
                                                verbose=verbose,
                                                test_name='TapeMeasure.velocity_ddq()' )

    def validate_velocity_ddqdq(self, delta=1e-6, tolerance=1e-6, verbose=False):
        def test(dq1):
            return self._system.test_derivative_dq(
                lambda : self.velocity_ddq(dq1),
                lambda q2: self.velocity_ddqdq(dq1, q2),
                delta, tolerance,
                verbose=verbose,
                test_name='TapeMeasure.velocity_dqdq()')

        result = [test(q1) for q1 in self._system.configs]
        return all(result)
