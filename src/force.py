import trep
import _trep
from _trep import _Force
import numpy as np

class Force(_Force):
    """
    Force is the base class for forces in a system.  It must be
    subclassed to implement types of forces.  It should not be used
    directly.

    See the full documentation for details on how to create new
    forces.
    """
    def __init__(self, system, name=None):
        """
        Add a force to the system.  This should be called first thing
        by a subclass when creating a new force.
        """
        _Force.__init__(self)
        assert isinstance(system, trep.System)
        self._system = system
        self.name = name
        system._add_force(self)
        system._structure_changed()

    def _create_input(self, name=None):
        new_input = trep.Input(self.system, name)
        new_input._force = self
        return new_input

    def opengl_draw(self):
        """
        opengl_draw() is called by the trep automatic visualization
        code to draw a graphical representation of the force.
        The opengl coordinate frame will be the system's world frame
        when opengl_draw() is called.
        """
        pass

    @property
    def system(self):
        "System that the force is a part of."
        return self._system

    def f(self, q):
        """
        Calculate the current force on the configuration variable q.
        """
        assert isinstance(q, _trep._Config)
        return self._f(q)

    def f_dq(self, q, q1):
        """       
        Calculate the current derivative of the force on the
        configuration variable q with respect to the value of the
        configuration variable q1.
        """
        assert isinstance(q,  _trep._Config)
        assert isinstance(q1, _trep._Config)
        return self._f_dq(q, q1)
    
    def f_ddq(self, q, dq1):
        """
        Calculate the current derivative of the force on the
        configuration variable q with respect to the velocity of the
        configuration variable q1.
        """
        assert isinstance(q,   _trep._Config)        
        assert isinstance(dq1, _trep._Config)        
        return self._f_ddq(q, dq1)
    
    def f_du(self, q, u1):
        """
        Calculate the current derivative of the force on the
        configuration variable q with respect to the input u1.
        """
        assert isinstance(q,  _trep._Config)
        assert isinstance(u1, _trep._Input)
        return self._f_du(q, u1)

    def f_dqdq(self, q, q1, q2):
        """
        Calculate the current second derivative of the force on the
        configuration variable q with respect to the values of
        configuration variables q1 and q2.
        """
        assert isinstance(q,  _trep._Config)        
        assert isinstance(q1, _trep._Config)        
        assert isinstance(q2, _trep._Config)        
        return self._f_dqdq(q, q1, q2)

    def f_ddqdq(self, q, dq1, q2):
        """
        Calculate the current second derivative of the force on the
        configuration variable q with respect to the velocity of q1
        and value of q2.
        """        
        assert isinstance(q,   _trep._Config)        
        assert isinstance(dq1, _trep._Config)        
        assert isinstance(q2,  _trep._Config)        
        return self._f_ddqdq(q, dq1, q2)
    
    def f_ddqddq(self, q, dq1, dq2):
        """
        Calculate the current second derivative of the force on the
        configuration variable q with respect to the velocities of q1
        and q2.
        """        
        assert isinstance(q,   _trep._Config)        
        assert isinstance(dq1, _trep._Config)        
        assert isinstance(dq2, _trep._Config)        
        return self._f_ddqddq(q, dq1, dq2)

    def f_dudq(self, q, u1, q2):
        """
        Calculate the current second derivative of the force on the
        configuration variable q with respect to the input u1 and
        value of q2.
        """        
        assert isinstance(q,  _trep._Config)        
        assert isinstance(u1, _trep._Input)        
        assert isinstance(q2, _trep._Config)        
        return self._f_dudq(q, u1, q2)

    def f_duddq(self, q, u1, dq2):
        """
        Calculate the current second derivative of the force on the
        configuration variable q with respect to the input u1 and
        velocity of q2.
        """        
        assert isinstance(q,   _trep._Config)        
        assert isinstance(u1,  _trep._Input)        
        assert isinstance(dq2, _trep._Config)        
        return self._f_duddq(q, u1, dq2)

    def f_dudu(self, q, u1, u2):
        """
        Calculate the current second derivative of the force on the
        configuration variable q with respect to the inputs u1 and u2.
        """        
        assert isinstance(q,  _trep._Config)        
        assert isinstance(u1, _trep._Input)        
        assert isinstance(u2, _trep._Input)        
        return self._f_dudu(q, u1, u2)

    def validate_f_dq(self, delta=1e-6, tolerance=1e-6, verbose=False):
        """Check f_dq() against approximate numeric derivative from f()"""
        def f():
            return np.array([self.f(q) for q in self.system.configs])
        def f_dq(q1):
            return np.array([self.f_dq(q,q1) for q in self.system.configs])
        
        return self.system.test_derivative_dq(f, f_dq,
                                              delta, tolerance,
                                              verbose=verbose,
                                              test_name='%s.f_dq()' % self.__class__.__name__)

