import trep
import _trep
from _trep import _Constraint
from itertools import product

class Constraint(_Constraint):
    """
    Constraint is the base class for holonomic constraints in a
    system.  It must be subclassed to implement types of constraints.
    It should not be used directly.

    See the full documentation for details on how to create new
    constraints.
    """
    def __init__(self, system, name=None, tolerance=1e-10):
        """
        Add a constraint to the system.  This should be called
        first thing by a subclass when creating a new constraint.
        """
        _Constraint.__init__(self)
        assert isinstance(system, trep.System)
        self._system = system
        self.name = name
        self.tolerance = tolerance
        system._add_constraint(self)
        system._structure_changed()

    def opengl_draw(self):
        """
        opengl_draw() is called by the trep automatic visualization
        code to draw a graphical representation of the constraint.
        The opengl coordinate frame will be the system's world frame
        when opengl_draw() is called.
        """
        pass

    @property
    def system(self):
        "System that the constraint is a part of."
        return self._system

    @property 
    def index(self):
        "Index of the constraint in system.constraints."
        return self._index

    @property
    def tolerance(self):
        "Tolerance to consider the constraint satisfies (|h(q)| < tolerance)"
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance):
        self._tolerance = tolerance
    
    def h(self):
        "Calculate the constraint function at the current configuration"
        return self._h()
    
    def h_dq(self, q1):
        """
        Calculate the derivative of the constraint function at the
        current configuration relative to configuration variable q1.
        """        
        assert isinstance(q1, _trep._Config)
        return self._h_dq(q1)
    
    def h_dqdq(self, q1, q2):
        """
        Calculate the second derivative of the constraint function at
        the current configuration relative to configuration varibles
        q1 and q2.
        """
        assert isinstance(q1, _trep._Config)
        assert isinstance(q2, _trep._Config)
        return self._h_dqdq(q1, q2)

    def h_dqdqdq(self, q1, q2, q3):
        """
        Calculate the third derivative of the constraint function at
        the current configuration relative to configuration varibles
        q1, q2, and q3.
        """
        assert isinstance(q1, _trep._Config)
        assert isinstance(q2, _trep._Config)
        assert isinstance(q3, _trep._Config)
        return self._h_dqdqdq(q1, q2, q3)
    
    def h_dqdqdqdq(self, q1, q2, q3, q4):
        """
        Calculate the fourth derivative of the constraint function at
        the current configuration relative to configuration varibles
        q1, q2, q3, and q4.
        """
        assert isinstance(q1, _trep._Config)
        assert isinstance(q2, _trep._Config)
        assert isinstance(q3, _trep._Config)
        assert isinstance(q4, _trep._Config)
        return self._h_dqdqdqdq(q1, q2, q3, q4)
    
    ############################################################################
    # Functions to test derivatives 

    def validate_h_dq(self, delta=1e-6, tolerance=1e-6, verbose=False):
        """Check h_dq() against approximate numeric derivative from h()"""
        return self.system.test_derivative_dq(self.h,
                                              self.h_dq,
                                              delta, tolerance,
                                              verbose=verbose,
                                              test_name='%s.h_dq()' % self.__class__.__name__)
    
    def validate_h_dqdq(self, delta=1e-6, tolerance=1e-6, verbose=False):
        """Check h_dqdq() against approximate numeric derivative from h_dq()"""
        def test(q1):
            return self.system.test_derivative_dq(
                lambda : self.h_dq(q1),
                lambda q2: self.h_dqdq(q1, q2),
                delta, tolerance,
                verbose=verbose,
                test_name='%s.h_dqdq()' % self.__class__.__name__)

        result = [test(q1) for q1 in self.system.configs]
        return all(result)

    def validate_h_dqdqdq(self, delta=1e-6, tolerance=1e-6, verbose=False):
        """Check h_dqdqdq() against approximate numeric derivative from h_dqdq()"""
        def test(q1, q2):
            return self._system.test_derivative_dq(
                lambda : self.h_dqdq(q1, q2),
                lambda q3: self.h_dqdqdq(q1, q2, q3),
                delta, tolerance,
                verbose=verbose,
                test_name='%s.h_dqdqdq()' % self.__class__.__name__)

        result = [test(q1,q2) for q1,q2 in product(self._system.configs, repeat=2)]
        return all(result)
        
    def validate_h_dqdqdqdq(self, delta=1e-6, tolerance=1e-6, verbose=False):
        """Check h_dqdqdqdq() against approximate numeric derivative from h_dqdqdq()"""
        def test(q1, q2, q3):
            return self._system.test_derivative_dq(
                lambda : self.h_dqdqdq(q1, q2, q3),
                lambda q4: self.h_dqdqdqdq(q1, q2, q3, q4),
                delta, tolerance,
                verbose=verbose,
                test_name='%s.h_dqdqdqdq()' % self.__class__.__name__)

        result = [test(q1,q2,q3) for q1,q2,q3 in product(self._system.configs, repeat=3)]
        return all(result)
        
