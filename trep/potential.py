import trep
import _trep
from _trep import _Potential
import numpy as np
from itertools import product

class Potential(_Potential):
    """
    Potential is the base class for potential energies in a system.
    It must be subclassed to implement types of potential energies.
    It should not be used directly.

    See the full documentation for details on how to create new
    potential energies.
    """
    def __init__(self, system, name=None):
        """
        Add a potential to the system.  This should be called before
        the subclass starts touching the system.
        """
        _Potential.__init__(self)
        assert isinstance(system, trep.System)
        self._system = system
        self.name = name
        system._add_potential(self)
        system._structure_changed()

    def opengl_draw(self):
        """
        opengl_draw() is called by the automatic visualization code to
        draw a graphical representation of the potential.  The OpenGL
        coordinate frame will be the system's world frame when
        opengl_draw() is called.
        """
        pass

    @property
    def system(self):
        "System that the Potential belongs to."
        return self._system

    def V(self):
        "Calculate the potential function at the current configuration."
        return self._V()

    def V_dq(self, q1):
        """
        Calculate the derivative of the potential function with
        respect to the value of configuration variable q1.
        """
        assert isinstance(q1, _trep._Config)
        return self._V_dq(q1)

    def V_dqdq(self, q1, q2):
        """
        Calculate the derivative of the potential function with
        respect to the value of configuration variables q1 and q2.
        """
        assert isinstance(q1, _trep._Config)
        assert isinstance(q2, _trep._Config)
        return self._V_dqdq(q1, q2)

    def V_dqdqdq(self, q1, q2, q3):
        """
        Calculate the derivative of the potential function with
        respect to the value of configuration variables q1, q2, and
        q3.
        """
        assert isinstance(q1, _trep._Config)
        assert isinstance(q2, _trep._Config)
        assert isinstance(q3, _trep._Config)
        return self._V_dqdqdq(q1, q2, q3)


    ############################################################################
    # Functions to test derivatives 

    def validate_V_dq(self, delta=1e-6, tolerance=1e-6, verbose=False):
        """Check V_dq() against approximate numeric derivative from V()"""
        return self.system.test_derivative_dq(self.V,
                                              self.V_dq,
                                              delta, tolerance,
                                              verbose=verbose,
                                              test_name='%s.V_dq()' % self.__class__.__name__)
    
    def validate_V_dqdq(self, delta=1e-6, tolerance=1e-6, verbose=False):
        """Check V_dqdq() against approximate numeric derivative from V_dq()"""
        def test(q1):
            return self.system.test_derivative_dq(
                lambda : self.V_dq(q1),
                lambda q2: self.V_dqdq(q1, q2),
                delta, tolerance,
                verbose=verbose,
                test_name='%s.V_dqdq()' % self.__class__.__name__)

        result = [test(q1) for q1 in self.system.configs]
        return all(result)

    def validate_V_dqdqdq(self, delta=1e-6, tolerance=1e-6, verbose=False):
        """Check V_dqdqdq() against approximate numeric derivative from V_dqdq()"""
        def test(q1, q2):
            return self._system.test_derivative_dq(
                lambda : self.V_dqdq(q1, q2),
                lambda q3: self.V_dqdqdq(q1, q2, q3),
                delta, tolerance,
                verbose=verbose,
                test_name='%s.V_dqdqdq()' % self.__class__.__name__)

        result = [test(q1,q2) for q1,q2 in product(self._system.configs, repeat=2)]
        return all(result)
