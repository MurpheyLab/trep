import trep
from trep import Force
from trep._trep import _DampingForce
import numpy as np

__all__ = ["Damping"]

class Damping(_DampingForce, Force):
    def __init__(self, system, default=0.0, coefficients={}, name=None):
        Force.__init__(self, system, name)
        _DampingForce.__init__(self)

        self._default = float(default)
        self.coefficients = {}
        
        for (config, coeff) in coefficients.iteritems():
            self.coefficients[system.get_config(config)] = float(coeff)

        system.add_structure_changed_func(self._structure_changed)
        self._structure_changed()

    def _structure_changed(self):
        # Build a 1D array with a damping coefficient for each
        # dynamic configuration variable.
        coefficients = np.ones(self.system.nQd, dtype=np.float, order='C')*self._default
        for (config, coeff) in self.coefficients.iteritems():
            coefficients[config.index] = coeff
        self._coefficients = coefficients


    def get_damping_coefficient(self, config):
        config = self.system.get_config(config)
        if config == None:
            raise ValueError("Couldn't find config")
        if config in self.coefficients:
            return self.coefficients[config]
        else:
            return self._default

    def set_damping_coefficient(self, config, coeff):        
        if coeff is None:
            self.coefficients.pop(self.system.get_config(config), None)
        else:
            self.coefficients[self.system.get_config(config)] = float(coeff)
        self._structure_changed()

    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, value):
        self._default = float(value)
        self._structure_changed()
