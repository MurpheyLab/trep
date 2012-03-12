import trep
from _trep import _Input

class Input(_Input):
    """
    The Input class represents inputs for forces in a mechanical
    system.  An input has a current value accessible through the value
    attribute.

    Instances of Input should only be created using
    Force._create_input()!
    
    Inputs are automatically created by Forces as needed, so you not
    need to create them unless you are creating a new type of Force.
    """
    def __init__(self, system, name=None):
        """
        Create a input in a system.
        """
        _Input.__init__(self)
        assert isinstance(system, trep.System)
        
        self._system = system
        self.name = name
        system._add_input(self)            
        self.system._structure_changed()
        
    def __repr__(self):
        return "<Input %r %f>" % (self.name or id(self), self.u)

    @property
    def u(self):
        "Current value of the input variable."
        return self._u
    @u.setter
    def u(self, v):
        self._u = v
        # Different input changes system dynamics.
        self.system._clear_cache() 

    @property 
    def system(self):
        "System that the input variable is a part of. (read-only)"
        return self._system

    @property
    def force(self):
        "Force object that uses the input. (read-only)"
        return self._force

    @property
    def index(self):
        "Index of the input in the system.inputs list. (read-only)"
        return self._index


