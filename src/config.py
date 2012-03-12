import trep
from _trep import _Config

class Config(_Config):
    """
    An instance of Config represents a single generalized coordinate
    in a mechanical system.  It can parameterize the rigid body
    transformation between coordinate frames or be part of a
    constraint definition.

    The variable can be kinematic or dynamic (Dynamic configuration
    variables are 'normal' configuration variables.  Kinematic
    configuration variables are considered perfectly controlled.
    Their trajectories are directly specified by the user instead of
    being determined by dynamics.
    """
    def __init__(self, system, name=None, kinematic=False):
        """
        Create a new dynamic or kinematic configuration variable.

        The variable will be added to the system.
        """
        _Config.__init__(self)
        self._masses = tuple()

        assert isinstance(system, trep.System)        
        self._system = system
        self.name = name
        self._kinematic = kinematic
        if self._kinematic:
            system._add_kin_config(self)            
        else:
            system._add_dyn_config(self)
        self.system._structure_changed()

    @property
    def kinematic(self):
        "Boolean value indicating if this configuration variable is kinematic."
        return self._kinematic
                
    def __repr__(self):
        return "<Config %r %f %f %f>" % (self.name or id(self), self.q, self.dq, self.ddq)

    @property 
    def frame(self):
        """
        The coordinate frame that directly uses this configuration
        variable, or None. (read-only)
        """
        for f in self.system.frames:
            if f.config == self:
                return f
        else:
            return None

    @property 
    def system(self):
        """
        The system that the configuration variable is a part
        of. (read-only)
        """
        return self._system

    @property
    def q(self):
        """Current value of the configuration variable."""
        return self._q    
    @q.setter
    def q(self, value):
        self._q = value
        self.system._clear_cache()

    @property
    def dq(self):
        """
        First time derivative (velocity) of the configuration
        variable.
        """
        return self._dq
    @dq.setter
    def dq(self, value):
        self._dq = value
        self.system._clear_cache()

    @property
    def ddq(self):
        """
        Second time derivative (acceleration) of the configuration
        variable.
        """
        return self._ddq

    @ddq.setter
    def ddq(self, value):
        self._ddq = value
        self.system._clear_cache()  # Only needs to clear dynamics stuff

    @property
    def index(self):
        """
        Index of the configuration variable in
        self.system.configs. (read-only)
        """
        return self._index

    @property
    def k_index(self):
        """
        Index of the configuration variable in
        self.system.kin_configs, or -1 for dynamic configuration
        variables.  (read-only)
        """
        return self._k_index

    @property
    def masses(self):
        """
        Tuple of all frames with non-zero masses that depend (directly
        or indirectly) on this configuration variable.
        """
        return self._masses

