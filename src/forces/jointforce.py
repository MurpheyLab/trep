import trep
from trep import Force
from trep._trep import _JointForce

class JointForce(_JointForce, Force):
    def __init__(self, system, config, finput, name=None):
        Force.__init__(self, system, name)
        _JointForce.__init__(self)

        if not system.get_config(config):
            raise ValueError("Could not find config %r" % config)
        self._config = system.get_config(config)
               
        self._input = self._create_input(finput)

    @property
    def finput(self):
        """The force input controlling the joint force."""
        return self._input

    @property
    def config(self):
        """The configuration variable that this force is applied to."""
        return self._config
    
