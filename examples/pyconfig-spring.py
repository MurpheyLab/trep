import sys
import trep
from trep import Potential
from trep import tx,ty,tz,rx,ry,rz
import trep.constraints
import trep.visual as visual


class PyConfigSpring(Potential):
    """
    Spring implements a fixed-length, fixed-spring-constant
    spring against a configuration variable:

    V(q) = 0.5*k*(q - x0)**2
    """
    def __init__(self, system, config, k, x0=0.0, name=None):
        Potential.__init__(self, system, name)

        if not system.get_config(config):
            raise ValueError("Could not find config %r" % config)
        self.config = system.get_config(config)
        
        self.x0 = float(x0)
        self.k = float(k)

    def V(self):
        return 0.5 * self.k * (self.config.q - self.x0)**2

    def V_dq(self, q1):
        if q1 == self.config:
            return self.k * (self.config.q - self.x0)
        else:
            return 0.0
            
    def V_dqdq(self, q1, q2):
        if q1 == self.config and q2 == self.config:
            return self.k
        else:
            return 0.0

    def V_dqdqdq(self, q1, q2, q3):
        return 0.0

    
tf = 10.0
dt = 0.01

# Here we define the mechanical system
system = trep.System()
frames = [
    ty(3), # Provided as an angle reference
    rx("theta"), [ty(3, mass=1)]
    ]
system.import_frames(frames)
#trep.potentials.Gravity(system, (0, 0, -9.8))
PyConfigSpring(system, 'theta', x0=0.7, k=20)
trep.forces.Damping(system, 1.2)

# These are the initial conditions for the variational integrator.
q0 = [0]
q1 = [0]

# Now we create and initialize a variational integrator for the system.
mvi = trep.MidpointVI(system)
mvi.initialize_from_configs(0.0, q0, dt, q1)

# This is the actual simulation loop.  We will store the results in
# two lists.
q = [mvi.q2]
t = [mvi.t2]
while mvi.t1 < tf:
    mvi.step(mvi.t2+dt)
    q.append(mvi.q2)
    t.append(mvi.t2)

# After the simulation is finished, we can visualize the results.  The
# viewer can automatically draw a primitive representation for
# arbitrary systems from the tree structure.
visual.visualize_3d([ visual.VisualItem3D(system, t, q) ])
