# This script simulates a pendulum with a single link driven by
# a torsional spring instead of gravity.

import sys
import trep
from trep import tx,ty,tz,rx,ry,rz
import trep.constraints
import trep.visual as visual

tf = 10.0
dt = 0.01

# Here we define the mechanical system
system = trep.System()
frames = [
    ty(3), # Provided as an angle reference
    rx("theta"), [ty(3, mass=1)]
    ]
system.import_frames(frames)
trep.potentials.ConfigSpring(system, 'theta', k=20, q0=0.6)

# These are the initial conditions for the variational integrator.
q0 = (0,)   # Note the comma, this is how you create a tuple with 
q1 = (0,)   # a single element

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
