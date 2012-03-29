# The planar closed-chain device seen in Nakamura and Yamane, 2000

# Note: The Y-axis translations do not affect the system dynamics.
#   They have been added to put the device components in front and
#   behind each other for drawing purposes.

# First define the open-chain coordinate frames of the system.  Here
# we have named each frame to correspond to a paper figure (Johnson
# and Murphey).  In general, however, we only need to name frames
# that we reference later in constraints.

import sys
import trep
from trep import tx, ty, tz, rx, ry, rz
import trep.visual as visual

tf = 10.0
dt = 0.01


# Define the mechanical system
system = trep.System()
frames = [
    rx('J', name='J'), [
        tz(-0.5, name='I', mass=1),
        tz(-1), [
            rx('H', name='H'), [
                tz(-1, name='G', mass=1),
                tz(-2, name='O2')]]],
    ty(1.5), [
        rx('K', name='K'), [
            tz(-1, name='L', mass=1),
            tz(-2), [
                rx('M', name='M'), [
                    tz(-0.5, name='N', mass=1),
                    tz(-1.0, name='O')]]]],
    ty(-1.5), [
        rx('A', name='A'), [
            tz(-1, name='B', mass=1),
            tz(-2), [
                rx('C', name='C'), [
                    tz(-0.375, name='D', mass=1),
                    tz(-0.75), [
                        rx('E', name='E'), [
                            tz(-0.5, name='F', mass=1),
                            tz(-1.0, name='G2')
                            ]
                        ]
                    ]
                ]
            ]
        ]
    ]

# Add the frames to the system.
system.import_frames(frames)

# Add gravity and damping
trep.potentials.Gravity(system, (0, 0, -9.8))
trep.forces.Damping(system, 0.1)

# Close the open chains with constraints.
trep.constraints.PointOnPlane(system, 'O', (0,1,0), 'O2')
trep.constraints.PointOnPlane(system, 'O', (0,0,1), 'O2')
trep.constraints.PointOnPlane(system, 'G', (0,1,0), 'G2')
trep.constraints.PointOnPlane(system, 'G', (0,0,1), 'G2')


# We can set the system's configuration by passing a dictionary of
# mapping the names of configuration variables to their values.  
system.q = {
    'K' : 0.523599,
    'M' : -1.34537,
    'J' : -0.523599,
    'H' : 1.21009,
    'A' : -0.523599,
    'C' : 1.5385,
    'E' : 1.22497
    }


# There is no guarantee that the configuration we just set is
# consistent with the system's constraints.  We can call
# System.satisfy_constraints() to have trep find a configuration that
# is consistent.  This just uses a root-finding method to solve h(q) =
# 0 starting from the current configuration.  There is no guarantee
# that it will always converge and no guarantee the final
# configuration is close to the original one.
system.satisfy_constraints()

# Now we'll extract the current configuration into a tuple to use as
# initial conditions for a variational integrator.
q0 = system.q

# Create and initialize the variational integrator
mvi = trep.MidpointVI(system)
mvi.initialize_from_configs(0.0, q0, dt, q0)

# This is our simulation loop.  We save the results in two lists.
q = [mvi.q2]
t = [mvi.t2]
while mvi.t1 < tf:
    mvi.step(mvi.t2+dt)
    q.append(mvi.q2)
    t.append(mvi.t2)

# After the simulation is finished, we can visualize the results.  The
# viewer can automatically draw a primitive representation for
# arbitrary systems from the tree structure.  print_instructions() has
# the viewer print out basic usage information to the console.
visual.visualize_3d([visual.VisualItem3D(system, t, q)])
