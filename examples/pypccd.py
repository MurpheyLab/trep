# This example shows how to create new constraints in Python.  It
# implements the same planar closed-chain device from the pccd.py
# example.  Compare the speed of a constraint implemented here and the
# built in equivalent.  Add-on constraints can also be written in C to
# get the same performance, but for working out bugs, I recommend
# working in Python first.  This constraint could also be optimized to
# improve performance.

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
import numpy as np

tf = 10.0
dt = 0.01


class PyPointOnPlane(trep.Constraint):
    def __init__(self, system, frame1, axis, frame2, name=None):
        trep.Constraint.__init__(self, system, name)
        self.frame1 = system.get_frame(frame1)
        self.frame2 = system.get_frame(frame2)
        self.axis = np.array(axis[:3] + (0,))

    def h(self):
        """ Calculate the constraint function h(q) """
        return np.dot(np.dot(self.frame1.g(),self.axis),(self.frame1.p() - self.frame2.p()))

    def h_dq(self, q_i):
        """ Calculate the derivative of h(q) with respect to q_i """
        h = 0.0
        h += np.dot(np.dot(self.frame1.g_dq(q_i),self.axis),(self.frame1.p() - self.frame2.p()))
        h += np.dot(np.dot(self.frame1.g(),self.axis),(self.frame1.p_dq(q_i) - self.frame2.p_dq(q_i)))
        return h
    
    def h_dqdq(self, q_i, q_j):
        """ Calculate the 2nd derivative of h(q) with respect to q_i and q_j """
        v = self.frame1.p() - self.frame2.p()
        v_dq_j = self.frame1.p_dq(q_j) - self.frame2.p_dq(q_j)
        v_dq_i = self.frame1.p_dq(q_i) - self.frame2.p_dq(q_i)
        v_dqdq = self.frame1.p_dqdq(q_i, q_j) - self.frame2.p_dqdq(q_i, q_j)
        
        h = 0.0
        h += np.dot(np.dot(self.frame1.g_dqdq(q_i, q_j)*self.axis),v)
        h += np.dot(np.dot(self.frame1.g_dq(q_i),self.axis),v_dq_j)
    
        h += np.dot(np.dot(self.frame1.g_dq(q_j),self.axis),v_dq_i)
        h += np.dot(np.dot(self.frame1.g(),self.axis), v_dqdq)
        return h
            


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
PyPointOnPlane(system, 'O', (0,1,0), 'O2')
PyPointOnPlane(system, 'O', (0,0,1), 'O2')
PyPointOnPlane(system, 'G', (0,1,0), 'G2')
PyPointOnPlane(system, 'G', (0,0,1), 'G2')

# We can set the system's configuration with a dictionary that maps
# the configuration variable names to the values we want to assign.
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
    # The Python constraints are much slower, so print out the time
    # occasionally to indicate our progress.
    if abs(mvi.t2 - round(mvi.t2)) < dt/2.0:
        print "t =",mvi.t2

# After the simulation is finished, we can visualize the results.  The
# viewer can automatically draw a primitive representation for
# arbitrary systems from the tree structure.  print_instructions() has
# the viewer print out basic usage information to the console.
visual.visualize_3d([visual.VisualItem3D(system, t, q)])






