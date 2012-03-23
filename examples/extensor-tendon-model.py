# This is a simplified model of an two-dimensional extensor tendon
# using a network of linear springs.  Three muscles pull on branches
# of the tendon with constant force and direction and the tendon
# settles into a steady state position.

import sys
import trep
from trep import tx, ty, tz, rx, ry, rz
import trep.visual as visual


tf = 10.0
dt = 0.01

# Define the mechanical structure of the tendon.
system = trep.System()
frames = [
    rx('theta-1'), [
        tz('x-1', name='A', mass=1), [
            rx('theta-2'), [
                tz('x-2', name='B', mass=1), [
                    rx('theta-3'), [
                        tz('x-3', name='C', mass=1)]]],
            rx('theta-4'), [
                tz('x-4', name='D', mass=1), [
                    rx('theta-5'), [
                        tz('x-5', name='E', mass=1)]]]]],
            
    rx('theta-6'), [
        tz('x-6', name='F', mass=1), [
            rx('theta-8'), [
                tz('x-8', name='H', mass=1), [
                    rx('theta-9'), [
                        tz('x-9', name='I', mass=1)]]]]]               
    ]
# Add frames to the system
system.import_frames(frames)
# Add springs between the coordinate frames and add some damping
trep.potentials.LinearSpring(system, "World", "A", 10.0, 1.0)
trep.potentials.LinearSpring(system, "A", "B", 10.0, 1.0)
trep.potentials.LinearSpring(system, "B", "C", 10.0, 1.0)
trep.potentials.LinearSpring(system, "A", "D", 10.0, 1.0)
trep.potentials.LinearSpring(system, "D", "E", 10.0, 1.0)
trep.potentials.LinearSpring(system, "World", "F", 10.0, 1.0)
trep.potentials.LinearSpring(system, "F", "D", 10.0, 1.0)
trep.potentials.LinearSpring(system, "F", "H", 10.0, 1.0)
trep.potentials.LinearSpring(system, "H", "I", 10.0, 1.0)
trep.forces.Damping(system, 4.0)
# Define forces for the system.  A HybridWrench applies force to the
# center of a coordinate frame but is specified in world coordinates,
# as opposed to a SpatialWrench that applies a force that is defined
# in world coordinates but applied at the point in the frame that
# intersects the world origin.
trep.forces.HybridWrench(system, 'C', (0, 1, -1))
trep.forces.HybridWrench(system, 'E', (0, 0, -1.414))
trep.forces.HybridWrench(system, 'I', (0, -1, -1))


# We can set the current configuration of a system by passing a
# dictionary of mapping the names of configuration variables to their
# values.
system.q = {
    "theta-1" : 0.5,
    "theta-2" : 0.0,
    "theta-3" : -0.5,
    "theta-4" : -1.0,
    "theta-5" : 0.5,
    "theta-6" : -0.5,
    "theta-8" : 0.0,
    "theta-9" : 0.5,
    "x-1" : -1.0,
    "x-2" : -1.0,
    "x-3" : -1.0,
    "x-4" : -1.0,
    "x-5" : -1.0,
    "x-6" : -1.0,
    "x-8" : -1.0,
    "x-9" : -1.0,
}

# Now we'll extract the current configuration into a tuple to use as
# initial conditions for a variational integrator.
q0 = system.q

# Create and initialize a variational integrator for the tendon.
mvi = trep.MidpointVI(system)
mvi.initialize_from_configs(0.0, q0, dt, q0)

# Here is the simulation loop.  We're saving the results in the lists 't' and 'q'.
q = [mvi.q2]
t = [mvi.t2]
while mvi.t1 < tf:
    mvi.step(mvi.t2+dt)
    q.append(mvi.q2)
    t.append(mvi.t2)
    # Print out the progress during the simulation.
    if abs(mvi.t2 - round(mvi.t2)) < dt/2.0:
        print "t =",mvi.t2


visual.visualize_3d([ visual.VisualItem3D(system, t, q) ])
