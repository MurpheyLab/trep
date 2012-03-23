# This is a basic humanoid puppet with a fixed head.  It has no
# inputs.  

import sys
import trep
from trep import tx, ty, tz, rx, ry, rz
import trep.visual as visual

# Set the length of simulation and the time step.
tf = 10.0
dt = 0.01

# Define the puppet's mechanical structure
system = trep.System()
frames = [
    tx('TorsoX'), [ty('TorsoY'), [tz('TorsoZ'), [
        rz('TorsoPsi'), [ry('TorsoTheta'), [rx('TorsoPhi',name='Torso'), [
            tz(-1.5, mass=50),
            tx(-1.011), [tz(0.658, name='Right Torso Hook')],
            tx( 1.011), [tz(0.658, name= 'Left Torso Hook')],
            tz(0.9, name='Head'), [tz(0.5, mass=(10,1,1,1))],
            # Define the left arm
            tx(1.3), [tz(0.4), [
                rz('LShoulderPsi'), [ry('LShoulderTheta'), [rx('LShoulderPhi', name='Left Shoulder'), [
                    tz(-0.95, name='Left Humerus', mass=(5,1,1,1)),
                    tz(-1.9), [
                        rx('LElbowTheta', name='Left Elbow'), [
                            tz(-1, name='Left Radius', mass=(4,1,1,1)),
                            tz(-2.001), [tx(0.14), [ty(-0.173, name='Left Finger')]]]]]]]]],
            # Define the right arm
            tx(-1.3), [tz(0.4), [
                rz('RShoulderPsi'), [ry('RShoulderTheta'), [rx('RShoulderPhi', name='Right Shoulder'), [
                    tz(-0.95, name='Right Humerus', mass=(5,1,1,1)),
                    tz(-1.9), [
                        rx('RElbowTheta', name='Right Elbow'), [
                            tz(-1, name='Right Radius', mass=(4,1,1,1)),
                            tz(-2.001), [tx(-0.14), [ty(-0.173, name='Right Finger')]]]]]]]]],
            # Define the left leg
            tx(0.5), [tz(-3.0), [
                rz('LHipPsi'), [ry('LHipTheta'), [rx('LHipPhi', name='Left Hip'), [
                    tz(-1.5, name='Left Femur', mass=(5,1,1,1)),
                    tz(-2.59), [ty(-0.322, name='Left Knee Hook')],
                    tz(-3.0), [
                        rx('LKneeTheta', name='Left Knee'), [
                            tz(-1.5, name='Left Tibia', mass=(4,1,1,1))]]]]]]],
            # Define the right leg
            tx(-0.5), [tz(-3.0), [
                rz('RHipPsi'), [ry('RHipTheta'), [rx('RHipPhi', name='Right Hip'), [
                    tz(-1.5, name='Right Femur', mass=(5,1,1,1)),
                    tz(-2.59), [ty(-0.322, name='Right Knee Hook')],
                    tz(-3.0), [
                        rx('RKneeTheta', name='right Knee'), [
                            tz(-1.5, name='Right Tibia', mass=(4,1,1,1))]]]]]]],
          ]]]]]],  # End of puppet definition
    
    # Define the coordinate frames for the truck that the puppet is suspended from
    tz(14, name='Frame Plane'), [
        tx( 1, name='Left Torso Spindle'),
        tx(-1, name='Right Torso Spinde'),
        tx( 1), [ty(-1, name='Left Arm Spindle')],
        tx(-1), [ty(-1, name='Right Arm Spindle')],
        tx( 1), [ty(-2, name='Left Leg Spindle')],
        tx(-1), [ty(-2, name='Right Leg Spindle')]]
    ]
system.import_frames(frames)

# Add gravity and damping
trep.potentials.Gravity(system, (0, 0, -9.8))
trep.forces.Damping(system, 0.1)

# Define the constraints that model the strings. In this case, the
# strings are all constant lengths.
trep.constraints.Distance(system, 'Left Torso Hook', 'Left Torso Spindle', 13.4)
trep.constraints.Distance(system, 'Right Torso Hook', 'Right Torso Spinde', 13.4)
trep.constraints.Distance(system, 'Left Finger', 'Left Arm Spindle', 15.4)
trep.constraints.Distance(system, 'Right Finger', 'Right Arm Spindle', 15.5)
trep.constraints.Distance(system, 'Left Knee Hook', 'Left Leg Spindle', 18.6)
trep.constraints.Distance(system, 'Right Knee Hook', 'Right Leg Spindle', 18.6)


# We can set the system's configuration by passing a dictionary of
# mapping the names of configuration variables to their values.  
system.q = {
    'TorsoX' : 1,
    'TorsoY' : 1,
    'LElbowTheta' : -1.57,
    'RElbowTheta' : -1.57,
    'LHipTheta' : -0.314,
    'RHipTheta' :  0.314,
    'LHipPhi' : -0.785,
    'RHipPhi' : -0.785,
    'LKneeTheta' : 0.785,
    'RKneeTheta' : 0.785,
    }
		    
# There is no guarantee that the configuration we just set is
# consistent with the system's constraints.  We can call
# System.satisfy_constraints() to have trep find a configuration that
# is consistent.  This just uses a root-finding method to solve h(q) =
# 0 starting from the current configuration.  There is no guarantee
# that it will always converge and no guarantee the final
# configuration is close to the original one.
#
# An alternative method for a system like the puppet is to measure
# then length between the ends of each string and set the string's
# length to that measured length.
system.satisfy_constraints()


# Now we'll extract the current configuration into a tuple to use as
# initial conditions for a variational integrator.
q0 = system.q

# The puppet also has kinematic configuration variables.  These need
# to be specified as inputs at each time step. We are going to hold
# them constant, so we extract them into a tuple now to pass as our
# input in each simulation step.
qk2 = system.qk

# Create and initialize a variational integrator for the system.
mvi = trep.MidpointVI(system)
mvi.initialize_from_configs(0.0, q0, dt, q0)

# Here is the simulation loop.  We're saving the results in the lists 't' and 'q'.
q = [mvi.q2]
t = [mvi.t2]
while mvi.t1 < tf:
    mvi.step(mvi.t2+dt, (), qk2)
    q.append(mvi.q2)
    t.append(mvi.t2)
    # The puppet can take a while to simulate, so print out the time
    # occasionally to indicate our progress.
    if abs(mvi.t2 - round(mvi.t2)) < dt/2.0:
        print "t =",mvi.t2

# After the simulation is finished, we can visualize the results.  The
# viewer can automatically draw a primitive representation for
# arbitrary systems from the tree structure.  print_instructions() has
# the viewer print out basic usage information to the console.
visual.visualize_3d([ visual.VisualItem3D(system, t, q) ])
