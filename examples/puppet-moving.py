# This is a non-interactive simulation of a humanoid puppet.  This
# simulation has a customized visualization and moves the inputs for
# the puppets.

import sys
import trep
from trep.visual import stlmodel
from trep import tx, ty, tz, rx, ry, rz
from math import sin, cos, pi as mpi

# Define the length of the simulation and the time step.
tf = 10.0
dt = 0.025

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
        tx( 1.44), [ty(-2, name='Left Arm Spindle')],
        tx(-1.44), [ty(-2, name='Right Arm Spindle')],
        tx( 0.5), [ty(-2.6, name='Left Leg Spindle')],
        tx(-0.5), [ty(-2.6, name='Right Leg Spindle')]]
    ]
system.import_frames(frames)

trep.potentials.Gravity(system, (0, 0, -9.8))
trep.forces.Damping(system, 0.1)
# Define the strings.  By passing a string instead of a number as the
# distance, a new kinematic configuration variable with that name is
# used to control the string's length.
trep.constraints.Distance(system, 'Left Torso Hook', 'Left Torso Spindle', 13.4)
trep.constraints.Distance(system, 'Right Torso Hook', 'Right Torso Spinde', 13.4)
trep.constraints.Distance(system, 'Left Finger', 'Left Arm Spindle', 'Left Arm String')
trep.constraints.Distance(system, 'Right Finger', 'Right Arm Spindle', 'Right Arm String')
trep.constraints.Distance(system, 'Left Knee Hook', 'Left Leg Spindle', 'Left Leg String')
trep.constraints.Distance(system, 'Right Knee Hook', 'Right Leg Spindle', 'Right Leg String')

# We can set the puppet's configuration with dictionary that maps
# configuration variable names to the values we want to assign.
# Alternatively, if we knew the ordering of the configuration
# variables, we could set q as a list of numbers.
system.q = {
    'LElbowTheta' : -1.57,
    'RElbowTheta' : -1.57,
    'LHipPhi' : -mpi/2,
    'RHipPhi' : -mpi/2,
    'LKneeTheta' : mpi/2,
    'RKneeTheta' : mpi/2,
    'Left Arm String' : 15.4,
    'Right Arm String' : 15.4,
    'Left Leg String' : 16.6,
    'Right Leg String' : 16.6
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
# to be specified as inputs at each time step. We extract their
# initial value into a tuple now so we can modify them during the simulation.
qk2_0 = system.qk

# Create and initialize a variational integrator for the puppet.  We
# use the same configuration (q0) as the two initial configurations of
# the puppet.  This is equivalent to simulating the puppet with no
# initial velocity.  Using two different configurations would result
# in a non-zero initial velocity.
mvi = trep.MidpointVI(system)
mvi.initialize_from_configs(0.0, q0, dt, q0)

# This is the simulation loop.  We save the progress of the simulation
# in the lists q and t.
q = [mvi.q2]
t = [mvi.t2]
while mvi.t1 < tf:
    # For our kinematic inputs, we first copy our starting position.
    qk2 = list(qk2_0) 
    # Now we want to move some of the strings up and down.
    qk2[system.get_config('Left Leg String').k_index] += -0.75*sin(0.6*mpi*mvi.t1)
    qk2[system.get_config('Right Leg String').k_index] += 0.75*sin(0.6*mpi*mvi.t1)
    qk2[system.get_config('Left Arm String').k_index] += 0.5*sin(0.6*mpi*mvi.t1)
    qk2[system.get_config('Right Arm String').k_index] += -0.5*sin(0.6*mpi*mvi.t1)

    # Right now trep stupidly requires that qk2 is a tuple.  This will be changed eventually.
    mvi.step(mvi.t2+dt, (), tuple(qk2))
    q.append(mvi.q2)
    t.append(mvi.t2)
    if abs(mvi.t2 - round(mvi.t2)) < dt/2.0:
        print "t =",mvi.t2


# The simulation is finished and now we are going to animate the
# results but with a more customized viewer.  We can start from the
# usual SystemTrajectoryViewer. 
viewer = trep.visual.SystemTrajectoryViewer(system, t, q)

# The first thing we do is turn off the auto-drawing.
viewer.auto_draw = False
# We can add display functions to frames in the system.  When the
# viewer is drawing the system, it will shift the opengl coordinate
# frame to the configuration of each frame and call it's display
# function.  In this case, we are loading an stlmodel for parts of the
# system and passing the model's draw function as a display function.
viewer.add_display_func('Torso',          stlmodel('./puppet-stl/torso.stl').draw)
viewer.add_display_func('Left Shoulder',  stlmodel('./puppet-stl/lefthumerus.stl').draw)
viewer.add_display_func('Right Shoulder', stlmodel('./puppet-stl/righthumerus.stl').draw)
viewer.add_display_func('Left Elbow',     stlmodel('./puppet-stl/leftradius.stl').draw)
viewer.add_display_func('Right Elbow',    stlmodel('./puppet-stl/rightradius.stl').draw)
viewer.add_display_func('Right Hip',      stlmodel('./puppet-stl/femur.stl').draw)
viewer.add_display_func('Left Hip',       stlmodel('./puppet-stl/femur.stl').draw)
viewer.add_display_func('right Knee',     stlmodel('./puppet-stl/tibia.stl').draw)
viewer.add_display_func('Left Knee',      stlmodel('./puppet-stl/tibia.stl').draw)
viewer.add_display_func('Head',           stlmodel('./puppet-stl/head.stl').draw)
viewer.color = [0.5, 0.5, 0.55]
viewer.background_color = (1.0, 1.0, 1.0, 1.0)

# Since we turned off auto-draw, the constraints are not being
# visualized.  So we can add a separate display function that simply
# calls each string's automatic visualization function.
def draw_constraints():
    for x in system.constraints:
        x.opengl_draw()
# By attaching this function to no frame, it will be called when the
# OpenGL coordinate frame is at the world frame.
viewer.add_display_func(None, draw_constraints)

# We can set the initial position and angle if we don't like the
# default values.  These can be found by trial and error or by adding
# code somewhere (like in the draw_constraints function) to print out
# the viewer's angle and position while you move the camera around. 
viewer.camera.camera_pos = [1.97, -15.728, 2.944]
viewer.camera.camera_ang = [-80.39, -18.36, 0.0]

viewer.run()
