# This is a non-interactive simulation of a humanoid puppet using
# continuous dynamics.  This simulation has a customized visualization
# and moves the inputs for the puppets.  This simulation illustrates
# how much more verbose continuous simulations are in trep.  This is
# mostly because I don't have any built in clases to represent the
# second order dynamics as a first order system.  Eventually something
# like that will be added to trep and these types of simulations will get simpler.

import sys
import trep
from trep.visual import stlmodel
from trep import tx, ty, tz, rx, ry, rz
from math import sin, cos, pi as mpi, exp
import numpy as np
from numpy import matrix, array
import scipy
from scipy.integrate import odeint

# Define the length of the simulation and the time step.
tf = 10.0
dt = 0.01



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

# We can set the puppet's configuration by passing a dictionary of
# mapping the names of configuration variables to their values.
# Alternatively, if we knew the ordering of the configuration
# variables, we could just pass a list of numbers.
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

# Now we write a function to simulate the dynamics and call it to run
# the simulation.

# We need to represent the mechanical system as a first-order
# differential equation rather than a second order differential
# equation to use standard numeric integrators.  For this example, we
# can let  x = [q dq] for our state and mu = [u ddqk] for the input.        
def simulate_system(system, tf, dt):
    n = len(system.configs)
    nk = len(system.kin_configs)

    def calc_ddqk(t):
        """
        This function is responsible for calculating the kinematic
        inputs at every time step.
        """
        ddqk = [0.0]*len(system.kin_configs)
        # Since the discrete simulation only works with velocities, we
        # were able to get away with this simulation without worrying
        # about the intial velocity.  The continuous case is much less
        # forgiving, so to give a similiar trajectory with q(0) =
        # dq(0) = 0, we add a (1-exp(-t)) term.
        
        # Want q = (1-exp(-t))*sin(f*mpi*t)
        # -> ddq =  2*exp(-t)*f*pi*cos(f*pi*t) - exp(-t)*sin(f*pi*t) - (1-exp(-t))*f*f*pi*pi*sin(f*pi*t)
        f = 0.6
        ddq = 2*exp(-t)*f*mpi*cos(f*mpi*t) - exp(-t)*sin(f*mpi*t) - (1-exp(-t))*f*f*mpi*mpi*sin(f*mpi*t)
        ddqk[system.get_config('Left Leg String').k_index] = 0.75*ddq
        ddqk[system.get_config('Right Leg String').k_index] = -0.75*ddq
        ddqk[system.get_config('Left Arm String').k_index] = -0.5*ddq
        ddqk[system.get_config('Right Arm String').k_index] = 0.5*ddq
        return ddqk
    
    def f(x, t):
        """
        This is the wrapper function that is called by the numeric integrator.
        """
        # Split up the current state into a configuration and its velocity.
        q = x.tolist()[:n]
        dq = x.tolist()[n:]
        # Set the current configuration and set the current kinematic inputs.
        system.set_state(q=q, dq=dq, ddqk=calc_ddqk(t), t=t)
        # We call System.f() to have trep calculate the acceleration
        # forom the current state.  The results are written to the ddq
        # field of each configuration variable.
        system.f()  
        ddq = system.ddq
        # Since our state is [q dq], its time derivative is [dq ddq]
        return np.concatenate((dq, ddq))

    # Extract the current state into a tuple.
    x0 = np.concatenate((system.q, system.dq))

    # Generate a list of time values we want to calculate the configuration for.
    t = [dt*i for i in range(int(tf/dt))]

    # Run the numerical integrator
    x = odeint(f, x0, t)

    # Extract the configuration out of the resulting trajectory.
    q = [xi.tolist()[:n] for xi in x]
    return (t,q)
# Call our function to simulate the system.
(t,q) = simulate_system(system, tf, dt)

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
