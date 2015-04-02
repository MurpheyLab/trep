# Simulation of a scissor lift.  The number of links in the lift can
# be changed.

import sys
import trep
from trep import Frame
import trep.visual as visual
import math
from math import pi as mpi
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *


# Define the simulation parameters
segments = 4      # Number of segments in the scissor lift.
m_link = 1.0      # Mass of each link in the lift.
I_link = 1.0      # Rotational inertia of each link in the lift.
L_link = 5.0      # Length of each link in the lift.
m_slider = 1.0    # Mass of the top slider of the lift.
theta_0 = 0.05*mpi  # Initial angle of the lift.
tf = 10.0         # Duration of the simulation
dt = 0.01         # Time step of the simulation


def simulate_system(system):
    """
    Simulates the system from its current configuration with no
    initial velocity for the duration and time step specified by global variables.
    """
    # Extract the current configuration into a tuple to use as the two
    # initial configurations of the variational integrator.
    q0 = system.q

    # Create and initialize the variational integrator.
    mvi = trep.MidpointVI(system)
    mvi.initialize_from_configs(0.0, q0, dt, q0)

    # Run the simulation and save the results into the lists 't' and 'q'.
    q = [mvi.q1]
    t = [mvi.t1]
    while mvi.t1 < tf:
        mvi.step(mvi.t2+dt)
        q.append(mvi.q1)
        t.append(mvi.t1) 
        # Print out progress during the simulation.
        if abs(mvi.t1 - round(mvi.t1)) < dt/2.0:
            print "t =",mvi.t1

    return (t, q)


def make_scissor_lift():
    """ Create a scissor lift according to the global simulation parameters. """
    def add_level(left, right, link=0):
        """
        Recursive function to build the scissor lift by attaching
        another set of links.  'left' and 'right' are the coordinate
        frames to attach the left and right links to.
        """
        if link == segments:  # Terminate the recusions
            return (left, right)

        # Create the base of the left link
        left = Frame(left, trep.RY, "L%02d" % link, "L%02d" % link)
        if link == 0:
            left.config.q = theta_0
        else:
            left.config.q = mpi + 2.0*theta_0

        # Create a frame at the middle of the link to attach the link's mass.
        left_mid = Frame(left, trep.TX, L_link/2.0)
        left_mid.set_mass(m_link, I_link, I_link, I_link)
        # Add the end of the link.
        left_end = Frame(left, trep.TX, L_link)

        right = Frame(right, trep.RY, "R%02d" % link, "R%02d" % link)
        if link == 0:                                      
            right.config.q = mpi - theta_0
        else:
            right.config.q = mpi-2.0 * theta_0
        right_mid = Frame(right, trep.TX, L_link/2.0)
        right_mid.set_mass(m_link, I_link, I_link, I_link)
        right_end = Frame(right, trep.TX, L_link)

        # Join the two links at the middle.
        trep.constraints.PointToPoint2D(system, 'xz', left_mid, right_mid)

        # Add a new level.  Note that left and right switch each time
        return add_level(right_end, left_end, link+1)

    # Create the new system
    system = trep.System()
    trep.potentials.Gravity(system, name="Gravity")
    # Add the top slider
    slider = Frame(system.world_frame, trep.TX, "SLIDER")
    slider.config.q = L_link*math.cos(theta_0)
    slider.set_mass(m_slider)
    # Create all the links in the system.
    add_level(system.world_frame, slider)
    # The scissor lift should satisfy the constraints, but call
    # satisfy_constraints() in case it needs to be nudged a little
    # from numerical error.
    system.satisfy_constraints()
    return system


# Create the scissor lift
system = make_scissor_lift()
# Simulate the system
(t, q) = simulate_system(system)


# Create a visualization of the scissor lift.
visual.visualize_2d([visual.VisualItem2D(system, t, q, plane='XZ')])

