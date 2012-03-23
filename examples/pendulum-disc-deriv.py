# Calculate the derivative of the discrete dynamics of a pendulum.

import sys
import math
import time
import trep
import trep.potentials
import numpy as np

links = 2
dt = 0.01
        
def make_pendulum(num_links):
    """
    make_pendulum(num_links) -> System
    
    Create a pendulum system with num_links.
    """
    def add_level(frame, link=0):
        """
        Recusively add links to a system by attaching a new link to
        the specified frame.
        """
        if link == num_links:
            return

        # Create a rotation for the pendulum.
        # The first argument is the name of the frame, the second is
        # the transformation type, and the third is the name of the
        # configuration variable that parameterizes the
        # transformation.
        child = trep.Frame(frame, trep.RY, "link-%d" % link, "link-%d" % link)

        # Move down to create the length of the pendulum link.
        child = trep.Frame(child, trep.TZ, -1)
        # Add mass to the end of the link (only a point mass, no
        # rotational inertia)
        child.set_mass(1.0)
        
        add_level(child, link+1)

    # Create a new system, add the pendulum links, and rotate the top
    # pendulum.
    system = trep.System()
    trep.potentials.Gravity(system, name="Gravity")
    add_level(system.world_frame)
    system.get_config("link-0").q = math.pi/4.0
    return system


# Create
system = make_pendulum(links)

# Now we'll extract the current configuration into a tuple to use as
# initial conditions for a variational integrator.
q0 = system.q

# Create and initialize the variational integrator
mvi = trep.MidpointVI(system)
mvi.initialize_from_configs(0.0, q0, dt, q0)

# We have to step forward because we can't calculate dq(k+1)/dq(k)
# until we've calculated q(k+1) first.
mvi.step(dt*2)
    
# Now we'll calculate the linearization of the dynamics at the initial
# configuration.  We need to think of the system as a first order
# discrete instead of a second order system.  Let's choose the state x
# = [q p] where q is the current configuration and p is the
# generalized momentum.  I've found this to be a better form than x =
# [q0, q1] because it lets you trivially penalize large velocities
# when doing optimal control calculation.
#
# The resulting derivatives are:
#  dx(k+1) = [ q(k+1)  p(k+1)]
#  d[x(k+1)]/d[x(k)] = [  d[q(k+1)]/d[q(k)]  d[q(k+1)]/d[p(k)]
#                         d[p(k+1)]/d[q(k)]  d[p(k+1)]/d[p(k)]

dfdx = np.vstack([
    np.hstack([mvi.q2_dq1(), mvi.q2_dp1()]),
    np.hstack([mvi.p2_dq1(), mvi.p2_dp1()])
    ])


print "The discrete linearization is: "
print dfdx

# As with the continuous linearization, the general case gets more
# confusing as you consider kinematic configuraiton variables and
# force inputs.  In the end though, its all just about properly
# organizing the values that trep calculates into matrices. 
