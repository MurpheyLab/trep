# Calculate the derivatives of the continuous dynamics of a pendulum.

import sys
import math
import time
import trep
import trep.potentials
import numpy as np


links = 2
        
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


# Create the system
system = make_pendulum(links)

# Now we'll calculate the linearization of the dynamics at the initial
# configuration.  We need to think of the system as a first order
# differential equation instead of a second order system.  Let's
# choose the state x = [q dq] so that dx/dt = [dq/dt ddq/dt] = [dq
# f(q,dq)]
n = len(system.configs)

# The derivative df/dx is:
#  [ d[dq]/d[q]   d[dq]/d[dq]    = [  0              I  
#    d[f]/d[q]    d[f]/d[dq] ]   =   d[f]/d[q]  d[f]/d[dq] ]

dfdx = np.vstack([
    np.hstack([np.zeros((n,n)), np.eye(n)]),
    np.hstack([system.f_dq(), system.f_ddq()])
    ])
    

# And that's all.  The more general case includes force inputs and
# kinematic inputs.  See the puppet-continuous-deriv.py for that.

print "The continuous linearization is: "
print dfdx
