# Simulate an arbitrarily long pendulum.

import sys
import math
import time
import trep
import trep.potentials
import trep.visual as visual

links = 5
dt = 0.01
tf = 10.0


def simulate_system(system):
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

    return (t,q)


        
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
        child = trep.Frame(frame, trep.RX, "link-%d" % link, "link-%d" % link)

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

# Simulate
start = time.clock()
(t, q) = simulate_system(system)
finish = time.clock()

# Display
print "%d-link pendulum, dt=%f, tf=%f... runtime=%f s" % (links, dt, tf, finish-start)
visual.visualize_3d([ visual.VisualItem3D(system, t, q) ])
