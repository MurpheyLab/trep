import trep
from trep import tx,ty,tz,rx,ry,rz
import time
import trep.visual as visual

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

system = trep.System()
system.import_frames([
    rx('theta1'), [
        tz(2, mass=1, name='pend1')
        ],
    ty(1), [
        rx('theta2'), [
            tz(2, mass=1, name='pend2')
            ]]
    ])

trep.potentials.LinearSpring(system, 'pend1', 'pend2', k=20, x0=1)
trep.forces.LinearDamper(system, 'pend1', 'pend2', c=1)
trep.potentials.Gravity(system, name="Gravity")

system.q = [3,-3]

# Simulate
start = time.clock()
(t, q) = simulate_system(system)
finish = time.clock()

# Display
print "Simulation: dt=%f, tf=%f, runtime=%f s" % (dt, tf, finish-start)
visual.visualize_3d([ visual.VisualItem3D(system, t, q) ])
