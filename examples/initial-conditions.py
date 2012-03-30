# This script shows different ways of initializing the GMVI
# variational integrator
import sys
from math import sin
import trep
from trep import tx,ty,tz,rx,ry,rz

tf = 10.0
dt = 0.01

# Define a simple 2 link pendulum with a force input on the middle joint
system = trep.System()
frames = [
    tz(-1), # Provided as an angle reference
    ry("theta1"), [
        tz(-2, mass=1), [
            ry("theta2"), [
                tz(-2, mass=1)]]]
    ]
system.import_frames(frames)
trep.potentials.Gravity(system, (0, 0, -9.8))
trep.forces.Damping(system, 1.2)
trep.forces.ConfigForce(system, 'theta2', 'theta2-torque')
mvi = trep.MidpointVI(system)  # Create the variational integrator


# Define a function to run the simulation and return the result
def simulate(mvi, u_func, dt, tf):
    q = [mvi.q2]
    p = [mvi.p2]
    t = [mvi.t2]
    while mvi.t1 < tf:
        mvi.step(mvi.t2+dt, u_func(mvi.t1))
        q.append(mvi.q2)
        p.append(mvi.p2)
        t.append(mvi.t2)
    return t,q,p


# Define a function to visualize results
def visualize(system, t, q):
    viewer = SystemTrajectoryViewer(system, t, q)
    viewer.print_instructions()
    viewer.run()




# The most basic way to integrate the variational integrator is using
# two configurations at the start of the simulation.
t0 = 0.0
t1 = t0 + dt
q0 = (0.005, 0.001)  # IC at t0
q1 = (0.006, 0.002)  # IC at t1
# We initialize the mvi this way using mvi.initialize_from_configs:
mvi.initialize_from_configs(t0, q0, t1, q1)


# This function will define the forcing applied to the middle joint.
def forcing1(t):
    return (4.0*sin(5.0*t), )

# Simulate using the first forcing function
(t1,q1,p1) = simulate(mvi, forcing1, dt, tf)
# Uncomment this line to see the result
# visualize(system, t1, q1)


# We can also initialize the integrator using a configuration/discrete
# momentum pair at one time instead of two configurations at two times.
t0 = 0.0
q0 = (0.005, 0.001)
p0 = (10.1, 10.0) # We'll give it a strong initialize momentum so the
                  # effect is obvious
mvi.initialize_from_state(t0, q0, p0)

# Simulate using the first forcing function
(t2,q2,p2) = simulate(mvi, forcing1, dt, tf)
# Uncomment this line to see the result
# visualize(system, t2, q2)



# Both the initialization functions take time as an argument, so they
# can be used to setup the integrator at anypoint along a trajectory.
# Suppose we want to see what happens if the inputs had changed at a
# point along a trajectory we already know.  We'll alter the inputs of
# the first simulated trajectory starting at the 500th time slice.
# (Note: this could also be done by running the entire simulation
# again with a new forcing function that changed at the desired time.)
i = 499
ta = t1[i]
qa = q1[i]
pa = q1[i]
mvi.initialize_from_state(ta, qa, pa)

def forcing2(t):
    return (0.0, )
# Simulate using the first forcing function
(t3,q3,p3) = simulate(mvi, forcing2, dt, tf)

# Join the two trajectory segments
t3 = t1[:i] + t3
q3 = q1[:i] + q3
p3 = p1[:i] + p3

# Uncomment this line to see the result
# visualize(system, t3, q3)


# We can do the same thing using initialize_from_configs too:
i = 499
ta = t1[i-1]
qa = q1[i-1]
tb = t1[i]
qb = q1[i]
mvi.initialize_from_configs(ta, qa, tb, qb)

def forcing2(t):
    return (0.0, )
# Simulate using the first forcing function
(t4,q4,p4) = simulate(mvi, forcing2, dt, tf)

# Join the two trajectory segments
t4 = t1[:i] + t4
q4 = q1[:i] + q4
p4 = p1[:i] + p4
