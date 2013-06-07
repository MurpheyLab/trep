# optimalSwitchingTime.py

# Import necessary python modules
import math
from math import pi
import numpy as np
from numpy import dot
import trep
import trep.discopt
from trep import tx, ty, tz, rx, ry, rz
import pylab

# Build a pendulum system
m = 1.0 # Mass of pendulum
l = 1.0 # Length of pendulum
q0 = 0. # Initial configuration of pendulum
t0 = 0.0 # Initial time
tf = 10.0 # Final time
dt = 0.01 # Sampling time
qBar = pi # Desired configuration
Qi = np.eye(2) # Cost weights for states
Ri = np.eye(1) # Cost weights for inputs
uMax = 5.0 # Max absolute input value
tau = 5.0 # Switching time

system = trep.System() # Initialize system

frames = [
    rx('theta', name="pendulumShoulder"), [
        tz(-l, name="pendulumArm", mass=m)]]
system.import_frames(frames) # Add frames

# Add forces to the system
trep.potentials.Gravity(system, (0, 0, -9.8)) # Add gravity
trep.forces.Damping(system, 1.0) # Add damping
trep.forces.ConfigForce(system, 'theta', 'theta-torque') # Add input

# Create and initialize the variational integrator
mvi = trep.MidpointVI(system)
mvi.initialize_from_configs(t0, np.array([q0]), t0+dt, np.array([q0]))

# Create discrete system
TVec = np.arange(t0, tf+dt, dt) # Initialize discrete time vector
dsys = trep.discopt.DSystem(mvi, TVec) # Initialize discrete system
xBar = dsys.build_state(qBar) # Create desired state configuration

# Design linear feedback controller
Qd = np.zeros((len(TVec), dsys.system.nQ)) # Initialize desired configuration trajectory
thetaIndex = dsys.system.get_config('theta').index # Find index of theta config variable
for i,t in enumerate(TVec):
    Qd[i, thetaIndex] = qBar # Set desired configuration trajectory
    (Xd, Ud) = dsys.build_trajectory(Qd) # Set desired state and input trajectory

Qk = lambda k: Qi # Create lambda function for state cost weights
Rk = lambda k: Ri # Create lambda function for input cost weights
KVec = dsys.calc_feedback_controller(Xd, Ud, Qk, Rk) # Solve for linear feedback controller gain
KStabilize = KVec[0] # Use only use first value to approximate infinite-horizon optimal controller gain

# Design energy shaping swing-up controller
system.get_config('theta').q = qBar
eBar = system.total_energy()
KEnergy = 1

# Create cost
cost = trep.discopt.DCost(Xd, Ud, Qi, Ri)

# Helper functions
def wrapTo2Pi(ang):
    return ang % (2*pi)

def wrapToPi(ang):
    return (ang + pi) % (2*pi) - pi

def simulateForward(tau, dsys, q0, xBar):
    mvi = dsys.varint
    tf = dsys.time[-1]

    # Reset discrete system state
    dsys.set(np.array([q0, 0.]), np.array([0.]), 0)

    # Initial conditions
    k = dsys.k
    t = dsys.time[k]
    q = mvi.q1
    x = dsys.xk
    fdx = dsys.fdx()
    xTilde = np.array([wrapToPi(x[0] - xBar[0]), x[1]])
    e = system.total_energy() # Get current energy of the system
    eTilde = e - eBar # Get difference between desired energy and current energy

    # Initial list variables
    K = [k] # List to hold discrete update count
    T = [t] # List to hold time values
    Q = [q] # List to hold configuration values
    X = [x] # List to hold state values
    Fdx = [fdx] # List to hold partial to x
    E = [e] # List to hold energy values
    U = [] # List to hold input values
    L = [] # List to hold cost values

    # Simulate the system forward
    while mvi.t1 < tf-dt:
        if mvi.t1 < tau:
            if x[1] == 0:
                u = np.array([uMax]) # Kick if necessary
            else:
                u = np.array([-x[1]*KEnergy*eTilde]) # Swing-up controller
        else:
            u = -dot(KStabilize, xTilde) # Stabilizing controller
        u = min(np.array([uMax]), max(np.array([-uMax]), u)) # Constrain input

        dsys.step(u) # Step the system forward by one time step

        # Update variables
        k = dsys.k
        t = TVec[k]
        q = mvi.q1
        x = dsys.xk
        fdx = dsys.fdx()
        xTilde = np.array([wrapToPi(x[0] - xBar[0]), x[1]])
        e = system.total_energy()
        eTilde = e - eBar
        l = cost.l(np.array([wrapTo2Pi(x[0]), x[1]]), u, k)

        # Append to lists
        K.append(k)
        T.append(t)
        Q.append(q)
        X.append(x)
        Fdx.append(fdx)
        E.append(e)
        U.append(u)
        L.append(l)

    J = np.sum(L)
    return (K, T, Q, X, Fdx, E, U, L, J)

# Optimize
cnt = 0
Tau = []
JVec = []
JdTau = float('Inf')
while cnt < 10 and abs(JdTau) > .001:
    cnt = cnt + 1

    # Simulate forward from 0 to tf
    (K, T, Q, X, Fdx, E, U, L, J) = simulateForward(tau, dsys, q0, xBar)

    # Simulate backward from tf to tau
    k = K[-1]
    lam = np.array([[0],[0]])
    while T[k] > tau + dt/2:
       lamDx = np.array([cost.l_dx(X[k], U[k-1], k)])
       f2Dx = Fdx[k]
       lamDt = -lamDx.T - dot(f2Dx.T,lam)
       lam = lam - lamDt*dt
       k = k - 1

    # Calculate dynamics of swing-up controller at switching time
    x = X[k]
    xTilde = np.array([wrapToPi(x[0] - xBar[0]), x[1]])
    u1 = -dot(KStabilize, xTilde)
    u1 = min(np.array([uMax]), max(np.array([-uMax]), u1))
    dsys.set(X[k], u1, k)
    f1 = dsys.f()

    # Calculate dynamics of stabilizing controller at switching time
    eTilde = E[k] - eBar
    u2 = np.array([-x[1]*KEnergy*eTilde])
    u2 = min(np.array([uMax]), max(np.array([-uMax]), u2))
    dsys.set(X[k], u2, k)
    f2 = dsys.f()

    # Calculate value  of change in cost to change in switch time
    JdTau = dot(f1-f2, lam)

    # Armijo - used to determine step size
    chi = 0
    alpha = .5
    beta = .1
    tauTemp = tau - (alpha**chi)*JdTau
    (KTemp, TTemp, QTemp, XTemp, FdxTemp, ETemp, UTemp, LTemp, JTemp) = simulateForward(tauTemp, dsys, q0, xBar)
    while JTemp > J + (alpha**chi)*beta*(JdTau**2):
        tauTemp = tau - (alpha**chi)*JdTau
        (KTemp, TTemp, QTemp, XTemp, FdxTemp, ETemp, UTemp, LTemp, JTemp) = simulateForward(tauTemp, dsys, q0, xBar)
        chi = chi + 1
    gamma = alpha**chi # Step size

    # Calculate new switching time
    tauPlus = tau - gamma*JdTau

    # Print iteration results
    print "Optimization iteration: %d" % cnt
    print "Current switch time: %.2f" % tau
    print "New switch time: %.2f" % tauPlus
    print "Current cost: %.2f" % J
    print "Parital of cost to switch time: %.2f" % JdTau
    print ""

    # Update to  new switching time
    tau = tauPlus

print "Optimal switch time: %.2f" % tau

# Simulate with optimal switching time
(K, T, Q, X, Fdx, E, U, L, J) = simulateForward(tau, dsys, q0, xBar)

# Visualize the system in action
trep.visual.visualize_3d([ trep.visual.VisualItem3D(system, T, Q) ])

# Plot results
ax1 = pylab.subplot(311)
pylab.plot(T, X)
pylab.title("Swing-up and stabilize with optimal switching time")
pylab.ylabel("X")
pylab.legend(["qd","p"])
pylab.subplot(312, sharex=ax1)
pylab.plot(T[1:], U)
pylab.xlabel("T")
pylab.ylabel("U")
pylab.subplot(313, sharex=ax1)
pylab.plot(T[1:], L)
pylab.xlabel("T")
pylab.ylabel("L")
pylab.show()
