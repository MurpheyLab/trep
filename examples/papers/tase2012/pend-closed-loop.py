import numpy as np
import sys
import trep
import trep.discopt as discopt

# set mass, length, and gravity:
m = 1.0; l = 1.0; g = 9.8;

# initial state and timesteps:
p0 = 1 # discrete generalized momentum
q0 = 1. # theta config
x0 = np.array([q0, p0]) # initial state
dt = 0.1 # timestep
tf = 10 # final time
tvec = np.arange(0,tf+dt,dt)

# create system
system = trep.System()
# define frames
frames = [
        trep.rz("theta_1", name="PendAngle"), [
            trep.ty(-l, name="PendMass", mass=m)]]
# add frames to system
system.import_frames(frames)
# add gravity potential
trep.potentials.Gravity(system, (0,-g,0))
# add a torque at the base
trep.forces.ConfigForce(system, "theta_1", "tau")

# create vi and dsys
mvi = trep.MidpointVI(system)
dsys = discopt.DSystem(mvi, tvec)

# load in the reference and optimal trajectory (pend-optimization must be run
# first):
Xd, Ud = dsys.load_state_trajectory("pend_ref.mat")
Xopt,Uopt = dsys.load_state_trajectory("pend_opt.mat")

# now we can get a stabilizing controller about the optimal trajectory
Q = lambda t: np.diag([1,1])
R = lambda t: np.diag([1])
K = dsys.calc_feedback_controller(Xopt, Uopt, Q=Q, R=R)

# set initial state
dsys.set(x0, np.array([0]), 0)

# create arrays for storing results
X = np.zeros(Xopt.shape)
U = np.zeros(Uopt.shape)

# simulate closed-loop system
X[0] = dsys.xk
for k in range(len(X)-1):
    U[k] = Uopt[k] + np.dot(K[k], Xopt[k] - X[k])
    if k == 0:
        dsys.set(X[k], U[k], k, xk_hint=Xopt[k+1])
    else:
        dsys.step(U[k], xk_hint=Xopt[k+1])
    X[k+1] = dsys.f()


