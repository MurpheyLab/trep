import numpy as np
import trep
import trep.discopt as discopt

# set mass, length, and gravity:
m = 1.0; l = 1.0; g = 9.8;

# initial state and timesteps:
p0 = 0.0 # discrete generalized momentum
q0 = 0.0 # theta config
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
# set initial state
dsys.set(x0, np.array([0]), 0)

# generate initial trajectory
Qtraj = np.array(len(tvec)*[system.q])
Utraj = np.array((len(tvec)-1)*[[0]])
(X0,U0) = dsys.build_trajectory(Q=Qtraj,rho=Utraj)

# generate reference trajectory
Qref = np.array(len(tvec)*[[np.pi]])
Qref[0:len(Qref)/2] = 0
Uref = np.array((len(tvec)-1)*[[0]])
(Xd, Ud) = dsys.build_trajectory(Q=Qref, u=Uref)

# define cost weights, and setup cost function
Qcost = np.diag([1000,1])
Rcost = np.diag([50])
cost = discopt.DCost(Xd, Ud, Qcost, Rcost)

# define an optimization object:
optimizer = discopt.DOptimizer(dsys, cost)
optimizer.optimize_ic = False
optimizer.descent_tolerance = 1e-6

# setup initial steps of optimization:
optimizer.first_method = "quasi"
optimizer.first_method_iterations = 4


# perform second order optimization:
print "Second order optimization"
finished = False
costs2 = []
dcosts2 = []
dcosts2_linear = []
method = "newton"
step_count = optimizer.first_method_iterations
Xn = X0.copy()
Un = U0.copy()
finished, Xn, Un = optimizer.optimize(X0, U0, max_steps=optimizer.first_method_iterations)
costs2.append(optimizer.calc_cost(Xn,Un))
while not finished:
    Kproj,dX,dU,Q,R,S = optimizer.calc_descent_direction(Xn, Un, method="steepest")
    dcosts2_linear.append(optimizer.calc_dcost(Xn,Un,dX,dU))
    finished,Xn,Un,dcost,ncost = optimizer.step(
        step_count, Xn, Un, method=method)
    step_count += 1
    costs2.append(ncost)
    dcosts2.append(dcost)
print "Optimization Finished Bool: ",finished,"\r\n"


# perform first order optimization:
print "First order optimization"
finished = False
costs1 = []
dcosts1 = []
dcosts1_linear = []
method = "steepest"
step_count = optimizer.first_method_iterations
Xg = X0.copy()
Ug = U0.copy()
finished, Xg, Ug = optimizer.optimize(X0, U0, max_steps=optimizer.first_method_iterations)
costs1.append(optimizer.calc_cost(Xg,Ug))
while not finished:
    Kproj,dX,dU,Q,R,S = optimizer.calc_descent_direction(Xg, Ug, method="steepest")
    dcosts1_linear.append(optimizer.calc_dcost(Xg,Ug,dX,dU))
    finished,Xg,Ug,dcost,ncost = optimizer.step(
        step_count, Xg, Ug, method=method)
    step_count += 1
    costs1.append(ncost)
    dcosts1.append(dcost)
print "Optimization Finished Bool: ",finished,"\r\n"

# store the results:
dsys.save_state_trajectory("pend_ref.mat", Xd, Ud)
dsys.save_state_trajectory("pend_opt.mat", Xn, Un)
