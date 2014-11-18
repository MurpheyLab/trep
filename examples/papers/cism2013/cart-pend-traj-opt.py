from math import pi
import numpy as np
import trep
import trep.visual as visual
import matplotlib.pylab as mp
import trep.discopt as discopt

# set mass, length, and gravity:
m = 1.0; l = 1.0; g = 9.8; mc = 1;

# define initial state
q0 = np.array([0, pi]) # x = [x_cart, theta]
p0 = np.array([0, 0])
X0 = np.hstack([q0,p0])

# define time parameters:
dt = 0.1
tf = 10.

# define reference trajectory
ad = 7*pi/8.
def fref(t):
    if 0<t and t<3:
        return pi
    elif 3<=t and t<7:
        return pi + (ad-pi)*np.sin(pi/4.*(t-3))
    elif t<=10:
        return pi
    else:
        print "Error!"

# create system
system = trep.System()
# define frames
frames = [
    trep.tx("x_cart", name="CartFrame", mass=mc), [
        trep.rz("theta", name="PendulumBase"), [
            trep.ty(-l, name="Pendulum", mass=m)]]]
# add frames to system
system.import_frames(frames)
# add gravity potential
trep.potentials.Gravity(system, (0,-g,0))
# add a horizontal force on the cart
trep.forces.ConfigForce(system, "x_cart", "cart_force")

# create a variational integrator, and a discrete system
t = np.arange(0,tf+dt,dt)
mvi = trep.MidpointVI(system)
dsys = discopt.DSystem(mvi, t)

# create an initial guess and reference trajectory
(Xinit,Uinit) = dsys.build_trajectory([q0.tolist()]*len(t))
qref = [[0, fref(x)] for x in t]
(Xref,Uref) = dsys.build_trajectory(qref)

# create cost functions:
Q = np.diag([100, 50000, 0.1, 0.1])
R = np.diag([1])
cost = discopt.DCost(Xref, Uref, Q, R)

# define an optimizer object
optimizer = discopt.DOptimizer(dsys, cost)

# setup and perform optimization
optimizer.first_method_iterations = 4
finished, X, U = optimizer.optimize(Xinit, Uinit)

mp.hold(True)
l1 = mp.plot(t, Xref[:,0:2],"--",lw=2, color="gray")[0]
l2 = mp.plot(t, X[:,0:2],lw=2,color="black")[0]
mp.hold(False)
mp.legend([l1,l2],["Reference", "Optimal"],
          loc="lower right")
mp.xlabel("time [s]")
mp.show()
mp.clf()
mp.cla()
mp.close()


