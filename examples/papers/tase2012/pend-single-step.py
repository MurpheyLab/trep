import numpy as np
import trep
import scipy.optimize as so

# set mass, length, and gravity:
m = 1.0; l = 1.0; g = 9.8;

# set state and step conditions:
pk = 0.5 # discrete generalized momentum
qk = 0.2 # theta config
uk = 0.8 # input torque
dt = 0.1 # timestep

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

# create and initialize variational integrator
mvi = trep.MidpointVI(system)
mvi.initialize_from_state(0, np.array([qk]), np.array([pk]))

# take single step with VI:
mvi.step(mvi.t1+dt, np.array([uk])) # args are t2, u1

# compare with manual computation results:
def DEL1(qkp1):
    return pk - (qkp1-qk)/dt - g*dt/2.*np.sin((qkp1+qk)/
                                              2.0) + uk*dt
# Implicitly solve DEL1 to get new config
qkp1 = so.newton(DEL1, qk)
# get new momentum
pkp1 = (qkp1-qk)/dt - g*dt/2.0*np.sin((qkp1+qk)/2.0)

# print results
print "=============================================="
print "trep VI results:\tanalytical results:"
print "=============================================="
print "qk+1 = ",mvi.q2[0],"\t","qk+1 = ",qkp1
print "pk+1 = ",mvi.p2[0],"\t","pk+1 = ",pkp1
print "=============================================="
