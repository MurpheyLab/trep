import numpy as np
import trep
import trep.discopt as discopt

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

# calc derivatives of discrete Lagrangian:
q = system.get_config("theta_1")
print "D1D1Ld = ",\
    dt/4*system.L_dqdq(q,q) - \
    1/2.*system.L_ddqdq(q,q) - \
    1/2*system.L_ddqdq(q,q) + \
    1/dt*system.L_ddqddq(q,q)
print "D2D1Ld = ",\
  dt/4*system.L_dqdq(q,q) + \
  1/2.*system.L_ddqdq(q,q) - \
  1/2*system.L_ddqdq(q,q) - \
  1/dt*system.L_ddqddq(q,q)

# calc derivatives of qk+1
print "dqk+1/dqk = ",mvi.q2_dq1()[0][0]
print "dqk+1/dpk = ",mvi.q2_dp1()[0][0]
print "dqk+1/duk = ",mvi.q2_du1()[0][0]

# calc derivatives of pk+1
print "dpk+1/dqk = ",mvi.p2_dq1()[0][0]
print "dpk+1/dpk = ",mvi.p2_dp1()[0][0]
print "dpk+1/duk = ",mvi.p2_du1()[0][0]

# calculate A and B using discopt module:
dsys = discopt.DSystem(mvi, np.array([0,dt]))
dsys.set(np.array([qk,pk]),np.array([uk]),0)
A = dsys.fdx()
B = dsys.fdu()
C = np.hstack((B, np.dot(A,B)))
print "A = \n",A
print "B = \n",B
print "C = \n",C
print "rank(C) = ",np.rank(C)

# calculate the second order linearization:
print "\n"
print "=============================="
print "Second Order Linearizations"
print "=============================="
# calc second derivatives of qk+1
# diagonal terms:
print "dqk+1/dqkdqk = ",mvi.q2_dq1dq1().squeeze()
print "dqk+1/dpkdpk = ",mvi.q2_dp1dp1().squeeze()
print "dqk+1/dukduk = ",mvi.q2_du1du1().squeeze()
# off diagonal terms
print "dqk+1/dqkdpk = ",mvi.q2_dq1dp1().squeeze()
print "dqk+1/dqkduk = ",mvi.q2_dq1du1().squeeze()
print "dqk+1/dpkduk = ",mvi.q2_dp1du1().squeeze()
# build full second-order linearization of qk+1
print "\delta^2q_{k+1} = \n",np.vstack((
    np.hstack((dsys.fdxdx([1,0,0]), dsys.fdxdu([1,0,0]))),
    np.hstack((dsys.fdxdu([1,0,0]).T, dsys.fdudu([1,0,0])))
    ))

# calc second derivatives of pk+1
# diagonal terms:
print "dpk+1/dqkdqk = ",mvi.p2_dq1dq1().squeeze()
print "dpk+1/dpkdpk = ",mvi.p2_dp1dp1().squeeze()
print "dpk+1/dukduk = ",mvi.p2_du1du1().squeeze()
# off diagonal terms
print "dpk+1/dqkdpk = ",mvi.p2_dq1dp1().squeeze()
print "dpk+1/dqkduk = ",mvi.p2_dq1du1().squeeze()
print "dpk+1/dpkduk = ",mvi.p2_dp1du1().squeeze()
# build full second-order linearization of qk+1
print "\delta^2p_{k+1} = \n",np.vstack((
    np.hstack((dsys.fdxdx([0,1,0]), dsys.fdxdu([0,1,0]))),
    np.hstack((dsys.fdxdu([0,1,0]).T, dsys.fdudu([0,1,0])))
    ))
