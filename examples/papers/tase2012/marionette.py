import sys
import math
from math import pi, sin
import numpy as np
import trep
import trep.puppets as puppets
import trep.discopt as discopt
import trep.discopt.dlqr as dlqr
import trep.visual


    
def create_offset_initial_condition(dsys, x0, u0):    
    max_err = 0
    for k in range(dsys.kf()):
        if k == 0:
            dsys.set(x0, u0, 0)
        else:
            dsys.step(u0)

        x1 = dsys.f()

        err = np.linalg.norm(x1 - x0)
        err_percent = err/np.linalg.norm(x0)

        if abs(err_percent) > 0.44:
            break

    err = np.linalg.norm(x1 - x0)
    err_percent = err/np.linalg.norm(x0)
    return x1


def simulate(dsys, x0, u_in, K=None, x_in=None):
    x_sim = [x0]
    u_sim = []

    for k in range(dsys.kf()):
        u = u_in[k]
        if K:
            u -= np.dot(K[k], (x_sim[k] - x_in[k]))
        u_sim.append(u)

        if k == 0:
            dsys.set(x_sim[0], u_sim[-1], 0)
        else:
            dsys.step(u)
        x_sim.append(dsys.f())
        
    return np.array(x_sim), np.array(u_sim)
    

def create_initial_trajectory():

    puppet = puppets.Puppet(joint_forces=False,
                            string_forces=False,
                            string_constraints=True)

    puppet.q = {
        "torso_rx" : -0.05,
        #"torso_ry" : 0.1,
        "torso_tz" : 0.0,
        "lelbow_rx" : 1.57,
        "relbow_rx" : 1.57,
        "lhip_rx" : pi/2 - 0.6,
        "rhip_rx" : pi/2 - 0.6,
        "lknee_rx" : -pi/2 + 0.6,
        "rknee_rx" : -pi/2 + 0.6,
        }
    puppet.project_string_controls()

    q0 = puppet.q 
    u0 = tuple([0.0]*len(puppet.inputs))
    qk2_0 = puppet.qk
    dt = 0.01
    # Create and initialize a variational integrator for the system.
    gmvi = trep.MidpointVI(puppet)
    gmvi.initialize_from_configs(0.0, q0, dt, q0)

    left_leg_index = puppet.get_config("left_leg_string-length").k_index
    right_leg_index = puppet.get_config("right_leg_string-length").k_index
    left_arm_index = puppet.get_config("left_arm_string-length").k_index
    right_arm_index = puppet.get_config("right_arm_string-length").k_index

    def calc_vk():
        v2 = [(q2-q1)/(gmvi.t2 - gmvi.t1) for (q2, q1) in zip(gmvi.q2, gmvi.q1)]
        v2 = v2[len(puppet.dyn_configs):]
        return v2

    q = [gmvi.q2]
    p = [gmvi.p2]
    v = [calc_vk()]
    t = [gmvi.t2]
    while gmvi.t1 < 10.0:
        # For our kinematic inputs, we first copy our starting position.
        qk2 = list(qk2_0) 
        # Now we want to move some of the strings up and down.
        qk2[left_leg_index] += -0.1*sin(0.6*pi*gmvi.t1)
        qk2[right_leg_index] += 0.1*sin(0.6*pi*gmvi.t1)
        qk2[left_arm_index] += 0.1*sin(0.6*pi*gmvi.t1)
        qk2[right_arm_index] += -0.1*sin(0.6*pi*gmvi.t1)
        qk2 = tuple(qk2)

        gmvi.step(gmvi.t2+dt, u0, qk2)
        q.append(gmvi.q2)
        p.append(gmvi.p2)
        v.append(calc_vk())
        t.append(gmvi.t2)
        # The puppet can take a while to simulate, so print out the time
        # occasionally to indicate our progress.
        if abs(gmvi.t2 - round(gmvi.t2)) < dt/2.0:
            print "t =",gmvi.t2


    dsys = discopt.DSystem(gmvi, t)

    q = np.array(q)
    rho = q[1:,puppet.nQd:]

    X,U = dsys.build_trajectory(q, p, v, None, rho)

    return puppet, dsys, t, X, U


def calc_error(X0, X1):
    error = 0.0
    nd = puppet.nQd
    for x0, x1 in zip(X0, X1):
        error += np.linalg.norm(x1[:nd] - x0[:nd])
    return error




print "Creating initial trajectory."
(puppet, dsys, t, X0, U0) = create_initial_trajectory()

print "Calculating the linearization about the known trajectory."
(A, B) = dsys.linearize_trajectory(X0, U0)

print "Creating perturbation for initial condition."
x0_perturbed = create_offset_initial_condition(dsys, X0[0], U0[0])

# Create cost matrices
Q = np.diag([100.0]*len(dsys.system.configs) +
            [1.0]*len(dsys.system.dyn_configs) +
            [1.0]*len(dsys.system.kin_configs))
Qproj = lambda k: Q
#Qproj = [np.eye(dsys.nX)]*(dsys.kf()+1)
Rproj = lambda k: np.eye(dsys.nU)

print "Solving LQR problem."
K = dlqr.solve_tv_lqr(A, B, Qproj, Rproj)[0]

print "Simulating without feedback...",
X_open, U_open = simulate(dsys, x0_perturbed, U0)
open_error = calc_error(X0, X_open)
print "done (error: %f)" % open_error

print "Simulating with feedback...  ",
X_closed, U_closed = simulate(dsys, x0_perturbed, U0, K, X0)
closed_error = calc_error(X0, X_closed)
print "done (error: %f)" % closed_error


(Q_closed, p, v, u, rho) = dsys.split_trajectory(X_closed, U_closed)
(Q_open, p, v, u, rho) = dsys.split_trajectory(X_open, U_open)
(Qd, p, v, u, rho) = dsys.split_trajectory(X0, U0)

reference1 = trep.puppets.PuppetVisual(dsys.system, t, Qd)
closed_visual = trep.puppets.PuppetVisual(dsys.system, t, Q_closed)

reference2 = trep.puppets.PuppetVisual(dsys.system, t, Qd, offset=[0,1,0])
open_visual = trep.puppets.PuppetVisual(dsys.system, t, Q_open, offset=[0,1,0])

trep.visual.visualize_3d([reference1, closed_visual, reference2, open_visual],
                         camera_pos=[-2.038, 0.5134, 0.424],
                         camera_ang=[-0.033, 0.0170, 0.0])
