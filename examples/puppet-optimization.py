# This is an alternate experiment to try optimizing a
# constraint-string puppet from a stationary trajectory to try to get
# better intermediate trajectories that illustrate the optimization
# process better.

from math import pi, sin, cos
import sys
import numpy as np
import datetime
import trep
from trep import discopt
from trep.puppets import Puppet, PuppetVisual
import scipy.io as sio
import pylab
import trep.visual as visual
import os.path
from trep.visual import visualize_3d


QD_COST = 100.0
QK_COST = 1.0
PD_COST = 1.0
VK_COST = 1.0
RHO_COST = 0.1


def generate_desired_trajectory():
    puppet = trep.puppets.Puppet(joint_forces=False, string_forces=False,
                                 string_constraints=True)


    puppet.q = {
        'torso_rx' : -0.05,
        #'torso_ry' : 0.1,
        'torso_tz' : 0.0,
        'lelbow_rx' : 1.57,
        'relbow_rx' : 1.57,
        'lhip_rx' : pi/2 - 0.6,
        'rhip_rx' : pi/2 - 0.6,
        'lknee_rx' : -pi/2 + 0.6,
        'rknee_rx' : -pi/2 + 0.6,
        ## 'Left Arm String' : 15.4,
        ## 'Right Arm String' : 15.4,
        ## 'Left Leg String' : 16.6,
        ## 'Right Leg String' : 16.6
        }
    puppet.project_string_controls()

    q0 = puppet.q
    u0 = tuple([0.0]*len(puppet.inputs))
    qk2_0 = puppet.qk
    dt = 0.01
    # Create and initialize a variational integrator for the system.
    mvi = trep.MidpointVI(puppet)
    mvi.initialize_from_configs(0.0, q0, dt, q0)

    #print qk2

    left_leg_index = puppet.get_config('left_leg_string-length').k_index
    right_leg_index = puppet.get_config('right_leg_string-length').k_index
    left_arm_index = puppet.get_config('left_arm_string-length').k_index
    right_arm_index = puppet.get_config('right_arm_string-length').k_index


    def calc_vk():
        v2 = [(q2-q1)/(mvi.t2 - mvi.t1) for (q2, q1) in zip(mvi.q2, mvi.q1)]
        v2 = v2[len(puppet.dyn_configs):]
        return v2

    q = [mvi.q2]
    p = [mvi.p2]
    v = [calc_vk()]
    rho = []
    t = [mvi.t2]
    while mvi.t1 < 10.0:
        # For our kinematic inputs, we first copy our starting position.
        qk2 = list(qk2_0) 
        # Now we want to move some of the strings up and down.
        qk2[left_leg_index] += -0.1*sin(0.6*pi*mvi.t1)
        qk2[right_leg_index] += 0.1*sin(0.6*pi*mvi.t1)
        qk2[left_arm_index] += 0.1*sin(0.6*pi*mvi.t1)
        qk2[right_arm_index] += -0.1*sin(0.6*pi*mvi.t1)
        qk2 = tuple(qk2)
        rho.append(qk2)

        mvi.step(mvi.t2+dt, u0, qk2)
        q.append(mvi.q2)
        p.append(mvi.p2)
        v.append(calc_vk())
        t.append(mvi.t2)
        # The puppet can take a while to simulate, so print out the time
        # occasionally to indicate our progress.
        if abs(mvi.t2 - round(mvi.t2)) < dt/2.0:
            print "t =",mvi.t2


    t = np.array(t)
    q = np.array(q)
    p = np.array(p)
    v = np.array(v)
    rho = np.array(rho)

    trep.system.save_trajectory('puppet-desired.mat',
                                puppet, t, q, p, v, None, rho)    



def create_system(joint_forces=False, string_forces=False,
                  string_constraints=False, config_trajectory=False):
    # Creates the mechanical system, variational integrator, and discrete system
    # Creates a desired trajectory suitable for the mechanical system from the ik data

    system = Puppet(joint_forces=joint_forces,
                                 string_forces=string_forces,
                                 string_constraints=string_constraints)

    if not os.path.isfile('puppet-desired.mat'):
        generate_desired_trajectory()        

    (t,Qd,p,v,u,rho) = trep.load_trajectory('puppet-desired.mat', system)
    # Disregard all data except t,Q
    dsys = discopt.DSystem(trep.MidpointVI(system), t)
    xid = dsys.build_trajectory(Qd)

    return dsys, xid


def create_initial_trajectory(dsys, xid, out_filename):


    x0 = xid[0][0]
    Q, p, v = dsys.split_state(x0)

    puppet = dsys.system
    puppet.q = Q
    puppet.project_string_controls()
    puppet.correct_string_lengths()
    Q = puppet.q

    x0 = dsys.build_state(Q, p, v)

    # Simulate the trajectory
    qk = puppet.qk

    X = [x0]
    U = []
    for k in range(len(xid[0])-1):
        u = qk
        if k == 0:
            dsys.set(X[0], u, k)
        else:
            dsys.step(u)
        X.append(dsys.f())
        U.append(u)

    X = np.array(X)
    U = np.array(U)
    dsys.save_state_trajectory(out_filename, X, U)
    


def optimize_trajectory(dsys, xid, in_filename, out_filename):
    # Optimizes a trajectory for the discrete system.  The initial
    # trajectory is read from in_filename.  The results are saved to
    # out_filename.

    weight = (
        [QD_COST]*len(dsys.system.dyn_configs) +  
        [QK_COST]*len(dsys.system.kin_configs) +
        [PD_COST]*len(dsys.system.dyn_configs) +
        [VK_COST]*len(dsys.system.kin_configs)
        )
    Qcost = np.diag(weight)
    weight = (
        #[MU2_COST]*len(dsys.system.inputs) +
        [RHO_COST]*len(dsys.system.kin_configs)
        )
    for force in dsys.system.joint_forces.values():
        weight[force.finput.index] = mu1_cost
    Rcost = np.diag(weight)
    cost = discopt.DCost(xid[0], xid[1], Qcost, Rcost)

    X, U = dsys.load_state_trajectory(in_filename)
    monitor = trep.discopt.DOptimizerVerboseMonitor()
    optimizer = discopt.DOptimizer(dsys, cost, monitor=monitor)
    optimizer.descent_tolerance = 1e-2

    optimizer.lower_order_iterations = 30

    finished, X, U = optimizer.optimize(X,U, max_steps=30)
    
    dsys.save_state_trajectory(out_filename, X, U)
    return (monitor.get_costs(), monitor.get_dcosts())
    

def visualize_trajectory(dsys, xid, filename=None):
    xi = dsys.load_state_trajectory(filename)
    (q, p, v, mu, rho) = dsys.split_trajectory(*xi)
    (qd, p, v, mu, rho) = dsys.split_trajectory(*xid)
    t = dsys._time
    t = t[:min(len(t), len(q))]
    q = q[:len(t)]

    position = [-1.695, -0.780, 0.424]
    angle = [0.462, 0.24, 0.0]

    #item1 = PuppetVisual(dsys.system, t, qd)
    item2 = trep.puppets.PuppetVisual(dsys.system, t, q)

    visual.visualize_3d([item2],
                        camera_pos=position,
                        camera_ang=angle)




(dsys, xid) = create_system(joint_forces=False, string_forces=False,
                                      string_constraints=True)

create_initial_trajectory(dsys, xid, "puppet-initial.mat")
convergence,dconvergence = optimize_trajectory(dsys, xid, "puppet-initial.mat", "puppet-result.mat")
visualize_trajectory(dsys, xid, 'puppet-result.mat')
                         


              




