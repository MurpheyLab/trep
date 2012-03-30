import sys
import math
from math import pi, sin
import time
import trep
import trep.potentials
import numpy as np

def write_csv(filename, t, u):
    out = open(filename, 'wt')

    for i in range(len(u)):
        line = (t[i],) + u[i]
        out.write(', '.join(['%f' % f for f in line]) + '\n')
    out.close()


def main(links, dt, tf, generalized_method, hybrid_wrenched):
    # Create the system and integrator
    system = make_pendulum(links, hybrid_wrenched)
    mvi = trep.MidpointVI(system)
      
    # Generate a reference trajectory
    (t_ref, q_ref, u_ref) = generate_reference(mvi, dt, tf)
    write_csv('ref.csv', t_ref, u_ref)

    # Uncomment to display the reference trajectory
    #visualize(system, t_ref, q_ref, u_ref)

    # Perform inverse dynamics
    if generalized_method:
        (t_sol, u_sol) = inverse_dynamics_generalized(mvi, t_ref, q_ref)
    else:
        (t_sol, u_sol) = inverse_dynamics_specialized(mvi, t_ref, q_ref)

    write_csv('sol.csv', t_sol, u_sol)

    # Simulate the system with the new inputs
    (t, q, u) = simulate_system(mvi, t_sol, u_sol, q_ref[0], q_ref[1])

    # Uncomment to display the reconstructed trajectory
    #visualize(system, t, q, u)

    verify_solution(q_ref, q)


def visualize(system, t, q, u=None):
    viewer = trep.visual.SystemTrajectoryViewer(system, t, q, u)
    viewer.print_instructions()
    viewer.run()

          
def make_pendulum(num_links, hybrid_wrenched=False):
    """
    make_pendulum(num_links, hybrid_wrenched) -> System
    
    Create a forced pendulum system with num_links.  The pendulum is
    actuated by either direct joint torques or body wrenches in the
    local X direction depending on the hybrid_wrenched parameter.
    """
    def add_level(frame, link=0):
        """
        Recusively add links to a system by attaching a new link to
        the specified frame.
        """
        if link == num_links:
            return

        # Create a rotation for the pendulum.
        # The first argument is the name of the frame, the second is
        # the transformation type, and the third is the name of the
        # configuration variable that parameterizes the
        # transformation.
        config_name = 'theta-%d' % link
        child = trep.Frame(frame, trep.RY, config_name, "link-%d" % link)

        if not hybrid_wrenched:
            trep.forces.ConfigForce(child.system, config_name, 'torque-%d' % link,
                                   name='joint-force-%d' % link)

        # Move down to create the length of the pendulum link.
        child = trep.Frame(child, trep.TZ, -1)
        # Add mass to the end of the link (only a point mass, no
        # rotational inertia)
        child.set_mass(1.0)

        if hybrid_wrenched:
            trep.forces.HybridWrench(child.system, child,
                                     ('wrench-x-%d' % link, 0.0, 0.0),
                                     name='wrench-%d' % link)
        
        add_level(child, link+1)

    # Create a new system, add the pendulum links, and rotate the top
    # pendulum.
    system = trep.System()
    trep.potentials.Gravity(system, name="Gravity")
    trep.forces.Damping(system, 1.0)
    add_level(system.world_frame)
    return system


def generate_reference(mvi, dt, tf):
    """
    Generate a reference trajectory for the inverse dynamics
    to... uhh.. invert.  Applies sinusoids with different amplitudes
    and phases to each joint.  Returns the calculated trajectory.
    """
    def forcing(t):
        Nu = len(mvi.system.inputs)
        u = [(0.8*(Nu-i+1)*sin(3.0*t + pi/4.0*i)) for i in range(Nu)]
        return tuple(u)

    t0 = 0.0
    q0 = tuple([0.0] * len(mvi.system.configs))
    u0 = forcing(t0)
    t1 = t0 + dt
    q1 = tuple([0.0] * len(mvi.system.configs))

    mvi.initialize_from_configs(t0, q0, t1, q1)
    t = [t0, t1]
    q = [q0, q1]
    u = [u0]

    while mvi.t1 < tf:
        u.append(forcing(mvi.t2))
        mvi.step(mvi.t2 + dt, u[-1])
        t.append(mvi.t2)
        q.append(mvi.q2)
    return t, q, u
    

def inverse_dynamics_specialized(mvi, t_ref, q_ref):
    """
    For systems where we have full joint actuation. (ie, a force input
    applied directly to each configuration variable, we calculate the
    inverse dynamics by evaluating the variational integrator equation
    with q1,p1,q2 known, and u=0.  The remainder of the equation is
    the applied discrete force necessary to make the equation equal to
    zero.

    There method will not work for general forces.
    """   
    zero_u = tuple( [0.0]*len(mvi.system.inputs) )

    t0 = t_ref[0]
    q0 = q_ref[0]
    u0 = zero_u
    t1 = t_ref[1]
    q1 = q_ref[1]

    mvi.initialize_from_configs(t0, q0, t1, q1)

    u = [list(u0)]

    for k in range(2, len(t_ref)):
        # First we advance the integrator
        mvi.t1 = mvi.t2
        mvi.q1 = mvi.q2
        mvi.p1 = mvi.p2

        # Set the next configuration
        mvi.t2 = t_ref[k]
        mvi.q2 = q_ref[k]

        # Set the inputs to zero
        mvi.u1 = zero_u

        # Calculate the DEL
        f = mvi.calc_f()

        # Calculate u from f
        u_k = ([-f[i]/(mvi.t2-mvi.t1)
                for i in range(len(mvi.system.inputs))])

        # Save the result
        u.append(u_k)

        # Set u1 to the new inputs so we can calculate the correct
        # momentum
        mvi.u1 = tuple(u_k)
        mvi.calc_p2()

        # And continue to the next time step

    # Convert to tuples for trep (stupid requirement that ought to be
    # removed).
    for j in range(len(u)):
        u[j] = tuple(u[j])
        
    return t_ref, u
    

def inverse_dynamics_generalized(mvi, t_ref, q_ref):
    """
    CURRENTLY NOT WORKING :/
    
    This is a more general, but more computationally expensive and
    slower, method for inverse dynamics.  Unlike the specialized
    algorithm, it can handle indirect actuation like body-wrenches.
    Additionally, it can handle cases where the next step is actually
    unreachable (e.g, that would violate constraints) because it
    solves a minimization problem.  Note that this code does not
    handle constrained systems, that requires a few more
    modifications.

    We will represent the variational integrator as a discrete
    mechanical system using trep's DSystem wrapper.  This gives us a
    form such that x[k+1] = f(x[k], u[k]).

    The idea is that we will minimize the cost function g(u):
    g(u) = 2 || x[k+1] - f(x[k], u[k]) ||
    using a iterative algorithm.
    """

    ## zero_u = tuple( [0.0]*len(mvi.system.inputs) )

    ## q0 = q_ref[0]
    ## u0 = zero_u
    ## q1 = q_ref[1]


    ## #u0 = (0.000000, 2.828427), 3.200000, 1.697056, 0.000000)
    ## #u0 = (0.000000, 2.818427, 3.200000, 1.697056, 0.000000)
    ## #u0 = (50.0,)
    ## #u0 = (5.0, 5.0, 5.0, 5.0, 5.0)
    ## q_sol = [q0, q1]
    ## u_sol = [u0]



    ## ## mvi.initialize_from_configs(t_ref[0], q_sol[0], u_sol[0],
    ## ##                              t_ref[1], q_sol[1])
    ## ## mvi.step(t_ref[2], (50.0,))
    ## ## print mvi.q2
    ## ## print mvi.q2_du1(mvi.system.configs[0], mvi.system.inputs[0])
    ## ## asdfasdf


    ## for k in range(1, len(q_ref)-1):

    ##     def calc_g(u, first_derivative=False, second_derivative=False):
            

    ##         mvi.initialize_from_configs(t_ref[k-1], q_sol[k-1], u_sol[k-1],
    ##                                      t_ref[k], q_sol[k])
    ##         mvi.step(t_ref[k+1], tuple(u))
    ##         e = np.matrix(mvi.q2).T - np.matrix(q_ref[k+1]).T
    ##         ## print '\n'
    ##         ## print mvi.q2
    ##         ## print q_ref[k+1]
    ##         ## print e
    ##         Q = 1000.0
    ##         cost = 0.5*Q*e.T*e
    ##         result = [cost[0,0]]

    ##         if first_derivative:
    ##             cost_du = np.matrix(np.zeros((1, len(mvi.system.inputs))))
    ##             for i,ui in enumerate(mvi.system.inputs):
    ##                 for a,qa in enumerate(mvi.system.configs):
    ##                     cost_du[0,i] += Q*e[a,0]*mvi.q2_du1(qa,ui)
    ##             result.append(cost_du)
                
    ##         if second_derivative:
    ##             cost_dudu = np.matrix(np.zeros((len(mvi.system.inputs),
    ##                                             len(mvi.system.inputs))))

    ##             for i,ui in enumerate(mvi.system.inputs):
    ##                 for j,uj in enumerate(mvi.system.inputs):
    ##                     for a,qa in enumerate(mvi.system.configs):
    ##                         cost_dudu[i,j] += Q*(
    ##                             mvi.q2_du1(qa,ui)*mvi.q2_du1(qa, uj)
    ##                             + e[a,0]*mvi.q2_du1du1(qa,ui, uj))
    ##             result.append(cost_dudu)
                    
    ##         return result
        
    ##     iterations = 0
    ##     second = False
    ##     u = u_sol[-1]

    ##     ## (g0, dg) = calc_g(u, True, False)

    ##     ## dg_approx = [0.0]*len(mvi.system.inputs)
    ##     ## for i in range(len(mvi.system.inputs)):
    ##     ##     delta = 1e-6
    ##     ##     u1 = list(u)
    ##     ##     u1[i] += delta
    ##     ##     (g1,) = calc_g(u1)
    ##     ##     dg_approx[i] = (g1-g0)/delta

    ##     ## print dg.tolist()[0]
    ##     ## print dg_approx
    ##     ## asdfadf
                   
        
    ##     while True:
    ##         if iterations > 100:
    ##             second = True
            
    ##         if second:
    ##             (g, dg, d2g) = calc_g(u, True, True)
    ##         else:
    ##             (g, dg) = calc_g(u, True, False)

    ##         # Check for terminal condition
    ##         norm_dg = np.linalg.norm(dg)
    ##         print "%4s-%3s: g=%s |dg|=%s" % (k, iterations, g, norm_dg)
    ##         if norm_dg < 1e-10:
    ##             break

    ##         ## if second:
    ##         if False:
    ##             q = d2g
    ##             z = -2*np.linalg.pinv(q)*dg.T
                
    ##             if dg*z >= -1e-18:
    ##                 print dg*z
    ##                 print "falling back to 1st order"
    ##                 q = np.matrix(np.eye(len(u)))
    ##                 z = -2*np.linalg.pinv(q)*dg.T
    ##         else:
    ##             z = -2*dg.T

    ##         ## print z
    ##         ## print "\n"
            
    ##         # Armijo
    ##         for m in range(1000):
    ##             lam = 0.7**m
    ##             u1 = np.matrix(u).T + lam*z
    ##             u1 = tuple(u1.T.tolist()[0])
    ##             #print "burp", u1
    ##             #try:
    ##             (g1,) = calc_g(u1)
    ##             #except StandardError:
    ##             #    pass

    ##             if g1 < g + 0.4*lam*(dg*z):
    ##                 break
                
    ##         else:
    ##             print dg*z
    ##             print g1
    ##             print g
    ##             print 0.4*lam*dg*z
    ##             raise StandardError('armijo failed to converge (%s)' % m)
    ##             break
    ##         u = u1
    ##         #print u
    ##         iterations += 1
        
    ##     q_sol.append(mvi.q2)
    ##     u_sol.append(u)

    ## return t_ref, u
    


def simulate_system(mvi, t, u, q0, q1):

    t0 = t[0]
    u0 = u[0]
    t1 = t[1]

    mvi.initialize_from_configs(t0, q0, t1, q1)
    q = [q0, q1]

    for i in range(1, len(t)-1):
        mvi.step(t[i+1], u[i])
        q.append(mvi.q2)

    return t, q, u


def verify_solution(q_ref, q):

    q_ref = np.array(q_ref)
    q = np.array(q)

    error = np.linalg.norm(q_ref - q)
    print "Error between reference and solution trajectories:"
    print error


# Run the script
main(links=5, dt=0.01, tf=10.0,
     generalized_method=False, hybrid_wrenched=False)



