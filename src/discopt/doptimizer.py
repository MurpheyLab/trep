import datetime
import numpy as np

import trep
import dlqr

import numpy.linalg
from numpy import dot

try:
    import matplotlib.pyplot as pyplot
    pyplot_available = True
except ImportError:
    pyplot_available = False


class DOptimizerMonitor(object):
    def optimize_begin(self, X, U):
        pass
    def optimize_end(self, converged, X, U, cost):
        pass


    def step_begin(self, iteration):
        pass
    def step_info(self, method, cost, dcost, X, U, dX, dU, Kproj):
        pass
    def step_method_failure(self, method, cost, dcost, fallback_method):
        pass
    def step_termination(self, cost, dcost):
        pass
    def step_completed(self, method, cost, nX, nU):
        pass


    def armijo_simulation_failure(self, armijo_iteration, nX, nU, bX, bU):
        pass
    def armijo_search_failure(self, X, U, dX, dU, cost0, dcost0, Kproj):
        pass
    def armijo_evaluation(self, armijo_iteration, nX, nU, bX, bU, cost, max_cost):
        pass
    

class DOptimizerDefaultMonitor(DOptimizerMonitor):
    def __init__(self):
        self.iteration = 0
        self.start_cost = 0
        self.start_dcost = 0
        self.method = ''

        self.cost_history = {}
        self.dcost_history = {}


    def msg(self, msg):
        print "%s %3d: %s" % (
            datetime.datetime.now().strftime('[%H:%M:%S]'),
            self.iteration, msg)


    def optimize_begin(self, X, U):
        self.cost_history = {}
        self.dcost_history = {}


    def optimize_end(self, converged, X, U, cost):
        print ""


    def step_begin(self, iteration):
        self.iteration = iteration

        
    def step_info(self, method, cost, dcost, X, U, dX, dU, Kproj):
        self.start_cost = cost
        self.start_dcost = dcost
        self.method = method
        self.cost_history[self.iteration] = cost
        self.dcost_history[self.iteration] = dcost


    def step_method_failure(self, method, cost, dcost, fallback_method):
        self.msg("Descent method %r failed (dcost=%s), fallbacking to %s" % (
            original_method, dcost, fallback_method))


    def step_termination(self, cost, dcost):
        self.msg("Optimization Terminated.  cost=%s   dcost=%s" % (cost, dcost))


    def step_completed(self, method, cost, nX, nU):
        self.msg("cost=(%s => %s)  dcost=%s method=%s  armijo=%d" % (
            self.start_cost, cost, self.start_dcost, method, self.armijo))


    def armijo_simulation_failure(self, armijo_iteration, nX, nU, bX, bU):
        self.msg("  Armijo simulation (%d) failed after %d steps." % (
            armijo_iteration, len(nX)))
        
    
    def armijo_search_failure(self, X, U, dX, dU, cost0, dcost0, Kproj):
        pass

    
    def armijo_evaluation(self, armijo_iteration, nX, nU, bX, bU, cost, max_cost):
        self.armijo = armijo_iteration


class DOptimizer(object):
    def __init__(self, dsys, cost,
                 first_order_iterations=5,
                 monitor=None):
        self.dsys = dsys
        self.cost = cost
        self.optimize_ic = False

        if monitor is None:
            self.monitor = DOptimizerDefaultMonitor()
        else:
            self.monitor = monitor

        # Default weights used to generate feedback controller for the
        # projection.  These can be changed if desired.
        Qproj = np.eye(self.dsys.nX)
        Rproj = np.eye(self.dsys.nU)
        self.Qproj = lambda k: Qproj
        self.Rproj = lambda k: Rproj

        self.armijo_beta = 0.7
        self.armijo_alpha = 0.00001
        self.armijo_max_iterations = 30

        self.descent_tolerance = 1e-6

        # Number of first order iterations to do at the start of an optimization
        self.first_order_iterations = 10


    def calc_cost(self, X, U):        
        cost = 0.0
        for k in range(len(X)-1):
            cost += self.cost.l(X[k], U[k], k)
        cost += self.cost.m(X[-1])
        return cost


    def calc_dcost(self, X, U, dX, dU):        
        dcost = 0.0
        for k in range(len(X)-1):
            dcost += (dot(self.cost.l_dx(X[k], U[k], k), dX[k]) +
                      dot(self.cost.l_du(X[k], U[k], k), dU[k]))
        dcost += dot(self.cost.m_dx(X[-1]), dX[-1])
        return dcost


    def calc_ddcost(self, X, U, dX, dU, Q, R, S):
        ddcost = 0.0
        for k in range(len(X) - 1):
            ddcost += (dot(dX[k], dot(Q(k), dX[k])) +
                     2*dot(dX[k], dot(S(k), dU[k])) +
                       dot(dU[k], dot(R(k), dU[k])))
        ddcost += dot(dot(dX[-1], Q(-1)), dX[-1])
        return ddcost
           

    def calc_steepest_model(self):
        Q = np.eye(self.dsys.nX)
        R = np.eye(self.dsys.nU)
        S = np.zeros((self.dsys.nX, self.dsys.nU))
        return (lambda k: Q, lambda k: R, lambda k: S)


    def calc_quasi_model(self, X, U):
        Q = [None]*len(X)
        S = [None]*(len(X)-1)
        R = [None]*(len(X)-1)

        Q[-1] = self.cost.m_dxdx(X[-1])
        for k in reversed(range(len(X)-1)):
            Q[k] = self.cost.l_dxdx(X[k], U[k], k)
            S[k] = self.cost.l_dxdu(X[k], U[k], k)
            R[k] = self.cost.l_dudu(X[k], U[k], k)
            
        return (lambda k: Q[k], lambda k: R[k], lambda k: S[k])
    

    def calc_newton_model(self, X, U, A, B, K):
        Q = [None]*len(X)
        S = [None]*(len(X)-1)
        R = [None]*(len(X)-1)

        z = self.cost.m_dx(X[-1])
        Q[-1] = self.cost.m_dxdx(X[-1])
        for k in reversed(range(len(X)-1)):
            Q[k] = self.cost.l_dxdx(X[k], U[k], k)
            S[k] = self.cost.l_dxdu(X[k], U[k], k)
            R[k] = self.cost.l_dudu(X[k], U[k], k)
            self.dsys.set(X[k], U[k], k,
                          xk_hint=X[k+1])
            Q[k] += self.dsys.fdxdx(z)
            S[k] += self.dsys.fdxdu(z)
            R[k] += self.dsys.fdudu(z)

            z = (self.cost.l_dx(X[k], U[k], k) -
                 dot(self.cost.l_du(X[k], U[k], k), K[k]) +
                 dot(z, (A[k] - dot(B[k], K[k]))))
        
        return (lambda k: Q[k], lambda k: R[k], lambda k: S[k])
        
    
    def calc_descent_direction(self, X, U, method='steepest'):
        (Kproj, A, B) = self.calc_projection(X, U, True)

        # All descent direction methods use the same linear cost
        # terms.
        q = np.zeros(X.shape)
        r = np.zeros(U.shape)
        for k in xrange(len(X)-1):
            q[k] = self.cost.l_dx(X[k], U[k], k)
            r[k] = self.cost.l_du(X[k], U[k], k)
        q[-1] = self.cost.m_dx(X[-1])

        # Calculate the quadratic model according to the desired
        # method.
        if method == 'steepest':
            (Q,R,S) = self.calc_steepest_model()
        elif method == 'quasi':
            (Q,R,S) = self.calc_quasi_model(X, U)
        elif method == 'newton':
            (Q,R,S) = self.calc_newton_model(X, U, A, B, Kproj)
        else:
            raise StandardError("Invalid descent direction method: %r" % method)

        (K,C,P,b) = dlqr.solve_tv_lq(A, B, q, r, Q, S, R)

        # If the optimization includes initial conditions, we need to
        # find an initial condition that minimizes the LQ solution.
        # This currently is only valid for unconstrained systems.
        if self.optimize_ic:
            dx0 = -np.linalg.solve(P, b)
        else:
            dx0 = np.zeros((self.dsys.nX,))

        # Calculate the descent direction by simulating the linearized
        # system using the LQ solution's optimal input.
        dX = np.zeros(X.shape)
        dU = np.zeros(U.shape)
        dX[0] = dx0
        for k in xrange(len(X)-1):
            dU[k] = -dot(K[k],dX[k]) - C[k] 
            dX[k+1] = dot(A[k],dX[k]) + dot(B[k],dU[k])
            
        return (Kproj, dX, dU, Q, R, S)


    def armijo_simulate(self, bX, bU, Kproj):
        nX = np.zeros(bX.shape)
        nU = np.zeros(bU.shape)
        try:
            nX[0] = bX[0]
            for k in range(len(bX)-1):
                nU[k] = bU[k] - dot(Kproj[k], nX[k] - bX[k])
                if k == 0:
                    self.dsys.set(nX[k], nU[k], k)
                else:
                    self.dsys.step(nU[k])
                nX[k+1] = self.dsys.f()
        except trep.ConvergenceError:
            return (False, nX[:k], nU[:k])
        return (True, nX, nU)

        
    def armijo_search(self, X, U, Kproj, dX, dU):

        cost0 = self.calc_cost(X, U)
        dcost0 = self.calc_dcost(X, U, dX, dU)
        
        for m in range(0, self.armijo_max_iterations):
            lam = self.armijo_beta**m
            max_cost = cost0 + self.armijo_alpha* lam * dcost0

            bX = X + lam*dX
            bU = U + lam*dU

            (result, nX, nU) = self.armijo_simulate(bX, bU, Kproj)

            if not result:
                self.monitor.armijo_simulation_failure(m, nX, nU, nX, bU)
                continue

            cost1 = self.calc_cost(nX, nU)
            self.monitor.armijo_evaluation(m, nX, nU, bX, bU, cost1, max_cost)
            if cost1 < max_cost:
                return (nX, nU, cost1)
        else:
            self.monitor.armijo_search_failure(X, U, dX, dU, cost0, dcost0, Kproj)
            raise StandardError("Armijo Failed to Converge")


    def step(self, iteration, X, U, method='steepest'):

        self.monitor.step_begin(iteration)

        (Kproj, dX, dU, Q, R, S) = self.calc_descent_direction(X, U, method)
        
        cost0 = self.calc_cost(X, U)
        dcost0 = self.calc_dcost(X, U, dX, dU)

        self.monitor.step_info(method, cost0, dcost0, X, U, dX, dU, Kproj)

        # Check for sane descent direction
        if dcost0 > 0:
            if method != 'steepest':
                fallback = self.select_fallback_method(iteration, method)
                self.monitor.step_method_failure(method, cost0, dcost0, fallback)
                return self.step(iteration, X, U, fallback)
            else:
                # This should never occur
                raise StandardError("Derivative of cost is positive for steepest descent.")

        # Check for terminal condition
        if abs(dcost0) < self.descent_tolerance:
            self.monitor.step_termination(cost0, dcost0)
            return (True, X, U, dcost0, cost0)

        # Line search in descent direction
        (X, U, cost1) = self.armijo_search(X, U, Kproj, dX, dU)

        self.monitor.step_completed(method, cost1, X, U)

        return (False, X, U, dcost0, cost1)


    def calc_projection(self, X, U, return_linearization=False):
        return self.dsys.calc_feedback_controller(X, U,
                                                  self.Qproj, self.Rproj,
                                                  return_linearization)

 
    def dproject(self, A, B, bdX, bdU, K):
        dX = np.zeros(bdX.shape)
        dU = np.zeros(bdU.shape)
        dX[0] = bdX[0]
        for k in xrange(len(bdX)-1):
            dU[k] = bdU[k] - dot(K[k], dX[k] - bdX[k])
            dX[k+1] = dot(A[k],dX[k]) + dot(B[k],dU[k])
        return dX, dU


    def select_method(self, iteration):
        if iteration < self.first_order_iterations:
            method = 'quasi'
        else:
            method = 'newton'
        return method

    def select_fallback_method(self, iteration, current_method):
        return 'steepest'

            
    def optimize(self, X, U, max_steps=50):
        X = np.array(X)
        U = np.array(U)

        self.monitor.optimize_begin(X, U)

        for i in range(max_steps):
            method = self.select_method(i)
            (converged, X, U, cost, method) = self.step(i, X, U, method)
            if converged:
                break

        self.monitor.optimize_end(converged, X, U, cost)
        return (converged, X, U)


    def descent_plot(self, X, U, method='steepest', points=40, legend=True):

        if not pyplot_available:
            raise StandardError("Importing matplotlib failed. Cannot create plot.")

        (Kproj, dX, dU, Q, R, S) = self.calc_descent_direction(X, U, method)

        def calc_cost(zi):
            return self.calc_cost(*self.dsys.project(X + zi*dX, U + zi*dU, Kproj))

        armijo_z = np.array(sorted([self.armijo_beta**m for m in range(20)]))
        z = np.linspace(-0.1, 1.01, points)
        z = np.concatenate((z, armijo_z))
        z.sort()

        # Used to calculate the model costs
        cost = self.calc_cost(X, U)
        dcost = self.calc_dcost(X, U, dX, dU)
        ddcost = self.calc_ddcost(X, U, dX, dU, Q, R, S)

        true_cost = np.zeros(z.shape)
        model_cost = np.zeros(z.shape)
        for i,zi in enumerate(z):
            true_cost[i] = calc_cost(zi)
            model_cost[i] = cost + dcost*zi + 0.5*ddcost*zi*zi


        armijo_cost = np.zeros(armijo_z.shape)
        for i,zi in enumerate(armijo_z):
            armijo_cost[i] = calc_cost(zi)

        armijo_max = np.zeros(z.shape)
        for i,zi in enumerate(z):
            armijo_max[i] = cost + self.armijo_alpha* zi * dcost


        pyplot.plot(z, model_cost-cost, '-,', linewidth=2.0, color='blue', label='Modeled Cost')
        pyplot.plot(z, true_cost-cost, '.-', linewidth=1.0, color='black', label='True Cost')
        pyplot.plot(armijo_z, armijo_cost-cost, 'o', color='gray', label='Armijo Evaluations')
        pyplot.plot(z, armijo_max-cost, '-.', color='black', label='Required Cost Improvement')
        if legend:
            pyplot.legend(loc=0)

        pyplot.title('Cost along descent direction for method: "%s".' % method)
        pyplot.xlabel('z')
        pyplot.ylabel(r'$\Delta$ cost')


    def check_dcost(self, X, U, method='steepest', delta=1e-6, tolerance=1e-5):

        (Kproj, dX, dU, Q, R, S) = self.calc_descent_direction(X, U, method)
        exact_dcost = self.calc_dcost(X, U, dX, dU)

        nX, nU = self.dsys.project(X - delta*dX, U - delta*dU, Kproj)
        cost0 = self.calc_cost(nX, nU)

        nX, nU = self.dsys.project(X + delta*dX, U + delta*dU, Kproj)
        cost1 = self.calc_cost(nX, nU)
                
        approx_dcost = (cost1 - cost0)/(2*delta)
        error = approx_dcost - exact_dcost
        result = (abs(error) <= tolerance)
        return (result, error, cost1, cost0, approx_dcost, exact_dcost)
    

    def check_ddcost(self, X, U, method='steepest', delta=1e-6, tolerance=1e-5):

        (Kproj, dX, dU, Q, R, S) = self.calc_descent_direction(X, U, method)
        if method != 'newton':
            (Q, R, S) = self.calc_descent_direction(X, U, 'newton')[-3:]

        exact_ddcost = self.calc_ddcost(X, U, dX, dU, Q, R, S)

        # Calculate cost0
        bX = X - delta*dX
        bU = U - delta*dU
        nX, nU = self.dsys.project(bX, bU, Kproj)
        (A, B) = self.dsys.linearize_trajectory(nX, nU)
        (ndX, ndU) = self.dproject(A, B, dX, dU, Kproj)
        dcost0 = self.calc_dcost(nX, nU, ndX, ndU)

        # Calculate cost1
        bX = X + delta*dX
        bU = U + delta*dU
        nX, nU = self.dsys.project(bX, bU, Kproj)
        (A, B) = self.dsys.linearize_trajectory(nX, nU)
        (ndX, ndU) = self.dproject(A, B, dX, dU, Kproj)
        dcost1 = self.calc_dcost(nX, nU, ndX, ndU)
        
        approx_ddcost = (dcost1 - dcost0)/(2*delta)
        error = approx_ddcost - exact_ddcost
        result = (abs(error) <= tolerance)
        
        return (result, error, dcost1, dcost0, approx_ddcost, exact_ddcost)

        
            
