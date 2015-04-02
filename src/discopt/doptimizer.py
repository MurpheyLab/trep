import time
import datetime
import numpy as np

import trep
import dlqr

import numpy.linalg
from numpy import dot
from collections import namedtuple

try:
    import matplotlib.pyplot as pyplot
    pyplot_available = True
except ImportError:
    pyplot_available = False


class DOptimizerMonitor(object):
    """
    This is the base class for Optimizer Monitors.  It does absolutely
    nothing, so you can use this as your monitor if you want
    completely silent operation.
    """
    
    def optimize_begin(self, X, U):
        """
        Called when DOptimizer.optimize() is called with the initial
        trajectory.
        """
        pass    
    def optimize_end(self, converged, X, U, cost):
        """
        Called before DOptimizer.optimzie() returns with the results
        of the optimization.
        """
        pass


    def step_begin(self, iteration):
        """
        Called at the start of each DOptimize.step().  Note that step
        calls itself with the new method when one method fails, so
        this might be called multiple times with the same iteration.

        All calls will be related to the same iteration until
        step_termination or step_completed are called.
        """
        pass
    def step_info(self, method, cost, dcost, X, U, dX, dU, Kproj):
        """
        Called after a descent direction has been calculated.
        """
        pass
    def step_method_failure(self, method, cost, dcost, fallback_method):
        """
        Called when a descent method results in a positive cost
        derivative. 
        """
        pass
    def step_termination(self, cost, dcost):
        """
        Called if dcost satisfies the descent tolerance, indicating
        that the current trajectory is a local minimizer.
        """
        pass
    def step_completed(self, method, cost, nX, nU):
        """
        Called at the end of an optimization step with information
        about the new trajectory.
        """
        pass


    def armijo_simulation_failure(self, armijo_iteration, nX, nU, bX, bU):
        """
        Called when a simulation fails (usually an instability) during
        the evaluation of the cost in an armijo step.  The Armijo
        search continues after this.
        """
        pass    
    def armijo_search_failure(self, X, U, dX, dU, cost0, dcost0, Kproj):
        """
        Called when the Armijo search reaches the maximum number of
        iterations without satisfying the sufficient decrease
        criteria.  The optimization cannot proceed after this.
        """
        pass
    def armijo_evaluation(self, armijo_iteration, nX, nU, bX, bU, cost, max_cost):
        """
        Called after each Armijo evaluation.  The semi-trajectory
        bX,bU was succesfully projected into the new trajectory nX,nU
        and its cost was measured.  The search will continue if the
        cost is greater than the maximum cost.
        """
        pass
    

class DOptimizerDefaultMonitor(DOptimizerMonitor):
    """
    This is the default DOptimizer Monitor.  It mainly prints status
    updates to stdout and records the cost and dcost history.
    """
    
    def __init__(self):
        self.iteration = 0
        self.start_cost = 0
        self.start_dcost = 0
        self.method = ''
        self.start_time = None

        self.cost_history = {}
        self.dcost_history = {}


    def msg(self, msg):
        if self.start_time is not None:
            delta = datetime.datetime.now() - self.start_time
            timestamp = time.strftime("+[%H:%M:%S]",time.gmtime(delta.seconds))            
        else:
            timestamp = datetime.datetime.now().strftime('[%H:%M:%S]')
        print "%s %3d: %s" % (timestamp, self.iteration, msg)

    def optimize_begin(self, X, U):
        self.cost_history = {}
        self.dcost_history = {}
        self.start_time = datetime.datetime.now()


    def optimize_end(self, converged, X, U, cost):
        print ""
        self.start_time = None


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
            self.method, dcost, fallback_method))


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

    def get_costs(self):
        return [self.cost_history[x] for x in sorted(self.cost_history.keys())]

    def get_dcosts(self):
        return [self.dcost_history[x] for x in sorted(self.dcost_history.keys())]
    


class DOptimizerVerboseMonitor(DOptimizerDefaultMonitor):
    """
    The verbose DOptimizer Monitor prints more information.
    """
   
    def optimize_begin(self, X, U):
        self.msg("Optimization starting at %s" % datetime.datetime.now().strftime('[%H:%M:%S]'))
        super(DOptimizerVerboseMonitor, self).optimize_begin(X, U)


    def optimize_end(self, converged, X, U, cost):
        self.msg("Optimization completed at %s" % datetime.datetime.now().strftime('[%H:%M:%S]'))
        super(DOptimizerVerboseMonitor, self).optimize_end(converged, X, U, cost)


    def step_info(self, method, cost, dcost, X, U, dX, dU, Kproj):
        self.msg("Current Trajectory cost: %f, dcost: %f, method=%s" % (cost, dcost, method))
        super(DOptimizerVerboseMonitor, self).step_info(method, cost, dcost, X, U, dX, dU, Kproj)

    
    def armijo_evaluation(self, armijo_iteration, nX, nU, bX, bU, cost, max_cost):
        if cost >= max_cost:
            self.msg("  Armijo evaluation (%d) is too expensive (%f >= %f)" % (
                armijo_iteration, cost, max_cost))
        super(DOptimizerVerboseMonitor, self).armijo_evaluation(
            armijo_iteration, nX, nU, bX, bU, cost, max_cost)


class DOptimizer(object):
    def __init__(self, dsys, cost,
                 first_method_iterations=10,
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
        self.first_method_iterations = first_method_iterations
        self.first_method = 'quasi'
        self.second_method = 'newton'
    
        # Named tuples types for function returns
        self.step_return = namedtuple('step', 'done nX nU dcost0 cost1')
        self.optimize_return = namedtuple('optimize', 'converged X U')
        self.check_dcost_return = namedtuple('check_dcost', 'result error cost1 cost0 approx_dcost exact_dcost')
        self.check_ddcost_return = namedtuple('check_ddcost', 'result error cost1 cost0 approx_ddcost exact_ddcost')
        self.model_return = namedtuple('descent_model', 'Q R S')
        self.descent_return = namedtuple('calc_descent_direction', 'Kproj dX dU Q R S')
        self.armijo_simulate_return = namedtuple('armijo_simulate', 'success nX nU')
        self.armijo_search_return = namedtuple('armijo_search', 'nX nU cost1')

    def calc_cost(self, X, U):
        """Calculate the cost of a trajectory X,U."""
        cost = 0.0
        for k in range(len(X)-1):
            cost += self.cost.l(X[k], U[k], k)
        cost += self.cost.m(X[-1])
        return cost


    def calc_dcost(self, X, U, dX, dU):
        """
        Calculate the derivative of the cost function evaluated at X,U
        in the direction of a tangent trajectory dX,dU.
        """
        dcost = 0.0
        for k in range(len(X)-1):
            dcost += (dot(self.cost.l_dx(X[k], U[k], k), dX[k]) +
                      dot(self.cost.l_du(X[k], U[k], k), dU[k]))
        dcost += dot(self.cost.m_dx(X[-1]), dX[-1])
        return dcost


    def calc_ddcost(self, X, U, dX, dU, Q, R, S):
        """
        Calculate the second derivative of the cost function evaluated
        at X,U in the direction of a tangent trajectoyr dX and dU.
        The second order model parameters must be specified in Q,R,S.
        These can be obtained through DOptimizer.calc_newton_model()
        or by Doptimizer.calc_descent_direction() when method='newton'.
        """
        ddcost = 0.0
        for k in range(len(X) - 1):
            ddcost += (dot(dX[k], dot(Q(k), dX[k])) +
                     2*dot(dX[k], dot(S(k), dU[k])) +
                       dot(dU[k], dot(R(k), dU[k])))
        ddcost += dot(dot(dX[-1], Q(-1)), dX[-1])
        return ddcost
           

    def calc_steepest_model(self):
        """
        Calculate a quadratic model to find a steepest descent
        direction.  This is simply Q=I, R=I, S=0.
        """
        
        Q = np.eye(self.dsys.nX)
        R = np.eye(self.dsys.nU)
        S = np.zeros((self.dsys.nX, self.dsys.nU))
        return self.model_return(lambda k: Q, lambda k: R, lambda k: S)


    def calc_quasi_model(self, X, U):
        """
        Calculate a quadratic model to find a quasi-newton descent
        direction.  This takes into account the derivative of the cost
        function without considered system dynamics.
        """
        Q = [None]*len(X)
        S = [None]*(len(X)-1)
        R = [None]*(len(X)-1)

        Q[-1] = self.cost.m_dxdx(X[-1])
        for k in reversed(range(len(X)-1)):
            Q[k] = self.cost.l_dxdx(X[k], U[k], k)
            S[k] = self.cost.l_dxdu(X[k], U[k], k)
            R[k] = self.cost.l_dudu(X[k], U[k], k)
            
        return self.model_return(lambda k: Q[k], lambda k: R[k], lambda k: S[k])
    

    def calc_newton_model(self, X, U, A, B, K):
        """
        Calculate a quadratic model to find a newton descent
        direction.  This solves the backwards discrete adjoint
        equation.
        """
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
        
        return self.model_return(lambda k: Q[k], lambda k: R[k], lambda k: S[k])
        
    
    def calc_descent_direction(self, X, U, method='steepest'):
        """
        Calculate the descent direction from the trajectory X,U using
        the specified method.  Valid methods are:

        'steepest'
        'quasi'
        'newton'

        The method returns the tuple (Kproj, dX, dU, Q, R, S).
        """
        (Kproj, A, B) = self.dsys.calc_feedback_controller(X, U,
                                                           self.Qproj, self.Rproj,
                                                           True)

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
            
        return self.descent_return(Kproj, dX, dU, Q, R, S)


    def armijo_simulate(self, bX, bU, Kproj):
        """
        This is a sub-function for armijo search.  It projects the
        trajectory bX,bU to a real trajectory like DSystem.project,
        but it also returns a partial trajectory if the simulation
        fails.
        """
        # If still spending a lot of time in armijo search, move the
        # cost comparison to this loop so we can abort the simulation
        # as soon as possible.        
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
        return self.armijo_simulate_return(True, nX, nU)

        
    def armijo_search(self, X, U, Kproj, dX, dU):
        """
        Perform an Armijo line search from the trajectory X,U along
        the tangent trajectory dX, dU.  Returns the tuple (nX, nU,
        nCost).
        """
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
                return self.armijo_search_return(nX, nU, cost1)
        else:
            self.monitor.armijo_search_failure(X, U, dX, dU, cost0, dcost0, Kproj)
            raise trep.ConvergenceError("Armijo Failed to Converge")


    def step(self, iteration, X, U, method='steepest'):
        """
        Perform an optimization step.

        Find a new trajectory nX, nU that has a lower cost than the
        trajectory X,U.  Valid methods are defined in
        DOptimizer.calc_descent_direction().

        Returns the named tuple (done, nX, nU, dcost0, cost1) where:

        'done' is a boolean that is True if the trajectory X,U cannot
        be improved (i.e, X,U is a local minimizer)

        nX,nU are the improved trajectory

        dcost0 is the derivative of the cost at X,U

        cost1 is the cost of the improved trajectory.
        """
        self.monitor.step_begin(iteration)

        (Kproj, dX, dU, Q, R, S) = self.calc_descent_direction(X, U, method)
        
        cost0 = self.calc_cost(X, U)
        dcost0 = self.calc_dcost(X, U, dX, dU)

        self.monitor.step_info(method, cost0, dcost0, X, U, dX, dU, Kproj)

        # Check for sane descent direction
        if dcost0 > 0:
            fallback = self.select_fallback_method(iteration, method)
            self.monitor.step_method_failure(method, cost0, dcost0, fallback)
            return self.step(iteration, X, U, fallback)

        # Check for terminal condition
        if abs(dcost0) < self.descent_tolerance:
            self.monitor.step_termination(cost0, dcost0)
            return self.step_return(True, X, U, dcost0, cost0)

        # Line search in descent direction
        (X, U, cost1) = self.armijo_search(X, U, Kproj, dX, dU)

        self.monitor.step_completed(method, cost1, X, U)

        return self.step_return(False, X, U, dcost0, cost1)


    def select_method(self, iteration):
        """
        Select a descent direction method for the specified iteration.

        This is called by optimize() to choose a descent direction
        method for each step.  The default implementation takes
        'self.first_method_iterations' steps of 'self.first_method'
        and then switches to 'self.second_method' steps.
        """
        if iteration < self.first_method_iterations:
            method = self.first_method
        else:
            method = self.second_method
        return method


    def select_fallback_method(self, iteration, current_method):
        """
        When DOptimizer.step() finds a bad descent direction (e.g,
        positive cost derivative), it calls this method to figure out
        what descent direction it should use next.
        """
        if current_method == 'newton':
            return 'quasi'
        elif current_method == 'quasi':
            return 'steepest'
        else:
            # This should never occur
            raise StandardError("Derivative of cost is positive for steepest descent.")

            
    def optimize(self, X, U, max_steps=50):
        """
        Iteratively optimize the trajectory X,U.

        This function calls DOptimizer.step() until a local minimizer
        is found or 'max_steps' iterations were taken.

        Returns the named tuple (converged, X, U) where:

        converged is a boolean indicating if the optimization finished
        on a local minimizer.

        X,U is the improved trajectory.
        """
        X = np.array(X)
        U = np.array(U)

        self.monitor.optimize_begin(X, U)

        for i in range(max_steps):
            method = self.select_method(i)
            (converged, X, U, cost, method) = self.step(i, X, U, method)
            if converged:
                break

        self.monitor.optimize_end(converged, X, U, cost)
        return self.optimize_return(converged, X, U)


    def descent_plot(self, X, U, method='steepest', points=40, legend=True):
        """
        Create a descent direction plot at X,U for the specified method.
        """
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


        pyplot.hold(True)
        pyplot.plot(z, model_cost-cost, '-,', linewidth=2.0, color='blue', label='Modeled Cost')
        pyplot.plot(z, true_cost-cost, '.-', linewidth=1.0, color='black', label='True Cost')
        pyplot.plot(armijo_z, armijo_cost-cost, 'o', color='gray', label='Armijo Evaluations')
        pyplot.plot(z, armijo_max-cost, '-.', color='black', label='Required Cost Improvement')
        pyplot.hold(False)
        if legend:
            pyplot.legend(loc=0)

        pyplot.title('Cost along descent direction for method: "%s".' % method)
        pyplot.xlabel('z')
        pyplot.ylabel(r'$\Delta$ cost')


    def check_dcost(self, X, U, method='steepest', delta=1e-6, tolerance=1e-5):
        """
        Check the calculated derivative of the cost function at X,U
        with a numeric approximation determined from the original cost
        function.
        """
        (Kproj, dX, dU, Q, R, S) = self.calc_descent_direction(X, U, method)
        exact_dcost = self.calc_dcost(X, U, dX, dU)

        nX, nU = self.dsys.project(X - delta*dX, U - delta*dU, Kproj)
        cost0 = self.calc_cost(nX, nU)

        nX, nU = self.dsys.project(X + delta*dX, U + delta*dU, Kproj)
        cost1 = self.calc_cost(nX, nU)
                
        approx_dcost = (cost1 - cost0)/(2*delta)
        error = approx_dcost - exact_dcost
        result = (abs(error) <= tolerance)
        return self.check_dcost_return(result, error, cost1, cost0, approx_dcost, exact_dcost)
    

    def check_ddcost(self, X, U, method='steepest', delta=1e-6, tolerance=1e-5):
        """
        Check the second derivative of the cost function at X,U with a
        numeric approximation determined from the first derivative.
        """
        (Kproj, dX, dU, Q, R, S) = self.calc_descent_direction(X, U, method)
        if method != 'newton':
            (Q, R, S) = self.calc_descent_direction(X, U, 'newton')[-3:]

        exact_ddcost = self.calc_ddcost(X, U, dX, dU, Q, R, S)

        # Calculate cost0
        bX = X - delta*dX
        bU = U - delta*dU
        nX, nU = self.dsys.project(bX, bU, Kproj)
        (A, B) = self.dsys.linearize_trajectory(nX, nU)
        (ndX, ndU) = self.dsys.dproject(A, B, dX, dU, Kproj)
        dcost0 = self.calc_dcost(nX, nU, ndX, ndU)

        # Calculate cost1
        bX = X + delta*dX
        bU = U + delta*dU
        nX, nU = self.dsys.project(bX, bU, Kproj)
        (A, B) = self.dsys.linearize_trajectory(nX, nU)
        (ndX, ndU) = self.dsys.dproject(A, B, dX, dU, Kproj)
        dcost1 = self.calc_dcost(nX, nU, ndX, ndU)
        
        approx_ddcost = (dcost1 - dcost0)/(2*delta)
        error = approx_ddcost - exact_ddcost
        result = (abs(error) <= tolerance)
        
        return self.check_ddcost_return(result, error, dcost1, dcost0, approx_ddcost, exact_ddcost)

        
            
