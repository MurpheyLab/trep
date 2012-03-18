import datetime
import numpy as np
import dlqr

import numpy.linalg
from numpy import dot

try:
    import matplotlib.pyplot as pyplot
    pyplot_available = True
except ImportError:
    pyplot_available = False


# You can use the monitor to get feedback during the optimization.
# Inherit from this class and overload the functions you are
# interested in.
## class DOptimizerMonitor(object):
##     pass


class DOptimizer(object):
    def __init__(self, dsys, cost):
        self.dsys = dsys
        self.cost = cost
        self.optimize_ic = False

        # Default weights used to generate feedback controller for the
        # projection.  These can be changed if desired.
        Qproj = np.eye(self.dsys.nX)
        Rproj = np.eye(self.dsys.nU)
        self.Qproj = lambda k: Qproj
        self.Rproj = lambda k: Rproj

        self.step_method = "N/A"  # Method used to generate last descent direction

        self.armijo_beta = 0.7
        self.armijo_alpha = 0.00001
        self.armijo_max_iterations = 30
        self.armijo_prev_m = 0

        self.descent_tolerance = 1e-6

        # Number of first order iterations to do at the start of an optimization
        self.first_order_iterations = 10
        # Number of first order iterations to do after a second order iteration fails
        self.first_order_fallbacks = 5

        # Number of steepest-descent iterations left before a newton's
        # method iteration is attempted
        self.first_order_left = 0


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
            self.dsys.set(X[k], U[k], k)
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
                      

    def armijo_search(self, X, U, Kproj, dX, dU):

        m0 = 0
        cost = self.calc_cost(X, U)
        dcost = self.calc_dcost(X, U, dX, dU)
        
        for m in range(m0, self.armijo_max_iterations):
            lam = self.armijo_beta**m
            max_cost = cost + self.armijo_alpha* lam * dcost

            bX = X + lam*dX
            bU = U + lam*dU

            nX = np.zeros(bX.shape)
            nU = np.zeros(bU.shape)
            try:
                
                nX[0] = bX[0]
                for k in range(len(X)-1):
                    nU[k] = bU[k] - dot(Kproj[k], nX[k] - bX[k])
                    if k == 0:
                        self.dsys.set(nX[k], nU[k], k)
                    else:
                        self.dsys.step(nU[k])
                    nX[k+1] = self.dsys.f()
                    
            except StandardError:
                print "armijo: simulation failed at m=%d, continuing" % m
                continue
            cost_n = self.calc_cost(nX, nU)
            
            if cost_n < max_cost:
                self.armijo_prev_m = m
                return (nX, nU, m)
        else:
            raise StandardError("Armijo Failed to Converge")


    def step(self, X, U, method='steepest'):

        (Kproj, dX, dU, Q, R, S) = self.calc_descent_direction(X, U, method)

        self.step_method = method

        # Check for sane descent direction
        dcost = self.calc_dcost(X, U, dX, dU)
        if dcost > 0:
            if method != 'steepest':
                # Fallback to steepest descent step
                print "fallback (%f)" % dcost
                #self.first_order_left += max(0, (self.first_order_fallbacks - 1))
                return self.step(X, U, 'steepest')
            else:
                # This should never occur
                raise StandardError("Derivative of cost is positive for steepest descent.")

        # Check for terminal condition
        if abs(dcost) < self.descent_tolerance:
            return (True, X, U, dX, dU, dcost, None)

        # Line search in descent direction
        (X, U, armijo_iterations) = self.armijo_search(X, U, Kproj, dX, dU)

        return (False, X, U, dX, dU, dcost, armijo_iterations)


    def calc_projection(self, X, U, return_linearization=False):
        (A, B) = self.dsys.linearize_trajectory(X, U)    
        Kproj = dlqr.solve_tv_lqr(A, B, self.Qproj, self.Rproj)[0]
        if return_linearization:
            return (Kproj, A, B)
        else:
            return Kproj

 
    def project(self, bX, bU, Kproj):
        nX = np.zeros(bX.shape)
        nU = np.zeros(bU.shape)

        nX[0] = bX[0]
        for k in range(len(bX)-1):
            nU[k] = bU[k] - dot(Kproj[k], nX[k] - bX[k])
            if k == 0:
                self.dsys.set(nX[k], nU[k], k)
            else:
                self.dsys.step(nU[k])
            nX[k+1] = self.dsys.f()
            
        return nX, nU


    def dproject(self, A, B, bdX, bdU, K):
        dX = np.zeros(bdX.shape)
        dU = np.zeros(bdU.shape)
        dX[0] = bdX[0]
        for k in xrange(len(bdX)-1):
            dU[k] = bdU[k] - dot(K[k], dX[k] - bdX[k])
            dX[k+1] = dot(A[k],dX[k]) + dot(B[k],dU[k])
        return dX, dU

            
    def optimize(self, X, U, max_steps=50):

        X = np.array(X)
        U = np.array(U)

        self.step_method = "N/A" 
        self.first_order_left = self.first_order_iterations
        self.armijo_prev_m = 0
        print "initial cost: %f" % self.calc_cost(X, U)

        for i in range(max_steps):
            if self.first_order_left > 0:
                method = 'quasi'
                self.first_order_left = max(0, self.first_order_left - 1)
            else:
                method = 'newton'

            (converged, X, U, dX, dU, dcost, armijo_iterations) = self.step(X, U, method)

            cost = self.calc_cost(X, U)

            if converged:
                print "Finished at cost: ", cost
                return (True, X, U)
            now = datetime.datetime.now().strftime('%H:%M:%S')
            print "[%s] %4d: method=%r cost=%f    dcost=%0.8f   ddcost=%f  %2d armijo iterations"  % (
                now, i, self.step_method, cost, dcost, 0, self.armijo_prev_m)

        return (False, X, U)


    def descent_plot(self, X, U, method='steepest', points=40, legend=True):

        if not pyplot_available:
            raise StandardError("Importing matplotlib failed. Cannot create plot.")

        (Kproj, dX, dU, Q, R, S) = self.calc_descent_direction(X, U, method)

        def calc_cost(z):
            return self.calc_cost(*self.project(X + zi*dX, U + zi*dU, Kproj))

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
            armijo_cost[i] = self.calc_cost(*self.project(X + zi*dX,
                                                          U + zi*dU,
                                                          Kproj))

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

        nX, nU = self.project(X - delta*dX, U - delta*dU, Kproj)
        cost0 = self.calc_cost(nX, nU)

        nX, nU = self.project(X + delta*dX, U + delta*dU, Kproj)
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
        nX, nU = self.project(bX, bU, Kproj)
        (A, B) = self.dsys.linearize_trajectory(nX, nU)
        (ndX, ndU) = self.dproject(A, B, dX, dU, Kproj)
        dcost0 = self.calc_dcost(nX, nU, ndX, ndU)

        # Calculate cost1
        bX = X + delta*dX
        bU = U + delta*dU
        nX, nU = self.project(bX, bU, Kproj)
        (A, B) = self.dsys.linearize_trajectory(nX, nU)
        (ndX, ndU) = self.dproject(A, B, dX, dU, Kproj)
        dcost1 = self.calc_dcost(nX, nU, ndX, ndU)
        
        approx_ddcost = (dcost1 - dcost0)/(2*delta)
        error = approx_ddcost - exact_ddcost
        result = (abs(error) <= tolerance)
        
        return (result, error, dcost1, dcost0, approx_ddcost, exact_ddcost)

        
            
