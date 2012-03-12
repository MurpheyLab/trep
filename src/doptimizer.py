import datetime
import numpy as np
import dlqr

from numpy import dot


# You can use the monitor to get feedback during the optimization.
# Inherit from this class and overload the functions you are
# interested in.
class DOptimizerMonitor(object):
    def initialized(self, optimizer, cost, xi): pass
    def calculated_descent_direction(self, optimizer, iteration, method, dxi, dcost): pass
    def step_finished(self, optimizer, iteration, cost): pass
    


class DOptimizer(object):
#class TrajectoryOptimization(object):
    def __init__(self, dsys, cost, monitor=None):
        self.dsys = dsys
        self.cost = cost

        if monitor:
            self.monitor = monitor
        else:
            # Install default monitor
            self.monitor = DOptimizerMonitor()            

        # Default weights used to generate feedback controller for the
        # projection.  These can be changed if desired.
        Qproj = np.eye(self.dsys.nX)
        Rproj = np.eye(self.dsys.nU)
        self.Qproj = lambda k: Qproj
        self.Rproj = lambda k: Rproj

        self.xi = None #  Current trajectory as tuple (X,U)
        self.cost_n = 0.0 # Cost of the Current Trajectory
        self.step_method = "N/A"  # Method used to generate last descent direction
        self.prev_step_method = "N/A" # Method used for previous descent

        self.dxi = None   # Current optimal decent as tuple (dX, dU)
        self.dcost = 0.0  # Current optimal dcost

        self.armijo_beta = 0.7
        self.armijo_alpha = 0.00001
        self.armijo_max_iterations = 50
        self.armijo_prev_m = 0

        self.descent_tolerance = 1e-6

        # Number of first order iterations to do at the start of an optimization
        self.first_order_iterations = 10
        # Number of first order iterations to do after a second order iteration fails
        self.first_order_fallbacks = 5

        # Number of steepest-descent iterations left before a newton's
        # method iteration is attempted
        self.first_order_left = 0
        self.iterations = 0



    def initialize(self, xi0):
        # Initializes the optimzer by calculating the cost of the
        # inital trajectory and resetting state variables.

        (X,U) = xi0
        X = np.array(X)
        U = np.array(U)
        self.xi = (X,U)
        self.dxi = (0*X, 0*U)        
        self.cost_n = self.calc_cost(self.xi)
        
        self.step_method = "N/A" 
        self.prev_step_method = "N/A"
        self.first_order_left = self.first_order_iterations
        self.dcost = 0.0  
        self.armijo_prev_m = 0

        self.monitor.initialized(self, self.xi, self.cost_n)


    def get_1st_order_descent_direction(self, fixed_ic):
        dsys = self.dsys          # for convenience
        (X, U) = self.xi
        (A, B) = self.calc_linearization(self.xi)

        # Generate feedback controller for projection
        self.Kproj = dlqr.solve_tv_lqr(A, B, self.Qproj, self.Rproj)[0]

        # Generate cost weights for LQ problem
        q = [None]*(dsys.kf()+1)
        r = [None]*dsys.kf()
        for k in range(dsys.kf()):
            q[k] = self.cost.l_dx(X[k], U[k], k)
            r[k] = self.cost.l_du(X[k], U[k], k)
        q[dsys.kf()] = self.cost.m_dx(X[dsys.kf()])

        # First order optimization
        Q = np.eye(dsys.nX)
        R = np.eye(dsys.nU)
        S = np.zeros((dsys.nX, dsys.nU))

        self.calc_dxi(A, B, q, r,
                      lambda k: Q,
                      lambda k: S,
                      lambda k: R,
                      fixed_ic)


    def get_2nd_order_descent_direction(self, fixed_ic):
        dsys = self.dsys          # for convenience
        (X, U) = self.xi
        (A, B) = self.calc_linearization(self.xi)
        kf = dsys.kf()

        # Generate feedback controller for projection
        self.Kproj = dlqr.solve_tv_lqr(A, B, self.Qproj, self.Rproj)[0]

        # Generate cost weights for LQ problem
        q = [None]*(dsys.kf()+1)
        r = [None]*dsys.kf()
        for k in range(dsys.kf()):
            q[k] = self.cost.l_dx(X[k], U[k], k)
            r[k] = self.cost.l_du(X[k], U[k], k)
        q[dsys.kf()] = self.cost.m_dx(X[dsys.kf()])

        # Second order optimization
        Q = [None]*(dsys.kf()+1)
        S = [None]*dsys.kf()
        R = [None]*dsys.kf()

        zT = self.cost.m_dx(X[kf])
        Q[kf] = self.cost.m_dxdx(X[kf])
        for k in reversed(range(kf)):
            Q[k] = self.cost.l_dxdx(X[k], U[k], k)
            S[k] = self.cost.l_dxdu(X[k], U[k], k)
            R[k] = self.cost.l_dudu(X[k], U[k], k)
            self.dsys.set(X[k], U[k], k)
            Q[k] += self.dsys.fdxdx(zT)
            S[k] += self.dsys.fdxdu(zT)
            R[k] += self.dsys.fdudu(zT)

            zT = (self.cost.l_dx(X[k], U[k], k) -
                  dot(self.cost.l_du(X[k], U[k], k), self.Kproj[k]) +
                  dot(zT, (A[k] - dot(B[k], self.Kproj[k]))))

        self.calc_dxi(A, B, q, r,
                      lambda k: Q[k],
                      lambda k: S[k],
                      lambda k: R[k],
                      fixed_ic)


    def calc_descent_direction(self, method='steepest', fixed_ic=True):
        if method == 'newton':            
            self.get_2nd_order_descent_direction(fixed_ic)
            self.dcost = self.calc_dcost(self.xi, self.dxi)
            if self.dcost > self.descent_tolerance:
                self.first_order_left += max(0, (self.first_order_fallbacks - 1))
                return self.calc_descent_direction('steepest (fallback)', fixed_ic)
        elif method.startswith('steepest'):
            self.get_1st_order_descent_direction(fixed_ic)
            self.dcost = self.calc_dcost(self.xi, self.dxi)
            self.first_order_left = max(0, self.first_order_left - 1)
        else:
            raise StandardError("Invalid descent direction method: %r" % method)

        # We keep track of the previous step method for an
        # optimization in the armijo algorithm.
        self.prev_step_method = self.step_method
        self.step_method = method

        # Check for sane descent directoin
        if self.dcost > 0:
            raise StandardError("dcost is positive?")

        self.monitor.calculated_descent_direction(self, self.iterations,
                                                  self.step_method, self.dxi, self.dcost)        
        return self.dcost
    

    def step(self, method='steepest', fixed_ic=True):
        # Perform an optimization step.  Valid methods are 'newton'
        # and 'steepest'.  Returns true if the current trajectory
        # satisfies the first-order optimality condition.  Otherwise
        # improves the trajectory and returns false.

        self.calc_descent_direction(method)

        # Check for terminal condition
        if abs(self.dcost) < self.descent_tolerance:
            return True

        # Line search in descent direction
        (xi_n, cost_n) = self.armijo_search(self.xi, self.cost_n, self.dxi, self.dcost)

        # Save the new trajectory and cost
        self.xi = xi_n
        self.cost_n = cost_n
        self.iterations += 1
        return False
                        

    def calc_linearization(self, xi):
        (X,U) = xi
        A = []
        B = []
        for k in range(self.dsys.kf()):
            if k == 0:
                self.dsys.set(X[0], U[0], 0)
            else:
                self.dsys.step(U[k])
            A.append(self.dsys.fdx())
            B.append(self.dsys.fdu())
        return (A,B)
                        

    def armijo_search(self, xi, cost, dxi, dcost):

        # This is a little optimization on the premise that if we took
        # N armijo steps for the last iteration, we will probably have
        # to take about the same number this time, so we can save time
        # by starting at m = N-C where C is some constant (in this
        # case 2).  If we changed optimization methods, there is
        # probably no relationship between the previous step, so
        # always start at 0 in that case.        
        if self.step_method != self.prev_step_method:
            m0 = 0
        else:
            m0 = max(0, self.armijo_prev_m-2)
        

        for m in range(m0, self.armijo_max_iterations):
            lam = self.armijo_beta**m
            try: 
                xi_n = self.project(self.add_dxi(xi, lam, dxi), m)
            except StandardError, e:
                #raise e
                print "armijo: simulation failed at m=%d, continuing" % m
                #raise 
                continue
            cost_n = self.calc_cost(xi_n)
            #print "cost_n: ", cost_n
            if cost_n < cost + self.armijo_alpha* lam * dcost:
                self.armijo_prev_m = m
                #print "Descent succeeds with m = %d" % m
                return (xi_n, cost_n)
        raise StandardError("Armijo Failed to Converge")

    def add_dxi(self, xi, lam, dxi):
        X,U = xi
        dX,dU = dxi
        return (X + lam*dX, U + lam*dU)

    def project(self, bxi, m):
        (bX, bU) = bxi
        
        (X,U) = self.dsys.build_trajectory()
        X[0] = bX[0]

        try:
            for k in range(self.dsys.kf()):
                U[k] = bU[k] - dot(self.Kproj[k], X[k] - bX[k])
                if k == 0:
                    self.dsys.set(X[k], U[k], k)
                else:
                    self.dsys.step(U[k])
                X[k+1] = self.dsys.f()
        except:
            print "failing at k = ", k
            raise
        finally:
            pass
            
        return (X,U)


    def calc_cost(self, xi):        
        cost = 0.0
        for k in range(self.dsys.kf()):
            cost += self.cost.l(xi[0][k], xi[1][k], k)
        cost += self.cost.m(xi[0][self.dsys.kf()])
        return cost

    def calc_dcost(self, xi, dxi):        
        dcost = 0.0
        X,U = xi
        dX,dU = dxi
        for k in range(self.dsys.kf()):
            dcost += (dot(self.cost.l_dx(X[k], U[k], k), dX[k]) +
                      dot(self.cost.l_du(X[k], U[k], k), dU[k]))
        dcost += dot(self.cost.m_dx(X[self.dsys.kf()]), dX[self.dsys.kf()])
        return dcost

    def calc_ddcost(self, xi, dxi):
        (X, U) = xi    # For convenience
        (dX, dU) = dxi
        kf = self.dsys.kf()

        ddcost = 0.0
        for k in range(self.dsys.kf()):
            ddcost += (dot(dot(dX[k], self.cost.l_dxdx(X[k], U[k], k), dX[k])) +
                     2*dot(dot(dX[k], self.cost.l_dxdu(X[k], U[k], k), dU[k])) +
                       dot(dot(dU[k], self.cost.l_dudu(X[k], U[k], k), dU[k])))
        ddcost += dot(dot(dX[kf], self.cost.m_dxdx(X[kf]), dX[kf]))
        return ddcos

    def calc_dxi(self, A, B, q, r, Q, S, R, fixed_ic):
        (K,C,P,b) = dlqr.solve_tv_lq(A, B, q, r, Q, S, R)

        if not fixed_ic:
            dx0 = -np.linalg.solve(P, b)
        else:
            dx0 = np.zeros((self.dsys.nX,))

        dX,dU = self.dxi
        dX[0] = dx0
        for k in xrange(self.dsys.kf()):
            dU[k] = -dot(K[k],dX[k]) - C[k] 
            dX[k+1] = dot(A[k],dX[k]) + dot(B[k],dU[k]) 
        self.dxi = (dX,dU)

            
    def optimize(self, xi0, max_steps, fixed_ic=True):
        self.initialize(xi0)
        print "initial cost: %f" % self.cost_n

        for i in range(max_steps):
            if self.first_order_left > 0:
                method = 'steepest'
            else:
                method = 'newton'
            # If we tried a newton model last time but the armijo line
            # search had to drastically reduce the step-size to see
            # any improvement, go back to steepest descent for a
            # while.
            if self.step_method == 'newton' and self.armijo_prev_m > 8:
                method = 'steepest (newton was bad model)'
                self.first_order_left += max(1, self.first_order_fallbacks)

            result = self.step(method, fixed_ic)
            self.monitor.step_finished(self, self.iterations, self.cost_n)

            if result:
                print "Finished at cost: ", self.cost_n
                return self.xi
            now = datetime.datetime.now().strftime('%H:%M:%S')
            print "[%s] %4d: method=%r cost=%f    dcost=%0.8f   ddcost=%f  %2d armijo iterations"  % (
                now, i, self.step_method, self.cost_n, self.dcost, 0, self.armijo_prev_m)

        return self.xi
    its = 0


