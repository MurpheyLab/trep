.. _trep_doptimizer:

:class:`DOptimizer` - Discrete Trajectory Optimization
======================================================

.. currentmodule:: trep.discopt

.. contents::
   :local:

Discrete trajectory optimization is performed with :class:`DOptimizer`
objects.  The optimizer finds a trajectory for a :class:`DSystem` that
minimizes a :class:`DCost`. 

The optimizer should work for arbitrary systems, not just ones based
on variational integrators.  You would need to define a compatible
:class:`DSystem` class that supports the right functionality.


.. admonition:: Examples

   ``pend-on-cart-optimization.py``


DOptimizer Objects
------------------

.. class:: DOptimizer(dsys, cost, first_method_iterations=10, monitor=None)

   You should create a new optimizer for each new system or cost, but
   for a given combination you can optimize as many different
   trajectories as you want.  The optimization is designed to mostly
   be used through the :meth:`optimize` method.  

   You can use the :class:`DOptimizerMonitor` class to monitor
   progress during optimizations.  If *monitor* is :data:`None`, the
   optimizer will use :class:`DOptimizerDefaultMonitor` which prints
   useful information to the console.

           
.. attribute:: DOptimizer.dsys

   The :class:`DSystem` that is optimized by this instance.


.. attribute:: DOptimizer.cost = cost

   The :class:`DCost` that is optimized by this instance.


.. attribute:: DOptimizer.optimize_ic 

   This is a Boolean indicating whether or not the optimizer should
   optimize the initial condition during an optimization.  

   Defaults to :data:`False`.

   .. warning:: This is broken for constrained systems.


.. attribute:: DOptimizer.monitor

   The :class:`DOptimizerMonitor` that this optimization reports
   progress to.  The default is a new instance of
   :class:`DOptimizerDefaultMonitor`.  


.. attribute:: DOptimizer.Qproj
               DOptimizer.Rproj

   These are the weights for an LQR problem used to generate a linear
   feedback controller for each trajectory.  Each should be a function
   of k that returns an appropriately sized 2D :class:`ndarray`.

   The default values are identity matrices.

             
.. attribute:: DOptimizer.descent_tolerance

   A trajectory is considered a local minimizer if the norm of the
   cost derivative is less than this. The default value is 1e-6.



Cost Functions
^^^^^^^^^^^^^^

.. method:: DOptimizer.calc_cost(X, U)

   Calculate the cost of a trajectory *X,U*.


.. method:: DOptimizer.calc_dcost(X, U, dX, dU)
        
   Calculate the derivative of the cost function evaluated at *X,U* in
   the direction of a tangent trajectory *dX,dU*\ .
        
   It is important that *dX,dU* be an actual tangent trajectory of the
   system at *X,U* to get the correct cost.  See :meth:`check_ddcost`
   for an example where this is important.


.. method:: DOptimizer.calc_ddcost(X, U, dX, dU, Q, R, S)
        
   Calculate the second derivative of the cost function evaluated at
   *X,U* in the direction of a tangent trajectory *dX,dU*\ .  The
   second order model parameters must be specified in *Q,R,S*\ .  These
   can be obtained through :meth:`calc_newton_model` or by
   :meth:`calc_descent_direction` when *method*\ ="newton".

   It is important that *dX,dU* be an actual tangent trajectory of the
   system at *X,U* to get the correct cost.  See :meth:`check_ddcost`
   for an example where this is important.
        

Descent Directions
^^^^^^^^^^^^^^^^^^           

.. method:: DOptimizer.calc_steepest_model()
        
   Calculate a quadratic model to find a steepest descent direction:

   .. math::

      Q = \mathcal{I} \quad R = \mathcal{I} \quad S = 0

        
.. method:: DOptimizer.calc_quasi_model(X, U)
        
   Calculate a quadratic model to find a quasi-newton descent
   direction.  This uses the second derivative of the un-projected cost
   function.

   .. math::

      Q = \derivII[h]{x}{x}  \quad R = \derivII[h]{u}{u} \quad S = \derivII[h]{x}{u}

   This method does not use the second derivative of the system
   dynamics, so it tends to be as fast :meth:`calc_steepest_model`,
   but usually converges much faster.
        
.. method:: DOptimizer.calc_newton_model(X, U, A, B, K)
        
   Calculate a quadratic model to find a newton descent direction.
   This is the true second derivative of the projected cost function:
   
   .. math::

      \begin{align*}
      Q(k_f) &= D^2m(x(k_f))
      \\
      Q(k) &= \derivII[\ell]{x}{x}(k) + z^T(k+1) \derivII[f]{x}{x}(k) 
      \\
      S(k) &= \derivII[\ell]{x}{u}(k) + z^T(k+1) \derivII[f]{x}{u}(k) 
      \\
      R(k) &= \derivII[\ell]{u}{u}(k) + z^T(k+1) \derivII[f]{u}{u}(k) 
      \end{align*}

   where:

   .. math::

      \begin{align*}
      z(k_f) &= Dm^T(x(k_f))
      \\
      z(k) &= \deriv[\ell]{x}^T(k) - \mathcal{K}^T(k) \deriv[\ell]{u}^T(k) +
              \left[ \deriv[f]{x}^T(k) - \mathcal{K}^T(k) \deriv[f]{u}^T(i)  \right] z(k+1)
              \end{align*}

   This method depends on the second derivative of the system's
   dynamics, so it can be significantly slower than other step methods.
   However, it converges extremely quickly near the minimizer.
        

.. method:: DOptimizer.calc_descent_direction(X, U, method='steepest')
        
   Calculate the descent direction from the trajectory *X,U* using the
   specified method.  Valid methods are:

      * "steepest" - Use :meth:`calc_steepest_model`

      * "quasi" - Use :meth:`calc_quasi_model`

      * "newton" - Use :meth:`calc_newton_model`

   The method returns the named tuple ``(Kproj, dX, dU, Q, R, S)``.


Armijo Line Search
^^^^^^^^^^^^^^^^^^
        
.. attribute:: DOptimizer.armijo_beta
               DOptimizer.armijo_alpha
               DOptimizer.armijo_max_iterations

   Parameters for the Armijo line search performed at each step along
   the calculated descent direction.  

   *armijo_beta* should be between 0 and 1 (not inclusive).  The
   default value is 0.7.

   *armijo_alpha* should be between 0 (inclusive) and 1 (not
   inclusive). The default value is 0.00001.

   *armijo_max_iterations* should be a positive integer.  If the line
   search cannot satisfy the sufficient decrease criteria after this
   number of iterations, a :exc:`trep.ConvergenceError` is raised.
   The default value is 30.


.. method:: DOptimizer.armijo_simulate(bX, bU, Kproj)
        
   This is a sub-function for armijo search.  It projects the
   trajectory bX,bU to a real trajectory like DSystem.project, but it
   also returns a partial trajectory if the simulation fails.  It is
   not intended to be used directly.
        

.. method:: DOptimizer.armijo_search(X, U, Kproj, dX, dU)
            
   Perform an Armijo line search from the trajectory X,U along the
   tangent trajectory dX, dU.  Returns the named tuple ``(nX, nU, nCost)``
   or raises :exc:`trep.ConvergenceError` if the search doesn't
   terminate before taking the maximum number of iterations.

   This method is used by :meth:`step` once a descent direction has
   been found.
        

Optimizing a Trajectory
^^^^^^^^^^^^^^^^^^^^^^^

.. method:: DOptimizer.step(iteration, X, U, method='steepest')
        
   Perform an optimization step using a particular method.  

   This finds a new trajectory nX, nU that has a lower cost than the
   trajectory X,U.  Valid methods are defined in
   DOptimizer.calc_descent_direction().

   If the specified method fails to find an acceptable descent
   direction, :meth:`step` will try again with the method returned
   by :meth:`select_fallback_method`.

   *iteration* is an integer that is used by
   :meth:`select_fallback_method` and passed to the
   :class:`DOptimizerMonitor` when reporting the current step
   progress.
   
   Returns the named tuple ``(done, nX, nU, dcost0, cost1)`` where:

      * *done* is a Boolean that is :data:`True` if the trajectory
        *X,U* cannot be improved (i.e, *X,U* is a local minimizer of
        the cost).
   
      * *nX,nU* are the improved trajectory
   
      * *dcost0* is the derivative of the cost at *X,U*.
   
      * *cost1* is the cost of the improved trajectory.
        

.. method:: DOptimizer.optimize(X, U, max_steps=50)
        
   Iteratively optimize the trajectory *X,U* until a local minimizer
   is found or *max_steps* are taken.  The descent direction method
   used at each step is determined by :meth:`select_method`.

   Returns the named tuple ``(converged, X, U)`` where:

      * *converged* is a Boolean indicating if the optimization
        finished on a local minimizer.

      * *X,U* is the improved trajectory.


.. attribute:: DOptimizer.first_method_iterations 

   Number of steps to take using :attr:`first_method` before switching
   to :attr:`second_method` for the remaining steps.  See
   :meth:`select_method` for more information on controlling the step
   method.

   *Default: 10*

.. attribute:: DOptimizer.first_method        
               
   Descent method to use for the first iterations of the optimization.

   *Default: "quasi"*

.. attribute:: DOptimizer.second_method

   Descent method to use for the optimzation after
   :attr:`first_method_iterations` iterations have been taken.

   *Default: "netwon"*

.. method:: DOptimizer.select_method(iteration)
        
   Select a descent direction method for the specified iteration.

   This is called by :meth:`optimize` to choose a descent direction
   method for each step.  The default implementation takes a
   pre-determined number (:attr:`lower_order_iterations`) of "quasi"
   steps and then switches to the "newton" method.  

   You can customize the method selection by inheriting
   :class:`DOptimizer` and overriding this method.
  

.. method:: DOptimizer.select_fallback_method(iteration, current_method)
        
   When :meth:`step` finds a bad descent direction (e.g, positive cost
   derivative), this method is called to figure out what descent
   direction it should try next.



Debugging Tools
^^^^^^^^^^^^^^^        

.. method:: DOptimizer.descent_plot(X, U, method='steepest', points=40, legend=True)
        
   Create a descent direction plot at X,U for the specified method.  

   This is a useful plot for examining the line search portion of an
   optimization step.  It plots several values along the descent
   direction line.  All values are plotted against the line search
   parameter :math:`z \in \mathbb{R}`:

     * The true cost :math:`g(\xi + z\delta\xi) - g(\xi)`

     * The modeled cost :math:`Dg(\xi)\op\delta\xi z + \half q\op(\delta\xi, \delta\xi)z^2`

     * The sufficient decrease line: :math:`g(\xi + z \delta\xi) < g(\xi) + \alpha z Dg(\xi)\op\delta\xi`

     * The armijo evaluation points: :math:`z = \beta^m`

   This is an example plot for a steepest descent plot during a
   particular optimization.  This plot shows that the true cost
   increases much faster than the model predicts.  As a result, 6
   Armijo steps are required before the sufficient decrease condition
   is satisfied.

   .. image:: descent-direction-plot-example.png
      
   Example usage::

     >>> import matplotlib.pyplot as pyplot
     >>> optimizer.descent_plot(X, U, method='steepest', legend=True)
     >>> pyplot.show() 


.. method:: check_dcost(X, U, method='steepest', delta=1e-6, tolerance=1e-5)
        
   Check the calculated derivative of the cost function at X,U with a
   numeric approximation determined from the original cost function.
        

.. method:: DOptimizer.check_ddcost(X, U, method='steepest', delta=1e-6, tolerance=1e-5)
        
   Check the second derivative of the cost function at X,U with a
   numeric approximation determined from the first derivative.
        




DOptimizerMonitor Objects
-------------------------

:class:`DOptimizer` objects report optimization progress to their
:attr:`monitor` object.  The base implementation does nothing.  The
default monitor :class:`DOptimizerDefaultMonitor` mainly prints

reports to the console.  You can define your own monitor to gather
more detailed information like saving each intermediate trajectory.

Note that if you do want to save any values, you should save copies.
The optimizer might reuse the same variables in each step to optimize
memory usage.


.. class:: DOptimizerMonitor

   This is the base class for Optimizer Monitors.  It does absolutely
   nothing, so you can use this as your monitor if you want
   completely silent operation.

    
.. method:: DOptimizerMonitor.optimize_begin(X, U)
        
   Called when DOptimizer.optimize() is called with the initial
   trajectory.
        
            
.. method:: DOptimizerMonitor.optimize_end(converged, X, U, cost)
        
   Called before DOptimizer.optimize() returns with the results of the
   optimization.
        
        
.. method:: DOptimizerMonitor.step_begin(iteration)
        
   Called at the start of each DOptimize.step().  Note that step calls
   itself with the new method when one method fails, so this might be
   called multiple times with the same iteration.

   All calls will be related to the same iteration until
   step_termination or step_completed are called.
        
        
.. method:: DOptimizerMonitor.step_info(method, cost, dcost, X, U, dX, dU, Kproj)
        
   Called after a descent direction has been calculated.
        
        
.. method:: DOptimizerMonitor.step_method_failure(method, cost, dcost, fallback_method)
        
   Called when a descent method results in a positive cost derivative.
        
        
.. method:: DOptimizerMonitor.step_termination(cost, dcost)
        
   Called if dcost satisfies the descent tolerance, indicating that
   the current trajectory is a local minimizer.
        
        
.. method:: DOptimizerMonitor.step_completed(method, cost, nX, nU)
        
   Called at the end of an optimization step with information about
   the new trajectory.

        
.. method:: DOptimizerMonitor.armijo_simulation_failure(armijo_iteration, nX, nU, bX, bU)
        
   Called when a simulation fails (usually an instability) during the
   evaluation of the cost in an armijo step.  The Armijo search
   continues after this.
        
            
.. method:: DOptimizerMonitor.armijo_search_failure(X, U, dX, dU, cost0, dcost0, Kproj)
        
   Called when the Armijo search reaches the maximum number of
   iterations without satisfying the sufficient decrease criteria.
   The optimization cannot proceed after this.
        
        
.. method:: DOptimizerMonitor.armijo_evaluation(armijo_iteration, nX, nU, bX, bU, cost, max_cost)
        
   Called after each Armijo evaluation.  The semi-trajectory bX,bU was
   successfully projected into the new trajectory nX,nU and its cost
   was measured.  The search will continue if the cost is greater than
   the maximum cost.
        


DOptimizerDefaultMonitor Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the default monitor for :class:`DOptimizer`.  It prints out
information to the console and records the cost and cost derivative
history. 


.. class:: DOptimizerDefaultMonitor

    This is the default DOptimizer Monitor.  It mainly prints status
    updates to stdout and records the cost and dcost history so you
    can create convergence plots.


.. attribute:: DOptimizerDefaultMonitor.cost_history
               DOptimizerDefaultMonitor.dcost_history
    
   Dictionaries mapping the iteration number to the cost and cost
   derivative, respectively.  These are reset at the beginning of each
   optimization.
