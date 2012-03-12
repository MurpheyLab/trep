.. _trep_midpointvi:

:class:`MidpointVI` - Midpoint Variational Integrator
=====================================================

.. currentmodule:: trep

The :class:`MidpointVI` class implements a variational integrator
using a midpoint quadrature.  The integrator works with any system,
including those with constraints, forces, and kinematic configuration
variables.  The integrator implements full first and second
derivatives of the discrete dynamics.

For information on variational integrators (including relevant
notation used in this manual), see :ref:`varint-intro`.

The Midpoint Variational Integrator is defined by the approximations
for :math:`L_d` and :math:`f^\pm`:

.. math::
   
   L_2(q_1, q_2, t_1, t_2) = (t_2-t_1)L\left(\frac{q_1 + q_2}{2}, 
                                                         \frac{q_2-q_1}{t_2-t_1}\right)

   f_2^-(q_{k-1}, q_k, t_{k-1}, t_k) = (t_k-t_{k-1})f\left(\frac{q_{k-1} + q_k}{2}, 
                                                           \frac{q_k-q_{k-1}}{t_k-t_{k-1}},
                                                           u_{k-1},
                                                           \frac{t_k-t_{k-1}}{2}\right)

   f_d^+(q_{k-1}, q_k, t_{k-1}, t_k) = 0

where :math:`L` and :math:`f` are the continuous Lagrangian and
generalized forcing on the system.  

.. contents::
   :local:

MidpointVI Objects
------------------

.. class:: MidpointVI(system, tolerance=1e-10, num_threads=None)

   Create a new empty mechanical system.  *system* is a valid
   :class:`System` object that will be simulation. 

   *tolerance* sets the desired tolerance of the root solver when
    solving the DEL equation to advance the integrator.

    :class:`MidpointVI` makes use of multithreading to speed up the
    calculations.  *num_threads* sets the number of threads used by
    this integrator.  If *num_threads* is :data:`None`, the integrator
    will use the number of available processors reported by Python's
    :mod:`multiprocessing` module.


.. attribute:: MidpointVI.system
           
   The :class:`System` that the integrator simulates.

.. attribute:: MidpointVI.nq

   Number of configuration variables in the system.
        
.. attribute:: MidpointVI.nd

   Number of dynamic configuration variables in the system.

.. attribute:: MidpointVI.nk

   Number of kinematic configuration variables in the system.

.. attribute:: MidpointVI.nu

   Number of input variables in the system.

.. attribute:: MidpointVI.nc

   Number of constraints in the system.


Integrator State 
^^^^^^^^^^^^^^^^

In continuous dynamics the :class:`System` only calculates the second
derivative, leaving the actual simulation to be done by separate
numerical integrator.  The discrete variational integrator, on the
other hand, performs the actual simulation by finding the next state.
Because of this, :class:`MidpointVI` objects store two discrete
states: one for the previous time step :math:`t_1`, and one for the
new/current time step :math:`t_2`.

Also unlike the continuous :class:`System`, these variables are rarely
set directly.  They are typically modified by the initialization and
stepping methods.

.. attribute:: MidpointVI.t1
               MidpointVI.q1
               MidpointVI.p1

   State variables for the previous time.  :attr:`q1` is the entire
   configuration (dynamic and kinematic).  :attr:`p1` is discrete
   momemtum, which only has a dynamic component, not a kinematic
   component.

.. attribute:: MidpointVI.t2
               MidpointVI.q2
               MidpointVI.p2

   State variables for the current/new time. 

.. attribute:: MidpointVI.u1
               
   The input vector at :math:`t_1`.  

.. attribute:: MidpointVI.lambda1

   The constraint force vector at :math:`t_1`.  These are the
   constraint forces used for the system to move from :math:`t_1` to
   :math:`t_2`.


Initialization
^^^^^^^^^^^^^^

.. method:: MidpointVI.initialize_from_state(t1, q1, p1, lambda1=None)
        
   Initialize the integrator from a known state (time, configuration,
   and momentum).  This prepares the integrator to start simulating
   from time :math:`t_1`.

   *lambda1* can optionally specify the initial constraint vector.
   This serves at the initial guess for the simulation's root solving
   algorithm.  If you have a simulation that you are trying to
   re-simulate, but from a different starting time (e.g, you saved a
   forward simulation and now want to move backwards and calculate the
   linearization at each time step for a LQR problem.), it is a good
   idea to save *lambda1* during the simulation and set it here.
   Otherwise, the root solver can find a slightly different solution
   which eventually diverges.  If not specified, it defaults to the
   zero vector.

               
.. method:: MidpointVI.initialize_from_configs(t0, q0, t1, q1, lambda1=None)
        
   Initialize the integrator from two consecutive time and
   configuration pairs.  This calculates :math:`p_1` from the two
   pairs and initializes the integrator with the state (t1, q1, p1).

   *lambda1* is optional.  See :meth:`initialize_from_state`.


Dynamics
^^^^^^^^
        
.. method:: MidpointVI.step(t2, u1=tuple(), k2=tuple(), max_iterations=200, 
                            lambda1=None, q2=None)

   Step the integrator forward to time t2 .  This advances the time
   and solves the DEL equation.  The current state will become the
   previous state (ie, :math:`t_2 \Rightarrow \t_1`, :math:`q_2
   \Rightarrow \q_1`, :math:`p_2 \Rightarrow \p_1`).  The solution
   will be saved as the new state, available through :attr:`t2`,
   :attr:`q2`, and :attr:`p2`.  :attr:`lambda` will be updated with
   the new constraint force, and :attr:`u1` will be updated with the
   value of *u1*.

   *lambda1* and *q2* can be specified to seed the root solving
    algorthm.  If they are :data:`None`, the previous values will be
    used.

   The method returns the number of root solver iterations needed to
   find the solution.
        
.. method:: MidpointVI.calc_f()
        
   Evaluate the DEL equation at the current states.  For dynamically
   consistent states, this should be zero.  Otherwise it is the
   remainder of the DEL.   

   The value is returned as a :mod:`numpy` array.
                

Derivatives of :math:`q_2`
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: MidpointVI.q2_dq1(q=None, q1=None)
            MidpointVI.q2_dp1(q=None, p1=None)
            MidpointVI.q2_du1(q=None, u1=None)
            MidpointVI.q2_dk2(q=None, k2=None)

   Calculate the first derivatives of :math:`q_2` with respect to the
   previous configuration, the previous momentum, the input forces, or
   the kinematic inputs.  

   If both the parameters are :data:`None`, the entire derivative is
   returned as a :mod:`numpy` array with derivatives across the rows.  

   If any parameters are specified, they must be appropriate objects.
   The function will return the specific information requested.  For
   example, :meth:`q2_dq1` will calculate the derivative of the new
   value of *q* with respect to the previous value of *q1*.

   Calculating the derivative of :math:`q_2` with respect to the
   *i*-th configuration variable :math:`\deriv[q_2]{q_1^i}`::

     >>> q1 = system.configs[i]
     >>> mvi.q2_dq1(q1=q1)
     array([ 0.133, -0.017,  0.026, ..., -0.103, -0.017,  0.511])

   Equivalently::

     >>> mvi.q2_dq1()[:,i]
     array([ 0.133, -0.017,  0.026, ..., -0.103, -0.017,  0.511])
   
   Calculating the derivative of the *j*-th new configuration with
   respect to the *i*-th kinematic input:

     >>> q = system.configs[j]
     >>> k2 = system.kin_configs[i]
     >>> mvi.q2_dk2(q, k2)
     0.023027007157071705

     # Or...
     >>> mvi.q2_dk2()[j,i]
     0.023027007157071705

     # Or...
     >>> mvi.q2_dk2()[q.index, k2.k_index]
     0.023027007157071705

.. method:: MidpointVI.q2_dq1dq1(q=None, q1_1=None, q1_2=None)
            MidpointVI.q2_dq1dp1(q=None, q1=None, p1=None)
            MidpointVI.q2_dq1du1(q=None, q1=None, u1=None)
            MidpointVI.q2_dq1dk2(q=None, q1=None, k2=None)
            MidpointVI.q2_dp1dp1(q=None, p1_1=None, p1_2=None)
            MidpointVI.q2_dp1du1(q=None, p1=None, u1=None)
            MidpointVI.q2_dp1dk2(q=None, p1=None, k2=None)
            MidpointVI.q2_du1du1(q=None, u1_1=None, u1_2=None)
            MidpointVI.q2_du1dk2(q=None, u1=None, k2=None)
            MidpointVI.q2_dk2dk2(q=None, k2_1=None, k2_2=None)
            
   Calculate second derivatives of the new configuration.  

   If no parameters specified, the entire second derivative is
   returned as a :mod:`numpy` array.  The returned arrays are indexed
   with the two derivatives variables as the first two dimensions and
   the new state as the last dimension.  

   To calculate the derivative of the *k*-th new configuration with
   respect to the *i*-th previous configuration and *j* previous
   momentum::

     >>> q = system.configs[k]
     >>> q1 = system.configs[i]
     >>> p1 = system.configs[j]
     >>> mvi.q2_dq1dp1(q, q1, p1)
     6.4874251262289155e-06

     # Or...
     >>> result = mvi.q2_dq1dp1(q)
     >>> result[i, j]
     6.4874251262289155e-06
     >>> result[q1.index, p1.index]
     6.4874251262289155e-06

     # Or....
     >>> result = mvi.q2_dq1dp1()
     >>> result[i, j, k]
     6.4874251262289155e-06
     >>> result[q1.index, p1.index, q.index]
     6.4874251262289155e-06


Derivatives of :math:`p_2`
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: MidpointVI.p2_dq1(p=None, q1=None)
            MidpointVI.p2_dp1(p=None, p1=None)
            MidpointVI.p2_du1(p=None, u1=None)
            MidpointVI.p2_dk2(p=None, k2=None)
            MidpointVI.p2_dq1dq1(p=None, q1_1=None, q1_2=None)
            MidpointVI.p2_dq1dp1(p=None, q1=None, p1=None)
            MidpointVI.p2_dq1du1(p=None, q1=None, u1=None)
            MidpointVI.p2_dq1dk2(p=None, q1=None, k2=None)
            MidpointVI.p2_dp1dp1(p=None, p1_1=None, p1_2=None)
            MidpointVI.p2_dp1du1(p=None, p1=None, u1=None)
            MidpointVI.p2_dp1dk2(p=None, p1=None, k2=None)
            MidpointVI.p2_du1du1(p=None, u1_1=None, u1_2=None)
            MidpointVI.p2_du1dk2(p=None, u1=None, k2=None)
            MidpointVI.p2_dk2dk2(p=None, k2_1=None, k2_2=None)

   Calculate the first and second derivatives of :math:`p_2`.  These
   follow the same conventions as the derivatives of :math:`q_2`.

Derivatives of :math:`\lambda_1`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: MidpointVI.lambda1_dq1(constraint=None, q1=None)
            MidpointVI.lambda1_dp1(constraint=None, p1=None)
            MidpointVI.lambda1_du1(constraint=None, u1=None)
            MidpointVI.lambda1_dk2(constraint=None, k2=None)

.. method:: MidpointVI.lambda1_dq1dq1(constraint=None, q1_1=None, q1_2=None)
            MidpointVI.lambda1_dq1dp1(constraint=None, q1=None, p1=None)
            MidpointVI.lambda1_dq1du1(constraint=None, q1=None, u1=None)
            MidpointVI.lambda1_dq1dk2(constraint=None, q1=None, k2=None)
            MidpointVI.lambda1_dp1dp1(constraint=None, p1_1=None, p1_2=None)
            MidpointVI.lambda1_dp1du1(constraint=None, p1=None, u1=None)
            MidpointVI.lambda1_dp1dk2(constraint=None, p1=None, k2=None)
            MidpointVI.lambda1_du1du1(constraint=None, u1_1=None, u1_2=None)
            MidpointVI.lambda1_du1dk2(constraint=None, u1=None, k2=None)
            MidpointVI.lambda1_dk2dk2(constraint=None, k2_1=None, k2_2=None)

   Calculate the first and second derivatives of :math:`\lambda_1`.
   These follow the same conventions as the derivatives of
   :math:`q_2`.  The constraint dimension is ordered according to
   :attr:`System.constraints`.
            
            
            
        
