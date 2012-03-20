.. _trep_dsystem:

:class:`DSystem` - Discrete System wrapper for Midpoint Variational Integrators
===============================================================================

.. currentmodule:: trep.discopt

.. contents::
   :local:


:class:`DSystem` objects represent a :class:`MidpointVI` variational
integrators as first order discrete systems of the form :math:`X(k+1)
= f(X(k), U(k), k)`. This representation is used by :class:`DOptimizer`
for discrete trajectory optimization. :class:`DSystem` also provides
methods for automatically calculating linearizations and feedback
controllers along trajectories.

The discrete state consists of the variational integrator's full
configuration, the dynamic momentum, and the kinematic velocity:

.. math::

   X(k) = \begin{bmatrix} q_d(k) \\ q_k(k) \\ p(k) \\ v_k(k) \end{bmatrix}

The configuration and momentum are the same as in :class:`MidpointVI`.
The kinematic velocity is calculated as:

.. math::

   v_k(k) = \frac{q_k(k) - q_k(k-1)}{t(k) - t(k-1)}

The discrete input consists of the variational integrator's force
inputs and the future state of the kinematic configurations:

.. math::

   U(k) = \begin{bmatrix} u(k) \\ \rho(k) \end{bmatrix}

where the kinematic inputs are denoted by :math:`\rho` thoughout this
code.  Additionally, the state input *U* is always capitalized to
distinguish it from the force input *u* which is always lower case.

:class:`DSystem` provides methods for converting between trajectories
for the discrete system and trajectories for the variational
integrator.


DSystem Objects
---------------

.. class:: DSystem(varint, t)

   :param varint: The variational integrator being represented.
   :type varint: :class:`MidpointVI` instance
   :param t: An array of times
   :type t: numpy array of floats, shape (N)

   Create a discrete system wrapper for the variational integrator
   *varint* and the time *t*.  The time *t* is the array :math:`t(k)`
   that maps a discrete time index to a time.  It should have the same
   length *N* as trajectories used with the system.


.. attribute:: DSystem.nX

   Number of states to the discrete system.

   *int*


.. attribute:: DSystem.nU

   Number of inputs to the discrete system.

   *int*

.. attribute:: DSystem.varint

   The variational integrator wrapped by this instance.

   :class:`MidpointVI`

.. attribute:: DSystem.system

   The mechanical system modeled by the variational integrator.

   :class:`System`

.. attribute:: DSystem.time

   The time of the discrete steps.

   numpy array, shape (N)

.. attribute:: xk

   Current state of the system.

   numpy array, shape (nX)

.. attribute:: uk

   Current input of the system.

   numpy array, shape (nU)

.. attribute:: k
               
   Current discrete time of the system.
   
   int

.. method:: DSystem.kf()

   :rtype: int

   Return the last available state that the system can be set to.
   This is one less than ``len(self.time)``.


State and Trajectory Manipulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: DSystem.build_state([Q=None, p=None, v=None])

   :param Q: A configuration vector
   :type Q: numpy array, shape (nQ)
   :param p: A momentum vector
   :type p: numpy array, shape (nd)
   :param v: A kinematic velocity vector
   :type v: numpy array, shape (nk)

   Build a state vector *Xk* from components.  Unspecified components
   are set to zero.


.. method:: DSystem.build_input([u=None, rho=None])

   :param u: An input force vector 
   :type u: numpy array, shape (nu)
   :param rho:  A kinematic input vector
   :type rho: numpy array, shape (nk)
   :type: numpy array, shape (nU)
     
   Build an input vector *Uk* from components.  Unspecified components
   are set to zero.


.. method:: DSystem.build_trajectory([Q=None, p=None, v=None, u=None, rho=None])
            
   :param Q: A configuration trajectory
   :type Q: numpy array, shape (N, nQ)
   :param p: A momentum trajectory
   :type p: numpy array, shape (N, nd)
   :param v: A velocity trajectory
   :type v: numpy array, shape (N, nk)
   :param u: An input force trajectory
   :type u: numpy array, shape (N-1, nu)
   :param rho: A kinematic input trajectory
   :type rho: numpy array, shape (N-1, nk)
   :rtype: tuple of (*X*, *U*)

   Combine component trajectories into a state and input trajectories.
   The state length is the same as the time base, the input length is
   one less than the time base.  Unspecified components are set to
   zero::

     >>> dsys.build_trajectory() # Create a zero state and input trajectory

        
.. method:: DSystem.split_state([X=None])

   :param X: A state vector for the system
   :type X: numpy array, shape (nX)
   :rtype: tuple of (*Q*, *p*, *v*)
   
   Split a state vector into its configuration, moementum, and
   kinematic velocity parts.  If *X* is :data:`None`, returns zero
   arrays for each component.


.. method:: DSystem.split_input([U=None])

   :param U: An input vector for the system
   :type U: numpy array, shape (nU)
   :rtype: tuple of (*u*, *rho*)
            
   Split a state input vector *U* into its force and kinematic input
   parts, (*u*, *rho*).  If *U* is :data:`None`, returns zero arrays of the
   appropriate size.


.. method:: DSystem.split_trajectory([X=None, U=None])
            
   :param X: A state trajectory
   :type X: numpy array, shape (N, nX)
   :param U: An input trajectory
   :type U: numpy array, shape (N-1, nU)
   :rtype: tuple of (*Q*, *p*, *v*, *u*, *rho*)

   Split the state trajectory (*X*, *U*) into its *Q*, *p*, *v*, *u*,
   *rho* components.  If *X* or *U* are :data:`None`, the corresponding
   components are arrays of zero.


.. method:: DSystem.convert_trajectory(dsys_a, X, U)

   :param dsys_a: Another discrete system
   :type dsys_a: :class:`DSystem`
   :param X: A state trajectory for *dsys_a*
   :type X: numpy array, shape (N, nX)
   :param U: An input trajectory for *dsys_a*
   :type U: numpy array, shape (N, nU)
   :rtype: trajectory for this system, tuple (*X*, *U*)

   Convert the trajectory (*X*, *U*) for *dsys_a* into a trajectory
   for this system.  This reorders the trajectory components according
   to the configuration and input variable names and drops components
   that aren't in this system.  Variables in this system that are not
   in *dsys_a* are replaced with zero.

   .. note:: 
      
      The returned path may not be a valid trajectory for this system
      in the sense that :math:`x(k+1) = f(x(k), u(k), k)`.  This
      function only reorders the information.


.. method:: DSystem.save_state_trajectory(filename, [X=None, U=None])

   :param filename: Location to save the trajectory
   :type filename: string
   :param X: A state trajectory
   :type X: numpy array, shape (N, nX)
   :param U: An input trajectory
   :type U: numpy array, shape (N-1, nU)

   Save a trajectory to a file.  This splits the trajectory with
   :meth:`split_trajectory` and saves the results with
   :func:`trep.save_traejctory`.  If *X* or *U* are not specified,
   they are replaced with zero arrays.


.. method:: DSystem.load_state_trajectory(filename)

   :param filename: Location of saved trajectory
   :type filename: string
   :rtype: tuple of (X, U)

   Load a trajectory from a file that was stored with
   :meth:`save_state_trajectory` or
   :func:`trep.save_trajectory`.  

   If the file does not contain complete information for the system
   (e.g, it was saved for a different system with different states, or
   the inputs were not saved), the missing components will be filled
   with zeros.


Dynamics
^^^^^^^^

.. method:: DSystem.set(self, xk, uk, k[, \
                        xk_hint=None, lambda_hint=None])

   Set the current state, input, and time of the discrete system.

   If *xk_hint* and *lambda_hint* are provided, these are used to
   provide hints to hints to :meth:`MidpointVI.step`.  If the solution
   is known (for example, if you are calculating the linearization
   about a known trajectory) this can result in faster performance by
   reducing the number of root solver iterations in the variational
   integrator.


.. method:: DSystem.step(self, uk[, xk_hint=None, lambda_hint=None])

   Advance the system to the next discrete time using the given
   input *uk*.

   This is equivalent to calling ``self.set(self.f(), uk, self.k+1)``.

   If *xk_hint* and *lambda_hint* are provided, these are used to
   provide hints to hints to :meth:`MidpointVI.step`.  If the solution
   is known (for example, if you are calculating the linearization
   about a known trajectory) this can result in faster performance by
   reducing the number of root solver iterations in the variational
   integrator.


.. method:: DSystem.f()

   :rtype: numpy array, shape (nX)

   Get the next state of the system, :math:`x(k+1)`.  


First Derivatives
^^^^^^^^^^^^^^^^^

.. method:: DSystem.fdx()

   :rtype: numpy array, shape (nX, nX)

.. method:: DSystem.fdu()

   :rtype: numpy array, shape (nX, nU)

   These functions return first derivatives of the system dynamics
   :math:`f()` as numpy arrays with the derivatives across the rows.


Second Derivatives
^^^^^^^^^^^^^^^^^^

.. method:: DSystem.fdxdx(z)

   :param z: adjoint vector
   :type z: numpy array, shape (nX)
   :rtype: numpy array, shape (nU, nU)

.. method:: DSystem.fdxdu(z)
   
   :param z: adjoint vector
   :type z: numpy array, shape (nX)
   :rtype: numpy array, shape (nX, nU)

.. method:: DSystem.fdudu(z)

   :param z: adjoint vector
   :type z: numpy array, shape (nX)
   :rtype: numpy array, shape (nU, nU)

   These functions return the product of the 1D array *z* and the
   second derivative of f.  For example:

   .. math::
      
      z^T \derivII[f]{u}{u}


Linearization and Feedback Controllers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: DSystem.linearize_trajectory(X, U)

   :rtype: tuple (A, B)

   Calculate the linearization of the system dynamics about a
   trajectory.  *X* and *U* do not have to be an exact trajectory of
   the system.

   Returns the linearization in a tuple ``(A, B)``\ .


.. method:: DSystem.project(bX, bU[, Kproj=None])

   :rtyple: tuple (X, U)

   Project *bX* and *bU* into a nearby trajectory for the system using
   a linear feedback law::

        X[0] = bX[0]
        U[k] = bU[k] - Kproj * (X[k] - bU[k])
        X[k+1] = f(X[k], U[k], k)

   If no feedback law is specified, one will be created by
   :meth:`calc_feedback_controller` along *bX* and *bU*.  This is
   typically a **bad idea** if *bX* and *bU* are not very close to an
   actual trajectory for the system.

   Returns the projected trajectory in a tuple ``(X, U)``\ .


.. method:: DSystem.calc_feedback_controller(X, U[, Q=None, R=None, return_linearization=False])

   :rtype: K or tuple (K, A, B)

   Calculate a stabilizing feedback controller for the system about a
   trajectory *X* and *U*.  The feedback law is calculated by solving
   the discrete LQR problem for the linearization of the system about
   *X* and *U*.

   *X* and *U* do not have to be an exact trajectory of the system,
   but if they are not close, the controller is unlikely to be
   effective.

   If the LQR weights *Q* and *R* are not specified, identity matrices
   are used.

   If *return_linearization* is :data:`False`, the return value is the
   feedback control law, *K*.  

   If *return_linearization* is :data:`True`, the method returns the
   linearization as well in a tuple:  ``(K, A, B)``.


Checking the Derivatives
^^^^^^^^^^^^^^^^^^^^^^^^
        
.. method:: DSystem.check_fdx(xk, uk, k[, delta=1e-5])
            DSystem.check_fdu(xk, uk, k[, delta=1e-5])
            DSystem.check_fdxdx(xk, uk, k[, delta=1e-5])
            DSystem.check_fdxdu(xk, uk, k[, delta=1e-5])
            DSystem.check_fdudu(xk, uk, k[, delta=1e-5])

   :param xk: A valid state of the system
   :type xk: numpy array, shape (nX)
   :param uk: A valid input to the system
   :type uk: numpy array, shape (nU)
   :param k: A valid discrete time index 
   :type k: int
   :param delta: The perturbation for approximating the derivative.

   These functions check derivatives of the discrete state dynamics
   against numeric approximations generated from lower derivatives
   (e.g, fdx() from f(), and fdudu() from fdu()).  A three point
   approximation is used::

     approx_deriv = (f(x + delta) - f(x - delta)) / (2 * delta)

   Each function returns a tuple ``(error, exact_norm, approx_norm)``
   where ``error`` is the norm of the difference between the exact and
   approximate derivative, ``exact_norm`` is the norm of the exact
   derivative, ``approx_norm`` is the norm of the approximate derivative.
