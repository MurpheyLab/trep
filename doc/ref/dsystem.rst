.. _trep_dsystem:

:class:`DSystem` - Discrete System wrapper for Midpoint Variational Integrators
===============================================================================

.. currentmodule:: trep.discopt

:class:`DSystem` objects represent a :class:`MidpointVI` variational
integrators as first order discrete systems of the form :math:`x(k+1)
= f(x(k), u(k), k)`. This representation is used by :class:`DOptimizer`
for discrete trajectory optimization. :class:`DSystem` also provides
methods for automatically calculating linearizations and feedback
controllers along trajectories.

The discrete state consists of the variational integrator's full
configuration, the dynamic momentum, and the kinematic velocity:

.. math::

   x(k) = \begin{bmatrix} q_d(k) \\ q_k(k) \\ p(k) \\ v_k(k) \end{bmatrix}

The configuration and momentum are the same as in :class:`MidpointVI`.
The kinematic velocity is calculated as:

.. math::

   v_k(k) = \frac{q_k(k) - q_k(k-1)}{t(k) - t(k-1)}

The discrete input consists of the variational integrator's force
inputs and the future state of the kinematic configurations:

.. math::

   u(k) = \begin{bmatrix} \mu(k) \\ q_k(k+1) \end{bmatrix}

where the force inputs are denoted by :math:`\mu(k)` to distinguish
them from the discrete system inputs :math:`u(k)`.  

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


.. attribute:: DSystem.nU

   Number of inputs to the discrete system."""


.. attribute:: DSystem.system

   The mechanical system modeled by the variational integrator.


.. attribute:: DSystem.time

   The time of the discrete steps.


.. attribute:: xk

   Current state of the system.


.. attribute:: uk

   Current input of the system.


.. attribute:: k
               
   Current discrete time of the system.


.. method:: DSystem.kf()

   Return the last available state that the system can be set to.
   This is one less than len(self.time).


State and Trajectory Manipulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: DSystem.build_state(Q=None, p=None, v=None)

   Build state vector from components.  Unspecified components
   are set to zero.


.. method:: DSystem.build_input(u=None, rho=None)

   Build input vector from components.  Unspecified components
   are set to zero.


.. method:: DSystem.build_trajectory(Q=None, p=None, v=None, u=None, rho=None)

   Combine component trajectories into a state and input trajectories.
   The state length is the same as the time base, the input length is
   one less than the time base.  Unspecified components are set to
   zero.

   dsys.build_trajectory() -> all zero state and input trajectories

        
.. method:: DSystem.split_state(X=None)

   Split a state vector into its configuration, moementum, and
   kinematic velocity parts.  If X is None, returns zero arrays for
   each component.

   Returns (Q,p,v)


.. method:: DSystem.split_input(U=None)
            
   Split a state input vector U into it's force and kinematic input
   parts, (u, rho).  If U is empty, returns zero arrays of the
   appropriate size.  


.. method:: DSystem.split_trajectory(X=None, U=None)

   Split an X,U state trajectory into its Q,p,v,u,rho components.
   If X or U are None, the corresponding components are arrays of
   zero.


.. method:: DSystem.import_trajectory(dsys_a, X, U)

   dsys_b = self
   
   Maps a trajectory X,U for dsys_a to a trajectory nX, nY for
   dsys_b.


.. method:: DSystem.save_state_trajectory(filename, X, U)

   Save a trajectory to a file.


.. method:: DSystem.load_state_trajectory(filename)

   Load a trajectory from a file.


Dynamics and Derivatives
^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: DSystem.set(xk, uk, k, lambda_k=None)

   Set the current state, input, and time of the discrete system.


.. method:: DSystem.step(uk)


   Advance the system to the next discrete time using the given
   values for the input.  Returns a numpy array.


.. method:: DSystem.f()

   Get the next state of the system, :math:`x(k+1)`.  


.. method:: DSystem.fdx()

   :rtype: numpy array, shape (nX, nX)

.. method:: DSystem.fdu()

   :rtype: numpy array, shape (nX, nU)

   These functions return first derivatives of the system dynamics
   :math:`f()` as numpy arrays with the derivatives across the rows.


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

   Calculate the linearization of the system dynamics about a
   trajectory.  *X* and *U* do not have to be an exact trajectory of
   the system.

   Returns the linearization in a tuple ``(A, B)``\ .


.. method:: DSystem.project(bX, bU[, Kproj=None])

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
