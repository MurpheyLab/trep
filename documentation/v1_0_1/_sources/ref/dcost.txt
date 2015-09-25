.. _trep_dcost:

:class:`DCost` - Discrete Trajectory Cost
=========================================

.. currentmodule:: trep.discopt

The :class:`DCost` class defines the incremental and terminal costs of
a trajectory during a discrete trajectory optimization.  It is used in
conjunction with :class:`DSystem` and :class:`DOptimizer`.  

The discrete trajectory optimization finds a trajectory that minimizes
a cost of the form:

.. math::

   h(\xi) = \sum_{k=0}^{k_f-1} \ell(x(k), u(k), k) + m(x(k_f))

:class:`DCost` defines the costs :math:`\ell(x, u, k)` and
:math:`m(x)` for a system and calculates their 1\ :sup:`st` and 2\
:sup:`nd` derivatives.

The current implementation defines a suitable cost for tracking a
desired trajectory:

.. math:: 

   \ell(x, u, k) = \frac{1}{2}\left((x - x_d(k))^T Q (x - x_d(k)) + (x - u_d(k))^T R (u - u_d(k))\right)

   m(x) = \frac{1}{2}(x - x_d(k_f))^T Q (x - x_d(k_f))

where :math:`x_d(k)` and :math:`u_d(k)` are the desired state and
input trajectories and :math:`Q` and :math:`R` are positive definite
matrices that define their weighting.

DCost Objects
-------------

.. class:: DCost(xd, ud, Q, R)

   :param xd: The desired state trajectory
   :type xd: numpy array, shape (N, nX)
   :param ud: The desired input trajectory
   :type ud: numpy array, shape (N-1, nU)
   :param Q: Cost weights for the states
   :type Q: numpy array, shape (nX, nX)
   :param R: Cost weights for the inputs
   :type R: numpy array, shape (nU, nU)

   Create a new cost object for the desired states *xd* weighted by
   *Q* and the desired inputs *ud* weighted by *R*.


.. attribute:: DCost.Q

   *(numpy array, shape (nX, nX))*

   The weights of the states.


.. attribute:: DCost.R

   *(numpy array, shape (nU, nU))*

   The weights of the inputs.



Costs
^^^^^

.. method:: DCost.l(xk, uk, k)

   :param xk: Current state
   :type xk: numpy array, shape (nX)
   :param uk: Current input
   :type uk: numpy array, shape (nU)
   :param k: Current discrete time
   :type k: int
   :rtype: float

   Calculate the incremental cost of *xk* and *uk* at discrete time
   *k*.


.. method:: DCost.m(xkf)

   :param xkf: Final state
   :type xkf: numpy array, shape (nX)
   :rtype: float

   Calculate the terminal cost of *xk*.


1\ :sup:`st` Derivatives
^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: DCost.l_dx(xk, uk, k)

   :param xk: Current state
   :type xk: numpy array, shape (nX)
   :param uk: Current input
   :type uk: numpy array, shape (nU)
   :param k: Current discrete time
   :type k: int
   :rtype: numpy array, shape (nX)

   Calculate the derivative of the incremental cost with respect to
   the state.

   
.. method:: DCost.l_du(xk, uk, k)

   :param xk: Current state
   :type xk: numpy array, shape (nX)
   :param uk: Current input
   :type uk: numpy array, shape (nU)
   :param k: Current discrete time
   :type k: int
   :rtype: numpy array, shape (nU)

   Calculate the derivative of the incremental cost with respect to
   the input.

   
.. method:: DCost.m_dx(xkf)

   :param xkf: Current state
   :type xkf: numpy array, shape (nX)
   :rtype: numpy array, shape (nX)

   Calculate the derivative of the terminal cost with respect to the
   final state.


2\ :sup:`nd` Derivatives
^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: DCost.l_dxdx(xk, uk, k)

   :param xk: Current state
   :type xk: numpy array, shape (nX)
   :param uk: Current input
   :type uk: numpy array, shape (nU)
   :param k: Current discrete time
   :type k: int
   :rtype: numpy array, shape (nX, nX)

   Calculate the second derivative of the incremental cost with
   respect to the state.  For this implementation, this is always
   equal to :attr:`Q`.


.. method:: DCost.l_dudu(xk, uk, k)

   :param xk: Current state
   :type xk: numpy array, shape (nX)
   :param uk: Current input
   :type uk: numpy array, shape (nU)
   :param k: Current discrete time
   :type k: int
   :rtype: numpy array, shape (nU, nU)

   Calculate the second derivative of the incremental cost with
   respect to the inputs.  For this implementation, this is always
   equal to :attr:`R`.


.. method:: DCost.l_dxdu(xk, uk, k)

   :param xk: Current state
   :type xk: numpy array, shape (nX)
   :param uk: Current input
   :type uk: numpy array, shape (nU)
   :param k: Current discrete time
   :type k: int
   :rtype: numpy array, shape (nX, nU)

   Calculate the second derivative of the incremental cost with
   respect to the state and inputs.  For this implementation, this is
   always equal to zero.


.. method:: DCost.m_dxdx(xkf)

   :param xkf: Current state
   :type xkf: numpy array, shape (nX)
   :rtype: numpy array, shape (nX, nX)

   Calculate the second derivative of the terminal cost.  For this
   implementation, this is always equal to :attr:`Q`.
