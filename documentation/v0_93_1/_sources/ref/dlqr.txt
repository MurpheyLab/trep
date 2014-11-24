.. _trep_dlqr:

Discrete LQ Problems
====================

.. currentmodule:: trep.discopt


The :mod:`trep.discopt` module provides functions for solving
time-varying discrete LQ problems.  


The Linear Quadratic Regulator (LQR) Problem
--------------------------------------------

The LQR problem is to find the input for a linear system that
minimizes a quadratic cost.  The optimal input turns out to be a
feedback law that is independent of the system's initial condition.
Because of this, the LQR problem is a useful tool to automatically
calculate a stabilizing feedback controller for a dynamic system.  For
nonlinear systems, the LQR problem is solved for the linearization of
the system about a trajectory to get a locally stabilizing controller.


**Problem Statement:** Given a discrete linear system Find the control
input :math:`u(k)` that minimizes a quadratic cost:

.. math::

   V(x(k_0), u(\cdot), k_0) = \sum_{k=k_0}^{k_f-1} \left[
   x^T(k)Q(k)x(k) + u^T(k)R(k)u(k) \right] + x^T(k_f) Q(k_f) x(k_f)

where 

.. math:: 

   \begin{align}
       R(k) &= R^T(k) \geq 0 \ \forall\ k \in \{k_0 \dots (k_f-1)\}
       \\
       Q(k) &= Q^T(k) \geq 0 \ \forall\ k \in \{k_0 \dots k_f\}
       \\
       x(k_0)&\text{ is known.} 
       \\
       x(k+1) &= A(k)x(k) + B(k)u(k)
   \end{align}

**Solution:** The optimal control :math:`u^*(k)` and optimal cost
:math:`V^*(x(k_0), k_0)` are

.. math::

   \begin{align}
       u^*(k) &= -\mathcal{K}(k) x(k)
       \\
       V^*(x(k_0), k_0) &= x^T(k_0) P(k_0) x(k_0)
   \end{align}

where  

.. math::

   \mathcal{K}(k) = \Gamma^{-1}(k) B^T(k) P(k+1) A(k)

   \Gamma(k) = R(k) + B^T(k)P(k+1)B(k)

and :math:`P(k+1)` is a symmetric time varying matrix satisfying a
discrete Ricatti-like equation:

.. math::

   \begin{align}
      P(k_f) &= Q(k_f) \\
      P(k) &= Q(k) + A^T(k)P(k+1)A(k) - \mathcal{K}^T(k)\Gamma(k)\mathcal{K}(k)
   \end{align}




.. function:: solve_tv_lqr(A, B, Q, R)
   
   :param A: Linear system dynamics
   :type A: Sequence of N numpy arrays, shape (nX, nX)
   :param B: Linear system input matrix
   :type B: Sequence of N numpy arrays, shape (nX, nU)
   :param Q: Quadratic State Cost
   :type Q: Function Q(k) returning numpy array, shape (nX, nX)
   :param R: Quadratic Input Cost
   :type R: Function R(k) returning numpy array, shape (nU, nU)
   :rtype: tuple (K, P)

   This function solve the time-varying discrete LQR problem for the
   linear system *A*, *B* and costs *Q* and *R*.

   *A* is a sequence of the linear system dynamics, ``A[k]``.

   *B* is a sequence of the linear system's input matrix, ``B[k]``.

   *Q* is a function ``Q(k)`` that returns the state cost matrix at
   time *k*.  For example, if :math:`Q(k) = \mathcal{I}`::

      Q = lambda k: numpy.eye(nX)

   *R* is a function ``Q(k)`` that returns the state cost matrix at
   time *k*.  For example, if the cost matrices are stored in an array
   *r_costs*::

      R = lambda k: r_costs[k]

   The function returns the optimal feedback law
   :math:`\mathcal{K(k)}` and the solution to the discrete Ricatti
   equation at k=0, :math:`P(0)`.  *K* is a sequence of N numpy arrays of shape
   (nU,nX).  *P* is a single (nX, nX) numpy array.



The Linear Quadratic (LQ) Problem
---------------------------------

The LQ problem is to find the input for a linear system that minimizes
a cost with linear and quadratic terms.  In trep, the LQ problem is a
sub-problem for discrete trajectory optimization that is used to
calculate the descent direction at each iteration.

**Problem Statement:** Find the control input :math:`u(k)` that
minimizes the cost:

.. math::

  V(x(k_0), u(\cdot), k_0) = 
  \sum_{k=k_0}^{k_f-1} \Bigg[
    2 \begin{bmatrix} q(k) \\ r(k) \end{bmatrix}^T
    \begin{bmatrix} x(k) \\ u(k) \end{bmatrix}
    +
    \begin{bmatrix} x(k) \\ u(k) \end{bmatrix}^T
    \begin{bmatrix} Q(k) & S(k) \\ S^T(k) & R(k) \end{bmatrix}
    \begin{bmatrix} x(k) \\ u(k) \end{bmatrix}
    \Bigg] \\
  + 2 q^T(k_f) x(k_f) + x^T(k_f)Q(k_f)x(k_f) 

where 

.. math::

   \begin{align*}
       R(k) &= R^T(k) > 0 \ \forall\ k \in \{k_0 \dots (k_f-1)\}
       \\
       Q(k) &= Q^T(k) \geq 0 \ \forall\ k \in \{k_0 \dots k_f\}
       \\
       x(k_0)&\text{ is known.} 
       \\
       x(k+1) &= A(k)x(k) + B(k)u(k)
   \end{align*}

**Solution:** The optimal control :math:`u^*(k)` and optimal cost
:math:`V^*(x(k_0), k_0)` are:

.. math::

   \begin{align*}
       u^*(k) &= -\mathcal{K}(k) x(k) - C(k)
       \\
       V^*(x(k_0), k_0) &= x^T(k_0) P(k_0) x(k_0) + 2 b^T(k_0) x(k_0) + c(k_0)
   \end{align*}

where:

.. math:: 

  K(k) = \Gamma^{-1}(k) \left[B^T(k)P(k+1)A(k) + S^T(k)\right]

  C(k) = \Gamma^{-1}(k) \left[B^T(k)b(k+1) + r(k) \right]

  \Gamma(k) = \left[ R(k) + B^T(k)P(k+1)B(k) \right]

and :math:`P(k)`, :math:`b(k)`, and :math:`c(k)` are solutions to
backwards difference equations:

.. math:: 

  \begin{align*}
      P(k_f) &= Q(k_f) 
      \\
      P(k) &= Q(k) + A^T(k)P(k+1)A(k) - \mathcal{K}^T(k)\Gamma(k)\mathcal{K}(k)
  \end{align*}

  \begin{align*}
      b(k_f) &= q(k_f)
      \\
      b(k) &= \left[A^T(k) - \mathcal{K}^T(k)B^T(k) \right]b(k+1) + q(k) - \mathcal{K}^T(k)r(k)
  \end{align*} 

  \begin{align*}
      c(k_f) &= 0
      \\
      c(k) &= c(k+1) - C(k)^T\Gamma(k) C(k)
  \end{align*}




.. function:: solve_tv_lq(A, B, q, r, Q, S, R)
              
   :param A: Linear system dynamics
   :type A: Sequence of N numpy arrays, shape (nX, nX)
   :param B: Linear system input matrix
   :type B: Sequence of N numpy arrays, shape (nX, nU)
   :param q: Linear State Cost
   :type q: Sequence of N numpy arrays, shape (nX)
   :param r: Linear Input Cost
   :type r: Sequence of N numpy arrays, shape (nU)
   :param Q: Quadratic State Cost
   :type Q: Function Q(k) returning numpy array, shape (nX, nX)
   :param S: Quadratic Cross Term Cost
   :type S: Function S(k) returning numpy array, shape (nX, nU)      
   :param R: Quadratic Input Cost
   :type R: Function R(k) returning numpy array, shape (nU, nU)
   :rtype: tuple (K, C, P, b)

   This function solve the time-varying discrete LQ problem for the
   linear system *A*, *B*.

   *A[k]* is a sequence of the linear system dynamics, :math:`A(k)`.

   *B[k]* is a sequence of the linear system's input matrix, :math:`B(k)`.

   *q[k]* is a sequence of the linear state cost, :math:`q(k)`.

   *r[k]* is a sequence of the linear input cost, :math:`r(k)`.

   *Q(k)* is a function that returns the quadratic state cost matrix
   *at time k*.  For example, if :math:`Q(k) = \mathcal{I}`::

      Q = lambda k: numpy.eye(nX)

   *S(k)* is a function that returns the quadratic cross term cost
   *matrix at time k*.  

   *R(k)* is a function that returns the state cost matrix at time
   *k*.  For example, if the cost matrices are stored in an array
   *r_costs*::

      R = lambda k: r_costs[k]

   The function returns the optimal feedback law
   :math:`\mathcal{K(k)}`, the affine input term :math:`C(k)`, and the
   last solution to two of the difference equations, :math:`P(0)` and
   :math:`b(0)`.

   *K* is a sequence of N numpy arrays of shape (nU,nX).

   *C* is a sequence of N numpy arrays of shape (nU).

   *P* is a single (nX, nX) numpy array.

   *b* is a single (nX) numpy array.

