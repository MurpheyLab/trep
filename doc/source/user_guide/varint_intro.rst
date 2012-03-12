.. _varint-intro:

Discrete Dynamics and Variational Integrators
=============================================

The variational integrator simulates a discrete mechanical systems.
Given a known state (time :math:`t_{k-1}`, configuration :math:`q_{k-1}`, and
discrete momentum :math:`p_{k-1}`) and inputs (force inputs :math:`u_{k-1}`
and next kinematic configuration :math:`\rho_{k-1}`), the integrator finds the next
state:

.. math::
   
   (t_{k-1}, q_{k-1}, p_{k-1}), (u_{k-1}, \rho_{k-1}) \Rightarrow (t_k, q_k, p_k)

The integrator also finds the discrete constraint force variable
:math:`\lambda_{k-1}`.

The variational integrator finds the next state by numerically solving
the constrained *Discrete Euler-Langrange (DEL)* equation for
:math:`q_k` and :math:`\lambda_{k-1}`:

.. math::
   
   p_{k-1} + D_1L_d(q_{k-1}, q_k, t_{k-1}, t_{k-1}) + f_d^-(q_{k-1}, q_k, u_{k-1}, t_{k-1}, t_k) - Dh^T(q_{k-1})\dot \lambda_{k-1} 
   
   h(q_k) = 0

and then calculating the new discrete momentum:

.. math::
   
   p_k = D_2L_d(q_{k-1}, q_k, t_{k-1}, t_k)


In :mod:`trep`, we simplify notation by letting :math:`k=2`, so that
we consider :math:`t_1`, :math:`q_1`, and :math:`p_1` to be the
previous state, and :math:`t_2`, :math:`q_2`, and :math:`p_2` to be
the new or current state.




.. math::

   L_2 = L_d(q_1, q_2, t_1, t_2)
   
   f_2^\pm = f_d^\pm(q_1, q_2, u_1, t_1, t_2)

   h_1 = h(q_1)

   h_2 = h(q_2)
