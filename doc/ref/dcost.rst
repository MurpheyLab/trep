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


