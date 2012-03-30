.. currentmodule:: trep.forces

:class:`ConfigForce` -- Apply forces to a configuration variable.
=================================================================

:class:`ConfigForce` creates an input variable that applies a force
directly to a configuration variable:

.. math::

   F_{q_i} = u(t)

where the :math:`u(t)` is a :class:`Input` variable.

.. table:: **Implemented Calculations**

      ===========   ===========
      Calculation   Implemented
      ===========   ===========
      f                 Y
      f_dq              Y
      f_ddq             Y
      f_du              Y
      f_dqdq            Y
      f_ddqdq           Y
      f_ddqddq          Y
      f_dudq            Y
      f_duddq           Y
      f_dudu            Y
      ===========   ===========

.. admonition:: Examples

   ``forced-pendulum-inverse-dynamics.py``, ``initial-conditions.py``,
   ``pend-on-cart-optimization.py``


.. class:: ConfigForce(system, config, finput, name=None)

   Create a new input to apply a force on a configuration variable.

   *config* should be an existing configuration variable (a name,
   index, or object).

   *finput* should be a string to name the new input variable.


.. attribute:: ConfigForce.finput

   The input variable (:class:`Input`) that controls this force.

.. attribute:: ConfigForce.config

   The configuration variable (:class:`Config`) that this force is applied to.
