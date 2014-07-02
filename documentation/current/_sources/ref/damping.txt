.. _trep_force_damping:

.. currentmodule:: trep.forces

:class:`Damping` -- Damping on Configuration Variables
======================================================

:class:`Damping` implements a damping :class:`Force` on the
generalized coordinates of a system:

.. math::
   
   F_{q_i} = -c_i\ \dq_i

where the damping constant :math:`c_i` is a positive real number.

One instance of :class:`Damping` defines damping parameters for every
configuration variable in the system.  You can specify values for
specific configuration variables and have a default value for
the other variables.


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

   ``damped-pendulum.py``, ``extensor-tendon-model.py``, ``forced-pendulum-inverse-dynamics.py``,
   ``initial-conditions.py``, ``pccd.py``,  ``pend-on-cart-optimization.py``, ``puppet-basic.py``,
   ``puppet-continuous-moving.py``, ``puppet-moving.py``, ``pyconfig-spring.py``, ``pypccd.py``, 
   ``radial.py``



.. class:: Damping(system, default=0.0, coefficients={}, name=None)

   Create a new damping force for the system.  *default* is the
   default damping coefficient.  

   Damping coefficients for specific configuration variables can be
   specified with *coefficients*.  This should be a dictionary mapping
   configuration variables (or their names or index) to the damping
   coefficient::
   
     trep.forces.Damping(system, 0.1, {'theta' : 1.0})

.. attribute:: Damping.default
               
   The default damping coefficient for configuration variable.


.. method:: Damping.get_damping_coefficient(config)

   Return the damping coefficient for the specified configuration
   variable.  If the configuration variable does not have a set value,
   the default value is returned.


.. method:: Damping.set_damping_coefficient(config, coeff)

   Set the damping coefficient for a specific configuration variable.
   If *coeff* is :data:`None`, the specific coefficient for that
   variable will be deleted and the default value will be used.

