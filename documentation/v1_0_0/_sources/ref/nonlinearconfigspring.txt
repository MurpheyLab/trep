:class:`NonlinearConfigSpring` -- Nonlinear spring acting on a configuration variable
=====================================================================================

.. currentmodule:: trep.potentials

Like :class:`ConfigSpring` which creates a spring
force directly on a generalized coordinate, a :class:`NonlinearConfigSpring` 
implements a force, :math:`F_{q}`  on a generalized coordinate which can be a function of configuration:

.. math::
   F_{q} = -f (m\cdot q+b)

where :math:`f` is a spline function provided by the user, :math:`m` is a linear scaling factor, and :math:`b` is an offset value.  


   .. table:: **Implemented Calculations**

      ===========   ===========
      Calculation   Implemented
      ===========   ===========
      V                 N
      V_dq              Y
      V_dqdq            Y
      V_dqdqdq          Y 
      ===========   ===========



NonlinearConfigSpring Objects
-----------------------------

.. class:: NonlinearConfigSpring(system, config, spline[, m=1.0, b=0.0, name=None])

   Create a new nonlinear spring acting on the specified configuration variable.
   The configuration variable must already exist in the system.

.. attribute:: NonlinearConfigSpring.config

   The configuration variable that spring depends and acts on.

   *(read only)*

.. attribute:: NonlinearConfigSpring.spline

   A :class:`Spline` object relating the configuration to force.  See the :class:`trep.Spline` documentation
   to create a spline object.
                   
.. attribute:: NonlinearConfigSpring.m

   A linear scaling factor on the configuration.

.. attribute:: NonlinearConfigSpring.b

   An offset factor on the scaled configuration.

