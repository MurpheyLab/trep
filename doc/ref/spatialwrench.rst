.. currentmodule:: trep.forces

:class:`SpatialWrench` -- Apply a spatial wrench to a frame.
============================================================

:class:`SpatialWrench` applies a fixed or variable wrench to a
coordinate frame.  The wrench is expressed in the world coordinate
frame and applied to the location of the world origin in the
coordinate frame.


.. math::

   F_{q_i} = \left(\deriv[g]{q_i} g^{-1}\right)^{\displaystyle\check{}} \cdot f

where :math:`g_1` is the coordinate frame the wrench is applied to and
:math:`f` is the wrench.  The wench is a vector of six numbers that
defined the forces applied to the x, y, and z axes and torques about
the x, y, and z axes.  In trep, each component of the wrench can be a
fixed real number or an input variable (:class:`trep.Input`).


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


.. class:: SpatialWrench(system, frame, wrench=tuple(), name=None):

   Create a new spatial wrench force to apply to *frame*.  

   *wrench* is a mixed tuple of 6 real numbers and strings.
   Components that are real numbers will be constant values for the
   wrench, while components that are strings will be controlled by
   input variables.  An instance of :class:`Input` will be created for
   each string, with the string defining the input's name.


.. attribute:: SpatialWrench.wrench

   A mixed tuple of numbers and inputs that define the wrench.

   *(read-only)*


.. attribute:: SpatialWrench.wrench_val

   A tuple of the current numeric values of the wrench.

.. attribute:: SpatialWrench.frame

   The frame that this wrench is applied to.

   *(read-only)*
