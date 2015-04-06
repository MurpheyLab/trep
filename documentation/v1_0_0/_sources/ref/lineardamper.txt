:class:`LinearDamper` -- Linear damper between two points
=========================================================

.. currentmodule:: trep.forces


:class:`LinearDamper` creates a damping force between the origins of
two coordinate frames in 3D space:

We can derive a mapping from the damper force to a force in the
generalized coordinates by looking at the work done by the damper.
The virtual work done by a damper is simply the damper force
multiplied by the change in the damper's length (distance):

.. math::

   x = ||p_1 - p_2||

   f = -c \dot x

   F_{q_i} = f \deriv[x]{q_i}


where :math:`p_1` and :math:`p_2` are the origins of two coordinate
frames and :math:`c` is the damper coefficient.


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

   ``dual_pendulums.py``


.. class:: LinearDamper(system, frame1, frame2, c[, name=None])

   Create a new damper between *frame1* and *frame2*.  The frames must
   already exist in the system.

.. attribute:: LinearDamper.frame1
               
   The coordinate frame at one end of the damper.

   *(read only)*

.. attribute:: LinearDamper.frame1
               
   The coordinate frame at the other end of the damper.

   *(read only)*

                   
.. attribute:: LinearDamper.c

   The damping coefficient of the damper.




