.. currentmodule:: trep

:class:`Damping` -- Damping
===========================

:class:`Damping` implements a damping :class:`Force` on the
generalized coordinates of a system:

.. math::
   
   F_{q_i} = -c_i\ \dq_i

where the damping constant :math:`c_i` is a positive real number.


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

