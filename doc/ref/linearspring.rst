:class:`LinearSpring` -- Linear spring between two points
=========================================================

.. currentmodule:: trep.potentials


:class:`LinearSpring` creates a spring force between the origins of
two coordinate frames in 3D space:

.. math::

   x = ||p_1 - p_2||

   V(q) = -k(x - x0)^2


where :math:`p_1` and :math:`p_2` are the origins of two coordinate
frames, :math:`k` is the spring stiffness and :math:`q_0` is the
natural length of the spring.


   .. table:: **Implemented Calculations**

      ===========   ===========
      Calculation   Implemented
      ===========   ===========
      V                 Y
      V_dq              Y
      V_dqdq            Y
      V_dqdqdq          Y 
      ===========   ===========

.. warning::
   
   The current implementation will fail if :math:`p_1` equals
   :math:`p_2` because of a divide by zero problem.  This problem will
   be corrected in the future for cases when :math:`x_0` is zero, but
   for now it should be avoided.  

   However, it cannot be corrected for the cases where :math:`x_0` is
   not zero.  If the two points are equal but :math:`x_0` is not zero,
   there should be a force.  But since there is no vector between the
   two points, the direction of this force is undefined.  When the
   natural length is zero, this problem can be corrected because the
   force also goes to zero.

Examples
--------

We can create a simple 1D harmonic oscillator using
:class:`LinearSpring` with a frame that is free to translate:

.. literalinclude:: code_snippets/linearspring-ex1.py

The :class:`LinearSpring` works between arbitrary frames, not just
frames connected by a translation.  Here, we create two pendulums and
connect their masses with a spring:

.. literalinclude:: code_snippets/linearspring-ex2.py

LinearSpring Objects
--------------------

.. class:: LinearSpring(system, frame1, frame2, k[, x0=0.0, name=None])

   Create a new spring between *frame1* and *frame2*.  The frames must
   already exist in the system.

.. attribute:: LinearSpring.frame1
               
   The coordinate frame at one end of the spring.

   *(read only)*

.. attribute:: LinearSpring.frame1
               
   The coordinate frame at the other end of the spring.

   *(read only)*

.. attribute:: LinearSpring.x0

   The natural length of the spring.  
                   
.. attribute:: LinearSpring.k

   The spring constant of the spring.




