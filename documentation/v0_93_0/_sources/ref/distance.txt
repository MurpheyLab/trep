:class:`Distance` - Maintain a specific distance between points
===============================================================

.. currentmodule:: trep.constraints

The :class:`Distance` constraint maintains a specific distance between
the origins of two coordinate frames.  The distance can be a fixed
number or controlled by a kinematic configuration variable.

The constraint equation is:

.. math::
   
   d^2 = (p_1 - p_2)^T (p_1 - p_2)

where :math:`p_1` and :math:`p_2` are the origins of two coordinate
frames and :math:`d` is the desired distance between them.

.. warning::

   This constraint is undefined when :math:`d` = 0.  If you want to
   constraint two points to be coincident, you can use three
   :class:`PointOnPlane` constraints instead.

.. admonition:: Examples

   ``puppet-basic.py``, ``puppet-continuous-moving.py``, ``puppet-moving.py``


.. class:: Distance(system, frame1, frame2, distance, name=None)

   Create a new constraint to maintain the distance between *frame1*
   and *frame2*.

   *frame1* and *frame2* should be existing coordinate frames in
   *system*.  They can be the :class:`Frame` objects themselves or
   their names.

   *distance* can be a real number or a string.  If it is a string,
   the constraint will create a new kinematic configuration variable
   with that name.


.. attribute:: Distance.config

   This is the kinematic configuration variable that controls the
   distance, or :data:`None` for a fixed distance constraint.

   *(read-only)*


.. attribute:: Distance.frame1
               Distance.frame2

   These are the two :class:`Frame` objects being constrained.

   *(read-only)*


.. attribute:: Distance.distance

   This is the *desired* constraint between the two coordinate frames.
   This is either the fixed value or the value of the configuration
   variable.  If you set the distance, the appropriate value will be
   updated.


.. method:: Distance.get_actual_distance()

   Calculate the current distance between the two coordinate frames.
   If the constraint is currently satisfied, this is equal to
   :attr:`distance`.

   If you have the system in a configuration you like, you can use
   this to set the correct distance::

     >>> constraint.distance = constraint.get_actual_distance()



