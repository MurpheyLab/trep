
:class:`PointToPoint` - Constrain the origins of two frames together
=======================================================================

.. currentmodule:: trep.constraints

The :class:`PointToPoint` constraints are designed to constrain the origin
of one frame to the origin of another frame (maintaining zero distance between
the frames).

The constraint equation is:

.. math::

   h(q) =  (p_1 - p_2) [i]

where :math:`p_1` and :math:`p_2` are the origin points of the two frames being constrained, and :math:`i` is
the basis component of the error vector.  

.. note::

    Defining a PointToPoint constraint creates multiple one-dimensional constraints for each axis being used.  
    For example, if a system is defined in 3D, the PointToPoint3D method creates 3 PointToPoint constraints, one for each
    axis.


.. admonition:: Examples

   ``scissor.py``



.. class:: PointToPoint3D(system, frame1, frame2, name=None)

   :param frame1: First frame to constrain 
   :type frame1: :class:`Frame`
   :param frame2: Second frame to constrain 
   :type frame2: :class:`Frame`

   Create a new set of 3 constraints to force the origin of *frame1* to the origin of *frame2*.
   The system must be defined in 3D, otherwise, for 2D systems, use :class:`PointToPoint2D`.



.. class:: PointToPoint2D(system, plane, frame1, frame2, name=None)

   :param plane: 2D plane of system, ie. 'xy', 'xz', or 'yz'
   :type plane: :class:`string`
   :param frame1: First frame to constrain 
   :type frame1: :class:`Frame`
   :param frame2: Second frame to constrain 
   :type frame2: :class:`Frame`

   Create a new set of 2 constraints to force the origin of *frame1* to the origin of *frame2*.
   The system must be defined in 2D, otherwise, for 3D systems, use :class:`PointToPoint3D`.

.. class:: PointToPoint1D(system, axis, frame1, frame2, name=None)

   :param axis: 1D axis of system, ie. 'x', 'y', or 'z'
   :type axis: :class:`string`
   :param frame1: First frame to constrain 
   :type frame1: :class:`Frame`
   :param frame2: Second frame to constrain 
   :type frame2: :class:`Frame`

   Create a new constraint to force the origin of *frame1* to the origin of *frame2*.
   The system must be defined in 1D, otherwise, for 2D or 3D systems, use :class:`PointToPoint2D` or :class:`PointToPoint3D`.


.. method:: PointToPoint3D.get_actual_distance()
            PointToPoint2D.get_actual_distance()
            PointToPoint1D.get_actual_distance()

   Calculate the current distance between the two coordinate frames.
   If the constraint is currently satisfied, this is equal to 0.

