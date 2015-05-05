
:class:`PointOnPlane` - Constraint a point to a plane
=====================================================

.. currentmodule:: trep.constraints

The :class:`PointOnPlane` constraint constraints a point (defined by
the origin of a coordinate frame) to lie in a plane (defined by the
origin of another coordinate frame and normal vector).

The constraint equation is:

.. math::

   h(q) =  (p_1 - p_2) \cdot (g_1 \cdot norm)

where :math:`g_1` and :math:`p_1` are the transformation of the frame
defining the plane and its origin, :math:`p_2` is the point being
constrained, and :math:`norm` is the normal vector of the plane
expressed in the local coordinates of :math:`g_1`.

By defining two of :class:`PointOnPlane` constraints with normal
plane, you can constrain a point to a line.  With three constraints,
you can constraint a point to another point.



.. admonition:: Examples

   ``pccd.py``, ``pypccd.py``, ``radial.py``



.. class:: PointOnPlane(system, plane_frame, plane_normal, point_frame, name=None)

   Create a new constraint to force the origin of *point_frame* to lie
   in a plane.  The plane is coincident with the origin of
   *plane_frame* and normal to the vector *plane_normal*.  The normal
   is expresed in the coordinates of *plane_frame*.


.. attribute:: PointOnPlane.plane_frame
               
   This is the coordinate frame the defines the plane.

   *(read-only)*


.. attribute:: PointOnPlane.normal

   This is the normal of the plane expressed in *plane_frame* coordinates.


.. attribute:: PointOnPlane.point_frame

   This is the coordinate frame that defines the point to be
   constrained.
