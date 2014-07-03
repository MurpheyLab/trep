:class:`Spline` -- Spline Objects
=================================

.. currentmodule:: trep

.. class:: Spline(points)

   :param points: A list of points defining the spline.  See details.


The :class:`Spline` class implements 1D `spline interpolation
<http://en.wikipedia.org/wiki/Spline_interpolation>`_ between a set of
points.  :class:`Spline` provides methods to evaluate points on the
spline curve as well as its first and second derivatives.  It can be
used by new types of forces, potentials, and constraints to implement
calculations.

The spline is defined by the list *points*.  Each entry in *points* is
a list of 2-4 numbers.  The first two numbers are the *x* and *y*
values of the points.  The remaining two numbers, if provided and not
:data:`None`, are the first and second derivatives of the curve at
that point.  Any derivatives not provided are determined by the
resulting interpolation.

For *N* points, The spline will comprise *N-1* polynomials of order
3-5, depending on how many derivatives were specified. :class:`Spline`
will choose the lowest order polynomials possible while still being
able to satisfy the specified values.  Specifying derivatives directly
is much more effective than placing several points infinitesimally
close to force the curve into a particular shape.

See the :ref:`internal documentation <internal_ref_Spline>` on
:class:`Spline` for more details about the implementation and how the
interpolating polynomials are found.


Spline Objects
--------------

.. attribute:: Spline.x_points

   :rtype: :class:`numpy.ndarray`

   List of the x points that define this spline.

   *(read-only)*


.. attribute:: Spline.y_points

   :rtype: :class:`numpy.ndarray`

   List of the y points that define this spline.

   *(read-only)*


.. attribute:: Spline.coefficients

   :rtype: :class:`numpy.ndarray`

   The coefficients of the interpolating polynomials.

   *(read-only)*

.. method:: Spline.copy()

   :rtype: :class:`Spline`

   Create a new copy of this :class:`Spline`.

.. method:: Spline.y(x)

   :type x: :class:`Float`
   :rtype: :class:`Float`

   Evaluate the spline at *x*.

.. method:: Spline.dy(x)

   :type x: :class:`Float`
   :rtype: :class:`Float`

   Evaluate the derivative of the spline at *x*.

.. method:: Spline.ddy(x)

   :type x: :class:`Float`
   :rtype: :class:`Float`

   Evaluate the second derivative of the spline at *x*.


