:class:`Gravity` -- Basic Gravity
=================================

.. currentmodule:: trep.potentials

.. class:: Gravity(system[, gravity=(0.0, 0.0, -9.8), name=None])

   (*Inherits from* :class:`Potential`)

   :param system: An instance of :class:`System` to add the gravity
                  to.
   :type system: :class:`System`
   :param gravity: The gravity vector
   :type gravity: Sequence of three :class:`Float`
   :param name: A string that uniquely identifies the gravity.


   :class:`Gravity` implements a basic constant acceleration gravity
   (:math:`F = m\vec{g}`).


   .. table:: **Implemented Calculations**

      ===========   ===========
      Calculation   Implemented
      ===========   ===========
      V                 Y
      V_dq              Y
      V_dqdq            Y
      V_dqdqdq          Y 
      ===========   ===========


Examples
--------

Adding gravity to a system is as easy as declaring an instance of
:class:`Gravity`::

   >>> system = build_custom_system()
   >>> trep.potentials.Gravity(system)
   <Gravity 0.000000 0.000000 -9.800000>

The :class:`System` saves a reference to the new :class:`Gravity`, so
we do not have to save a refence to prevent it from being garbage
collected.

The default gravity points in the negative *Z* direction.  We can
specify a new gravity vector when we add the gravity.  For example, we
can make gravity point in the positive *Y* direction::

   >>> system = build_custom_system()
   >>> trep.potentials.Gravity(system, (0, 9.8, 0))
   <Gravity 0.000000 9.800000 0.000000>

Gravity Objects
---------------

.. attribute:: Gravity.gravity

   The gravity vector for this instance of gravity.


Visualization
-------------

.. method:: Gravity.opengl_draw()

   :class:`Gravity` does not draw a visual representation.

