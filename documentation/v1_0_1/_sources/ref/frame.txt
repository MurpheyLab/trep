.. _trep_frame:

:class:`Frame` -- Coordinate Frame
==================================

.. currentmodule:: trep

The basic geometry of a mechanical system is defined by tree of
coordinate frames in :mod:`trep`.  The root of the tree is the fixed
:attr:`System.world_frame`.  Every other coordinate frame is defined by
a coordinate transformation relative to its parent.  The coordinate
transformations are either fixed or parameterized by a single
configuration variable.  For example, a rotational joint is modeled as
a rotation transformation where the angle is controlled by a
configuration variable.

Coordinate frames also define the masses in the system.  For every
mass, a coordinate frame must be placed with the origin at the center
of mass and the axes aligned with the principle axes of the rotational
inertia.  Coordinate frames can also be mass-less, or have a mass but
no rotational inertia to model point masses.

A :mod:`Frame` object can calculate its global position and body
velocity, and their derivatives.

.. contents::
   :local:


.. _transform_types:

Frame Transformation Types
--------------------------

There are a fixed set of coordinate transformations to define each
frame:

===================  ===========
Constant             Description
===================  ===========
.. data:: RX         Rotation about the parent's X axis.
.. data:: RY         Rotation about the parent's Y axis.
.. data:: RZ         Rotation about the parent's Z axis.
.. data:: TX         Translation about the parent's X axis.
.. data:: TY         Translation about the parent's Y axis.
.. data:: TZ         Translation about the parent's Z axis.
.. data:: CONST_SE3  Constant SE(3) transformation from parent.
.. data:: WORLD      Unique World Frame.
===================  ===========

The first six transformations can either be parameterized by a fixed
constant or a configuration variable.  The :data:`CONST_SE3`
transformation can only be fixed.

The :data:`WORLD` transformation is reserved for the system's
:attr:`world_frame <System.world_frame>`. 


.. _frame_definitions:

Defining the Frames
-------------------

New coordinate frames can be directly defined using the :class:`Frame`
constructor.  For example, this will create a simple pendulum:

.. _frame_pendulum_example1:

.. code-block:: python

  >>> import trep
  >>> system = trep.System()
  >>> frame1 = trep.Frame(system.world_frame, trep.RX, "theta")
  >>> frame2 = trep.Frame(frame1, trep.TZ, -1, mass=4)

This gets tedious and makes it difficult to see the mechanical
structure, so :mod:`trep` provides an alternate method to declare
frames using :meth:`Frame.import_frames` and a few extra functions:

.. function:: tx(param, name=None, kinematic=False, mass=0.0)
              ty(param, name=None, kinematic=False, mass=0.0)
              tz(param, name=None, kinematic=False, mass=0.0)
              rx(param, name=None, kinematic=False, mass=0.0)
              ry(param, name=None, kinematic=False, mass=0.0)
              rz(param, name=None, kinematic=False, mass=0.0)
              const_se3(se3, name=None, kinematic=False, mass=0.0)
              const_txyz(xyz, name, kinematic, mass)

The parameters are the same as for creating new :class:`Frame` objects
directly, except the parent and transform type are gone.  The
transform type is implied by the function name.  The parent will be
implied by how we use the definition.

:meth:`Frame.import_frames` expects a list of these definitions.  For
each definition, the frame will create a new frame and add it to it's
children.

For example, suppose we had a frame with 6 children::

  >>> child1 = Frame(parent, trep.TX, 'q1')
  >>> child2 = Frame(parent, trep.TY, 'q2', name='favorite_child')
  >>> child3 = Frame(parent, trep.TZ, 'q3')
  >>> child4 = Frame(parent, trep.RX, 'q4')
  >>> child5 = Frame(parent, trep.TX, 'q5')

Using :meth:`Frame.import_frames`, this becomes::

  >>> children = [
  ...     trep.tx('q1'),
  ...     trep.ty('q2', name='favorite_child'),
  ...     trep.tz('q3'),
  ...     trep.rx('q4'),
  ...     trep.tx('q5')
  ...     ]
  >>> parent.import_frames(children)

We can also add children to these children.  If
:meth:`Frame.import_frames` finds a list after a frame definition,
then it will call the new call the new child's :meth:`import_frames`
method with the new list.  

For example, the :ref:`pendulum <frame_pendulum_example1>` we created
earlier, can be defined as:

  >>> import trep
  >>> from trep import rx, tz
  >>> children = [
  ...     rx('theta'), [
  ...             tz(-1, mass=4)
  ...     ]]
  >>> system.world_frame.import_frames(children)

Since :meth:`Frame.import_frames` works recursively, we can describe
arbitrarily complex trees.  Consider this more complicated example:

.. image:: fig-pccd-relabeled.png

In the above image, the system is entirely 2D, and we are looking at the *x-z*
plane of the *World* frame.  Thus the arrows of each labeled coordinate frame
are showing the positive *x* and positive *z* axes of that frame. The
corresponding frame definition is::

   import trep
   from trep import tx, ty, tz, rx, ry, rz
   
   system = trep.System()
   frames = [
       ry('H', name='H'), [
           tz(-0.5, name='I', mass=1),
           tz(-1), [
               ry('J', name='J'), [
                   tz(-1, name='K', mass=1),
                   tz(-2, name='L')]]],
       tx(-1.5), [
           ry('M', name='M'), [
               tz(-1, name='N', mass=1),
               tz(-2), [
                   ry('O', name='O'), [
                       tz(-0.5, name='P', mass=1),
                       tz(-1.0, name='Q')]]]],
       tx(1.5), [
           ry('A', name='A'), [
               tz(-1, name='B', mass=1),
               tz(-2), [
                   ry('C', name='C'), [
                       tz(-0.375, name='D', mass=1),
                       tz(-0.75), [
                           ry('E', name='E'), [
                               tz(-0.5, name='F', mass=1),
                               tz(-1.0, name='G')
                               ]
                           ]
                       ]
                   ]
               ]
           ]
       ]
   system.import_frames(frames)
   
This is much more concise than defining the frames directly.  It is
also easier to see the structure of the system by taking advantage of
how most Python editors will indent the nested lists.  Trep also
provides the function :meth:`Frame.tree_view` for creating a visual
representation of a system's tree structure.  For the above system we
can view the tree with the following command::

   >>> print system.get_frame('World').tree_view()
   <Frame 'World'>
      <Frame 'H' RY(H)>
         <Frame 'I' TZ(-0.5) 1.000000>
         <Frame 'None' TZ(-1.0) 0.000000>
            <Frame 'J' RY(J)>
               <Frame 'K' TZ(-1.0) 1.000000>
               <Frame 'L' TZ(-2.0) 0.000000>
      <Frame 'None' TX(-1.5) 0.000000>
         <Frame 'M' RY(M)>
            <Frame 'N' TZ(-1.0) 1.000000>
            <Frame 'None' TZ(-2.0) 0.000000>
               <Frame 'O' RY(O)>
                  <Frame 'P' TZ(-0.5) 1.000000>
                  <Frame 'Q' TZ(-1.0) 0.000000>
      <Frame 'None' TX(1.5) 0.000000>
         <Frame 'A' RY(A)>
            <Frame 'B' TZ(-1.0) 1.000000>
            <Frame 'None' TZ(-2.0) 0.000000>
               <Frame 'C' RY(C)>
                  <Frame 'D' TZ(-0.375) 1.000000>
                  <Frame 'None' TZ(-0.75) 0.000000>
                     <Frame 'E' RY(E)>
                        <Frame 'F' TZ(-0.5) 1.000000>
                        <Frame 'G' TZ(-1.0) 0.000000>


A convenience function is also provided to create constant SE(3)
transformations from an angle and an axis.

.. function:: rotation_matrix(theta, axis)
    
   Build a 4x4 SE3 matrix corresponding to a rotation of theta
   radians around axis.



Frame Objects
-------------


.. class:: Frame(parent, transform, param[, name=None, kinematic=False, mass=0.0])
    

   Create a new coordinate frame attached to *parent*.  *transform*
   must be one of the :ref:`transformation constants
   <transform_types>` and defines how the new frame is related to the
   parent frame.   

   *param* is either a number or a string.  If *param* is an number,
   the frame is fixed relative to the parent.  
    
   If *param* is a string, the frame's coordinate transformation is
   controlled by a configuration variable.  A new configuration
   variable will be created using *param* as the name.  By default,
   the new configuration variable will be dynamic.  If *kinematic* is
   :data:`True`, it will be kinematic.

   If *transform* is :data:`CONST_SE`, *param* must be a 4x4 matrix
   that defines the Frame's constant SE(3) transformation relative to
   *parent*.

   *name* is an optional name for the frame.

   *mass* defines the inertial properties of the frame.  If *mass* is
   a single number, it is the frame's linear mass and the rotational
   inertia will be zero.

   *mass* can also be a list of 4 numbers that define the Frame's
   :attr:`mass`, :attr:`Ixx`, :attr:`Iyy`, and :attr:`Izz` inertial
   properties.

          


.. method:: Frame.uses_config(q)

   :param q: A configuration variable in the system.
   :type q: :class:`Config`

   Determine if this coordinate frame depends on the configuration
   variable *q*.

   When a frame does not depend on a configuration variable, the
   derivatives of its position and velocity will always be zero.  You
   can usually improve the performance of new constraints, potentials,
   and forces by checking for this and avoiding unnecessary
   calculations.


.. attribute:: Frame.system

   The :class:`System` that the frame belongs to.

   *(read only)*


.. attribute:: Frame.config

   The :class:`Config` that parameterizes the frame's transformation.
   This will be :data:`None` for fixed transformations.
        
   *(read only)*


.. attribute:: Frame.parent

   The parent frame of this frame.  This is always :data:`None` for
   the :attr:`System.world_frame`, and always a valid :data:`Frame`
   otherwise.

   *(read only)*


.. attribute:: Frame.children

   A tuple of the frame's child frames.

   *(read only)*


.. method:: Frame.flatten_tree()

   Create a list of the frame and its entire sub-tree.  There is no
   guarantee on the ordering other than it won't change as long as no
   frames are added to the system.

.. method:: Frame.tree_view(indent=0)
        
   Return a string that visually describes this frame and it's
   descendants.


Importing and Exporting Frames
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^     
                
.. method:: Frame.import_frames(children)
        
   Import a tree of frames from a tree description.  See
   :ref:`frame_definitions`.  The tree will be added to this frame's
   children.

.. method:: Frame.export_frames(tabs=0, tab_size=4)
        
   Create python source code to define this frame and it's sub-tree.
   The code is returned as a string.


Transform Information
^^^^^^^^^^^^^^^^^^^^^

.. attribute:: Frame.transform_type
   
   Transformation type of the coordinate frame.  This will be one of
   the constants described in :ref:`transform_types`.

   *(read only)*

.. attribute:: Frame.transform_value
        
   Current value of the frame's transformation parameters.  This
   will either be the fixed transformation parameter or the value
   of the frame's configuration variable.   

.. method:: Frame.set_SE3(Rx=(1,0,0), Ry=(0,1,0), Rz=(0,0,1), p=(0,0,0))
        
   Set the SE3 transformation for a const_SE3 frame.
                

Inertial Properties
^^^^^^^^^^^^^^^^^^^
    
.. attribute:: Frame.mass
               Frame.Ixx
               Frame.Iyy
               Frame.Izz

.. method:: Frame.set_mass(mass, Ixx=0.0, Iyy=0.0, Izz=0.0)
        
   The coordinate frame can have mass at its origin and rotational
   inertia about each axis.  


Local Frame Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: Frame.lg()
            Frame.lg_dq()
            Frame.lg_dqdq()
            Frame.lg_dqdqdq()
            Frame.lg_dqdqdqdq()
   
   These functions calculate the coordinate transformation to the
   frame from its parent in SE(3) and the derivatives of coordinate
   transformation with respect to the frame's configuration variable.
   If the frame is fixed, the derivatives will be zero.  The returned
   values are 4x4 :mod:`numpy` arrays.
     
.. method:: Frame.lg_inv()
            Frame.lg_inv_dq()
            Frame.lg_inv_dqdq()
            Frame.lg_inv_dqdqdq()
            Frame.lg_inv_dqdqdqdq()

   These functions calculate the inverse coordinate transformation to
   the frame from its parent (the transformation from the frame to its
   parent.) and its derivatives.
   
        
Global Frame Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         
.. method:: Frame.g()
            Frame.g_dq(q1)
            Frame.g_dqdq(q1, q2)
            Frame.g_dqdqdq(q1, q2, q3)
            Frame.g_dqdqdqdq(q1, q2, q3, q4)

   These functions calculate the global coordinate transformation in
   SE(3) for the frame (i.e, the coordinate transformation from the
   world frame to this frame) and its derivatives with respect to
   arbitrary configuration variables.  The returned values are 4x4
   :mod:`numpy` arrays.
        

.. method:: Frame.g_inv()
            Frame.g_inv_dq(q1)
            Frame.g_inv_dqdq(q1, q2)
        
   These functions calculate the inverse of the global coordinate
   transformation in SE(3) for the frame (i.e, the coordinate
   transformation from this frame to the world frame) and its
   derivatives with respect to arbitrary configuration variables.  The
   returned values are 4x4 :mod:`numpy` arrays.

.. method:: Frame.p()
            Frame.p_dq(q1)
            Frame.p_dqdq(q1, q2)
            Frame.p_dqdqdq(q1, q2, q3)
            Frame.p_dqdqdqdq(q1, q2, q3, q4)
        
   These functions calculate the global position of the coordinate
   frame in R^3 for the frame (i.e, the origin's location with respect
   to the world frame to this frame) and its derivatives with respect
   to arbitrary configuration variables.  The returned values are 4x4
   :mod:`numpy` arrays.


Body Velocity Calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: Frame.twist_hat()
.. method:: Frame.vb()

   Calculate the twist and the body velocity of the coordinate frame
   in se(3).  The returned values are 4x4 :mod:`numpy` arrays.
        
.. method:: Frame.vb_dq(q1)
            Frame.vb_ddq(dq1)

   Calculate first derivative of the body velocity with respect to the
   value or velocity of a configuration variable.  The returned values
   are 4x4 :mod:`numpy` arrays.

.. method:: Frame.vb_dqdq(q1, q2)
            Frame.vb_dqdqdq(q1, q2, q3)
        
   Calculate second derivative of the body velocity with respect to the
   values of configuration variables.  The returned values are 4x4
   :mod:`numpy` arrays.

.. method:: Frame.vb_ddqdq(dq1, q2)
            Frame.vb_ddqdqdq(dq1, q2, q3)
            Frame.vb_ddqdqdqdq(dq1, q2, q3, q4)
        
   Calculate derivative of the body velocity with respect to the
   velocity of *dq1* and the values of the other configuration
   variables.  The returned values are 4x4 :mod:`numpy` arrays.
        

