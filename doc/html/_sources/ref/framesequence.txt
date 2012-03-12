:class:`FrameSequence` -- Measuring distances between frames
============================================================

.. currentmodule:: trep

.. class:: FrameSequence(system, frames)

   :param system: The :class:`System` that the frames belong to.
   :param frames: A list of :class:`Frame` objects or frame names.

   A :class:`FrameSequence` object calculates the length of the line
   you get from playing "connect the dots" with the origins of a list
   of coordinate frames.  :class:`FrameSequence` can calculate the
   length of the line and its derivatives with respect to
   configuration variables, and the velocity of the length
   (:math:`\tfrac{dx}{dt}`) and its derivatives.

   *(figure here)*

   :class:`FrameSequence` can be used as the basis for new
   constraints, potentials, and forces, or used independently for your
   own calculations.

Length and Velocity Calculations
--------------------------------

   Let :math:`(p_0, p_1, p_2 \dots p_{n} )` be the points at the origins
   of the frames specified by :attr:`frames`.  The length, :math:`x` is
   calculated as follows.

   .. math::
   
      v_k = p_{k+1} - p_k

      x_k = \sqrt{v_k^T v_k}

      x = \sum_{k=0}^{n-1} x_k

   The velocity is calculated by applying the chain rule to :math:`x`:

   .. math::

      \dot{x} = \sum_k \sum_i \frac{\partial x_k}{\partial q_i} \dot{q}_i

   These calculations, and their derivatives, are optimized internally
   to take advantage of the fact that many of these terms are zero,
   significantly reducing the amount of calculation to do.

   See the :ref:`internal documentation <internal_ref_FrameSequence>`
   on :class:`FrameSequence` for more details about the implementation
   and for the equations to calculate the derivatives.
    
   .. warning::

      The derivatives of the length and velocity do not exist when the
      length of any part of the segment is zero.
      :class:`FrameSequence` does not check for this condition and
      will return :data:`NaN` or cause a divide-by-zero error.  Be
      careful to avoid these cases.

FrameSequence Objects
---------------------

.. attribute:: FrameSequence.system

   The system that the :class:`FrameSequence` works in.

   *(read-only)*

.. attribute:: FrameSequence.frames

   A tuple of :class:`Frame` objects that define the lines being
   measured.

   *(read-only)*

.. method:: FrameSequence.length()
            
   :rtype: :class:`Float`

   Calculate the total length of the line segments at the system's
   current configuration.

.. method:: FrameSequence.length_dq(q1)

   :param q1: Derivative variable
   :type q1: :class:`Config`
   :rtype: :class:`Float`

   Calculate the derivative of the length with respect to the value of
   *q1*.

.. method:: FrameSequence.length_dqdq(q1, q2)

   :param q1: Derivative variable
   :type q1: :class:`Config`
   :param q2: Derivative variable
   :type q2: :class:`Config`
   :rtype: :class:`Float`

   Calculate the second derivative of the length with respect to the
   value of *q1* and the value of *q2*.

.. method:: FrameSequence.length_dqdqdq(q1, q2, q3)

   :param q1: Derivative variable
   :type q1: :class:`Config`
   :param q2: Derivative variable
   :type q2: :class:`Config`
   :param q3: Derivative variable
   :type q3: :class:`Config`
   :rtype: :class:`Float`

   Calculate the third derivative of the length with respect to the
   value of *q1*, the value of *q2*, and the value of *q3*.

.. method:: FrameSequence.velocity()

   :rtype: :class:`Float`

.. method:: FrameSequence.velocity_dq(q1)

   :param q1: Derivative variable
   :type q1: :class:`Config`
   :rtype: :class:`Float`

   Calculate the derivative of the velocity with respect to the value
   of *q1*.

.. method:: FrameSequence.velocity_ddq(dq1)

   :param dq1: Derivative variable
   :type dq1: :class:`Config`
   :rtype: :class:`Float`

   Calculate the derivative of the velocity with respect to the
   velocity of *q1*.

.. method:: FrameSequence.velocity_dqdq(q1, q2)

   :param q1: Derivative variable
   :type q1: :class:`Config`
   :param q2: Derivative variable
   :type q2: :class:`Config`
   :rtype: :class:`Float`

   Calculate the second derivative of the velocity with respect to the
   value of *q1* and the value of *q2*.

.. method:: FrameSequence.velocity_dddqdq(dq1, dq2)

   :param dq1: Derivative variable
   :type dq1: :class:`Config`
   :param q2: Derivative variable
   :type q2: :class:`Config`
   :rtype: :class:`Float`

   Calculate the second derivative of the velocity with respect to the
   velocity of *q1* and the value of *q2*.

Visualization
-------------

.. method:: FrameSequence.opengl_draw(width=1.0, color=(1.0,1.0,1.0))

   Draw a representation of the line defined by the
   :class:`FrameSequence` with the specified width and color.  The
   current OpenGL coordinate system should be the root coordinate
   frame of the :class:`System`.

   This method can be called by constraints, forces, and potentials
   that are based on the :class:`FrameSequence`.       



Verifying Derivatives
---------------------

.. method:: FrameSequence.validate_length_dq([delta=1e-6, tolerance=1e-6, verbose=False])
            FrameSequence.validate_length_dqdq([delta=1e-6, tolerance=1e-6, verbose=False])
            FrameSequence.validate_length_dqdqdq([delta=1e-6, tolerance=1e-6, verbose=False])
            FrameSequence.validate_velocity_dq([delta=1e-6, tolerance=1e-6, verbose=False])
            FrameSequence.validate_velocity_ddq([delta=1e-6, tolerance=1e-6, verbose=False])
            FrameSequence.validate_velocity_dqdq([delta=1e-6, tolerance=1e-6, verbose=False])
            FrameSequence.validate_velocity_ddqdq([delta=1e-6, tolerance=1e-6, verbose=False])

   Unlike :class:`Constraint`, :class:`Potential`, and :class:`Force`,
   :class:`FrameSequence` is used directly and all of the calculations
   are already implemented and verified.  These functions are
   primarily used for testing during developement, but they might be
   useful for debugging if you are using a :class:`FrameSequence` and
   having trouble.
