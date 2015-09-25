:class:`Constraint` -- Base Class for Holonomic Constraints
===========================================================

.. currentmodule:: trep

.. class:: Constraint(system[, name=None, tolerance=1e-10])

   :param system: An instance of :class:`System` to add the constraint
                  to.
   :type system: :class:`System`
   :param name: A string that uniquely identifies the constraint.
   :param tolerance: Tolerance to consider the constraint satisfied.
   :type tolerance: :class:`Float`

   This is the base class for all holonomic constraints in a
   :class:`System`.  It should never be created directly.  Constraints
   are created by instantiating a specific type of constraint.

   See :ref:`builtin_constraints` for the built-in types of
   constraints. Additional constraints can be added through either the 
   Python or C-API.

   Holonomic constraints restrict the allowable configurations of a
   mechanical system.  Every constraint has an associated constraint
   function :math:`h(q) : Q \rightarrow R`.  A configuration :math:`q`
   is acceptable if and only if :math:`h(q) = 0`.


Constraint Objects
------------------

.. attribute:: Constraint.system
               
   The :class:`System` that this constraint belongs to.

   *(read-only)*

.. attribute:: Constraint.name

   The name of this constraint or :data:`None`.


.. attribute:: Constraint.index

   The index of the constraint in :attr:`System.constraints`.  This is
   also the index of the constraint's force in any values of
   :math:`\lambda` or its derivatives used through :mod:`trep`.

   *(read-only)*

.. attribute:: Constraint.tolerance

   The constraint should be considered satisfied if :math:`|h(q)| <
   tolerance`.  This is primarly used by the variational integrator
   when it finds the next configuration, or by
   :meth:`System.satisfy_constraints()`.


.. method:: Constraint.h()

   :rtype: :class:`Float`

   Return the value of the constraint function at the system's current
   state.  This function should be implemented by derived Constraints.


.. method:: Constraint.h_dq(q1)
           
   :param q1: Derivative variable
   :type q1: :class:`Config`
   :rtype: :class:`Float`

   Return the derivative of h with respect to *q1*.


.. method:: Constraint.h_dqdq(q1, q2)
           
   :param q1: Derivative variable
   :type q1: :class:`Config`
   :param q2: Derivative variable
   :type q2: :class:`Config`
   :rtype: :class:`Float`

   Return the second derivative of h with respect to *q1* and *q2*.

   
.. method:: Constraint.h_dqdqdq(q1, q2, q3)
           
   :param q1: Derivative variable
   :type q1: :class:`Config`
   :param q2: Derivative variable
   :type q2: :class:`Config`
   :param q3: Derivative variable
   :type q3: :class:`Config`
   :rtype: :class:`Float`

   Return the third derivative of h with respect to *q1*, *q2*, and
   *q3*.



.. method:: Constraint.h_dqdqdqdq(q1, q2, q3, q4)
           
   :param q1: Derivative variable
   :type q1: :class:`Config`
   :param q2: Derivative variable
   :type q2: :class:`Config`
   :param q3: Derivative variable
   :type q3: :class:`Config`
   :param q4: Derivative variable
   :type q4: :class:`Config`
   :rtype: :class:`Float`

   Return the fourth derivative of h with respect to *q1*, *q2*, *q3*,
   and *q4*.



Verifying Derivatives of the Constraint
---------------------------------------

It is important that the derivatives of :meth:`h` are correct.  The
easiest way to check their correctness is to approximate each
derivative using numeric differentiation.  These methods are provided
to perform this test.  The derivatives are only compared at the
current configuration of the system.  For improved coverage, try
running each test several times at different configurations.

.. method:: Constraint.validate_h_dq(delta=1e-6, tolerance=1e-6, verbose=False)
            Constraint.validate_h_dqdq(delta=1e-6, tolerance=1e-6, verbose=False)
            Constraint.validate_h_dqdqdq(delta=1e-6, tolerance=1e-6, verbose=False)
            Constraint.validate_h_dqdqdqdq(delta=1e-6, tolerance=1e-6, verbose=False)

   :param delta: Amount to add to each configuration 
   :param tolerance: Acceptable difference between the calculated and
                     approximate derivatives
   :param verbose: Boolean to print error and result messages.
   :rtype: Boolean indicating if all tests passed

   Check the derivatives against the approximate numeric derivative
   calculated from one less derivative (ie, approximate :meth:`h_dq`
   from :meth:`h` and :meth:`h_dqdq` from :meth:`h_dq`).  

   See :meth:`System.test_derivative_dq` for details of the
   approximation and comparison.


Visualization
-------------

.. method:: Constraint.opengl_draw()

   Draw a representation of this constraint in the current OpenGL
   context.  The OpenGL coordinate frame will be in the System's root
   coordinate frame.
   
   This function is called by the automatic visualization tools.  The
   default implementation does nothing.  

