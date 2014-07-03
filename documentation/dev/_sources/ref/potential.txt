.. _trep_potential:

:class:`Potential` -- Base Class for Potential Energies
=======================================================

.. currentmodule:: trep

.. class:: Potential(system[, name=None])

   :param system: An instance of :class:`System` to add the potential to.
   :type system: :class:`System`
   :param name: A string that uniquely identifies the potential energy.

   This is the base class for all potential energies in a
   :class:`System`.  It should never be created directly.  Potential
   energies are created by instantiating a specific type of potential
   energy.  

   See :ref:`builtin_potential_energies` for the built-in types of
   potential energy.  :ref:`creating_potential_energies` describes how
   to create new potential energies in Python or C.

   Potential energies represent conservative forces in a mechanical
   system like gravity and springs.  Implementing these forces as
   potentials energies instead of generalized forces will result in
   improved simulations with better energetic and momentum conserving
   properties.

Potential Objects
-----------------

.. attribute:: Potential.system
               
   The :class:`System` that this potential belongs to.

   *(read-only)*

.. attribute:: Potential.name

   The name of this potential energy or :data:`None`.

.. method:: Potential.V()

   :rtype: :class:`Float`

   Return the value of this potential energy at the system's current
   state.  This function should be implemented by derived Potentials.

   .. table:: **Required for Calculations**

      ===================    ========
      Desired Calculation    Required
      ===================    ========
      Continuous Dynamics      n
      1st Derivative           n
      2nd Derivative           n
      Discrete Dynamics        n
      1st Derivative           n
      2nd Derivative           n
      ===================    ========

   .. note:: 
      
      This actual potential value is not used in discrete or
      continuous time dynamics/derivatives, so you do not need to
      implement it unless you need it for your own calculations.
      However, implementing it allows one to compare the derivative
      :meth:`V_dq` with a numeric approximation based on :meth:`V` to
      help debug your potential.

.. method:: Potential.V_dq(q1)
           
   :param q1: Derivative variable
   :type q1: :class:`Config`
   :rtype: :class:`Float`

   Return the derivative of V with respect to *q1*.

   .. table:: **Required for Calculations**

      ===================    ========
      Desired Calculation    Required
      ===================    ========
      Continuous Dynamics      Y
      1st Derivative           Y
      2nd Derivative           Y
      Discrete Dynamics        Y
      1st Derivative           Y
      2nd Derivative           Y
      ===================    ========

.. method:: Potential.V_dqdq(q1, q2)
           
   :param q1: Derivative variable
   :type q1: :class:`Config`
   :param q2: Derivative variable
   :type q2: :class:`Config`
   :rtype: :class:`Float`

   Return the second derivative of V with respect to *q1* and *q2*.

   .. table:: **Required for Calculations**

      ===================    ========
      Desired Calculation    Required
      ===================    ========
      Continuous Dynamics      n
      1st Derivative           Y
      2nd Derivative           Y
      Discrete Dynamics        Y
      1st Derivative           Y
      2nd Derivative           Y
      ===================    ========
   
.. method:: Potential.V_dqdqdq(q1, q2, q3)
           
   :param q1: Derivative variable
   :type q1: :class:`Config`
   :param q2: Derivative variable
   :type q2: :class:`Config`
   :param q3: Derivative variable
   :type q3: :class:`Config`
   :rtype: :class:`Float`

   Return the third derivative of V with respect to *q1*, *q2*, and
   *q3*.

   .. table:: **Required for Calculations**

      ===================    ========
      Desired Calculation    Required
      ===================    ========
      Continuous Dynamics      n
      1st Derivative           n
      2nd Derivative           Y
      Discrete Dynamics        n
      1st Derivative           n
      2nd Derivative           Y
      ===================    ========


Verifying Derivatives of the Potential
--------------------------------------

It is important that the derivatives of :meth:`V` are correct.  The
easiest way to check their correctness is to approximate each
derivative using numeric differentiation.  These methods are provided
to perform this test.  The derivatives are only compared at the
current configuration of the system.  For improved coverage, try
running each test several times at different configurations.

.. method:: Potential.validate_V_dq(delta=1e-6, tolerance=1e-6, verbose=False)
            Potential.validate_V_dqdq(delta=1e-6, tolerance=1e-6, verbose=False)
            Potential.validate_V_dqdqdq(delta=1e-6, tolerance=1e-6, verbose=False)

   :param delta: Amount to add to each configuration 
   :param tolerance: Acceptable difference between the calculated and
                     approximate derivatives
   :param verbose: Boolean to print error and result messages.
   :rtype: Boolean indicating if all tests passed

   Check the derivatives against the approximate numeric derivative
   calculated from one less derivative (i.e,, approximate :meth:`V_dq`
   from :meth:`V` and :meth:`V_dqdq` from :meth:`V_dq`).  

   See :meth:`System.test_derivative_dq` for details of the
   approximation and comparison.


Visualization
-------------

.. method:: Potential.opengl_draw()

   Draw a representation of this potential energy in the current
   OpenGL context.  The OpenGL coordinate frame will be in the
   System's root coordinate frame.
   
   This function is called by the automatic visualization tools.  The
   default implementation does nothing.  

