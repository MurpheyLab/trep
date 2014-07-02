.. _trep_force:

.. currentmodule:: trep

:class:`Force` -- Base Class for Forces
=======================================


.. class:: Force(system[, name=None])

   :param system: An instance of :class:`System` to add the force
                  to.
   :type system: :class:`System`
   :param name: A string that uniquely identifies the force.

   This is the base class for all forces in a :class:`System`.  It
   should never be created directly.  Forces are created by
   instantiating a specific type of force..

   See :ref:`builtin_forces` for the built-in types of
   forces. :ref:`creating_forces` describes how to create
   new forces in Python or C.

   Forces are used to include non-conservative and control forces in a
   mechanical system.  Forces must be expressed in the generalized
   coordinates of the system.  Conservative forces like gravity or
   spring-like potentials should be implemented using
   :class:`Potential` instead.

Force Objects
------------------

.. attribute:: Force.system
               
   The :class:`System` that this force belongs to.

   *(read-only)*

.. attribute:: Force.name

   The name of this force or :data:`None`.

.. method:: Force.f(q)

   :param q: Configuration variable 
   :type q: :class:`Config`

   Calculate the force on configuration variable *q* at the current
   state of the system.

   .. table:: **Required for Calculations**

      ===================    ========
      Desired Calculation    Required
      ===================    ========
      Continuous Dynamics      ?
      1st Derivative           ?
      2nd Derivative           ?
      Discrete Dynamics        ?
      1st Derivative           ?
      2nd Derivative           ?
      ===================    ========

.. method:: Force.f_dq(q, q1)

   :param q: Configuration variable 
   :type q: :class:`Config`
   :param q1: Configuration variable 
   :type q1: :class:`Config`

   Calculate the derivative of the force on configuration variable *q*
   with respect to the value of *q1*.

   .. table:: **Required for Calculations**

      ===================    ========
      Desired Calculation    Required
      ===================    ========
      Continuous Dynamics      ?
      1st Derivative           ?
      2nd Derivative           ?
      Discrete Dynamics        ?
      1st Derivative           ?
      2nd Derivative           ?
      ===================    ========


.. method:: Force.f_ddq(q, dq1)

   :param q: Configuration variable 
   :type q: :class:`Config`
   :param dq1: Configuration variable 
   :type dq1: :class:`Config`

   Calculate the derivative of the force on configuration variable *q*
   with respect to the velocity of *dq1*.

   .. table:: **Required for Calculations**

      ===================    ========
      Desired Calculation    Required
      ===================    ========
      Continuous Dynamics      ?
      1st Derivative           ?
      2nd Derivative           ?
      Discrete Dynamics        ?
      1st Derivative           ?
      2nd Derivative           ?
      ===================    ========


.. method:: Force.f_du(q, u1)

   :param q: Configuration variable 
   :type q: :class:`Config`
   :param u1: Input variable 
   :type u1: :class:`Input`

   Calculate the derivative of the force on configuration variable *q*
   with respect to the value of *u1*.

   .. table:: **Required for Calculations**

      ===================    ========
      Desired Calculation    Required
      ===================    ========
      Continuous Dynamics      ?
      1st Derivative           ?
      2nd Derivative           ?
      Discrete Dynamics        ?
      1st Derivative           ?
      2nd Derivative           ?
      ===================    ========

.. method:: Force.f_dqdq(q, q1, q2)

   :param q: Configuration variable 
   :type q: :class:`Config`
   :param q1: Configuration variable 
   :type q1: :class:`Config`
   :param q2: Configuration variable 
   :type q2: :class:`Config`

   Calculate the second derivative of the force on configuration
   variable *q* with respect to the value of *q1* and the value of
   *q2*.

   .. table:: **Required for Calculations**

      ===================    ========
      Desired Calculation    Required
      ===================    ========
      Continuous Dynamics      ?
      1st Derivative           ?
      2nd Derivative           ?
      Discrete Dynamics        ?
      1st Derivative           ?
      2nd Derivative           ?
      ===================    ========


.. method:: Force.f_ddqdq(q, dq1, q2)

   :param q: Configuration variable 
   :type q: :class:`Config`
   :param dq1: Configuration variable 
   :type dq1: :class:`Config`
   :param q2: Configuration variable 
   :type q2: :class:`Config`

   Calculate the second derivative of the force on configuration
   variable *q* with respect to the velocity of *dq1* and the value of
   *q2*.

   .. table:: **Required for Calculations**

      ===================    ========
      Desired Calculation    Required
      ===================    ========
      Continuous Dynamics      ?
      1st Derivative           ?
      2nd Derivative           ?
      Discrete Dynamics        ?
      1st Derivative           ?
      2nd Derivative           ?
      ===================    ========

.. method:: Force.f_ddqddq(q, dq1, dq2)

   :param q: Configuration variable 
   :type q: :class:`Config`
   :param dq1: Configuration variable 
   :type dq1: :class:`Config`
   :param dq2: Configuration variable 
   :type dq2: :class:`Config`

   Calculate the second derivative of the force on configuration
   variable *q* with respect to the velocity of *dq1* and the velocity of
   *q2*.

   .. table:: **Required for Calculations**

      ===================    ========
      Desired Calculation    Required
      ===================    ========
      Continuous Dynamics      ?
      1st Derivative           ?
      2nd Derivative           ?
      Discrete Dynamics        ?
      1st Derivative           ?
      2nd Derivative           ?
      ===================    ========


.. method:: Force.f_dudq(q, u1, q2)

   :param q: Configuration variable 
   :type q: :class:`Config`
   :param u1: Input variable 
   :type u1: :class:`Input`
   :param q2: Configuration variable 
   :type q2: :class:`Config`

   Calculate the second derivative of the force on configuration
   variable *q* with respect to the value of *u1* and the value of *q2*.

   .. table:: **Required for Calculations**

      ===================    ========
      Desired Calculation    Required
      ===================    ========
      Continuous Dynamics      ?
      1st Derivative           ?
      2nd Derivative           ?
      Discrete Dynamics        ?
      1st Derivative           ?
      2nd Derivative           ?
      ===================    ========


.. method:: Force.f_duddq(q, u1, dq2)

   :param q: Configuration variable 
   :type q: :class:`Config`
   :param u1: Input variable 
   :type u1: :class:`Input`
   :param dq2: Configuration variable 
   :type dq2: :class:`Config`

   Calculate the second derivative of the force on configuration
   variable *q* with respect to the value of *u1* and the velocity of
   *q2*.

   .. table:: **Required for Calculations**

      ===================    ========
      Desired Calculation    Required
      ===================    ========
      Continuous Dynamics      ?
      1st Derivative           ?
      2nd Derivative           ?
      Discrete Dynamics        ?
      1st Derivative           ?
      2nd Derivative           ?
      ===================    ========


.. method:: Force.f_dudu(q, u1, u2)

   :param q: Configuration variable 
   :type q: :class:`Config`
   :param u1: Input variable 
   :type u1: :class:`Input`
   :param u2: Input variable 
   :type u2: :class:`Input`

   Calculate the second derivative of the force on configuration
   variable *q* with respect to the value of *u1* and the value of
   *u2*.

   .. table:: **Required for Calculations**

      ===================    ========
      Desired Calculation    Required
      ===================    ========
      Continuous Dynamics      ?
      1st Derivative           ?
      2nd Derivative           ?
      Discrete Dynamics        ?
      1st Derivative           ?
      2nd Derivative           ?
      ===================    ========



Verifying Derivatives of the Force
---------------------------------------

It is important that the derivatives of :meth:`f` are correct.  The
easiest way to check their correctness is to approximate each
derivative using numeric differentiation.  These methods are provided
to perform this test.  The derivatives are only compared at the
current configuration of the system.  For improved coverage, try
running each test several times at different configurations.

.. method:: Force.validate_h_dq(delta=1e-6, tolerance=1e-6, verbose=False)
            Force.validate_h_ddq(delta=1e-6, tolerance=1e-6, verbose=False)
            Force.validate_h_du(delta=1e-6, tolerance=1e-6, verbose=False)
            Force.validate_h_dqdq(delta=1e-6, tolerance=1e-6, verbose=False)
            Force.validate_h_ddqdq(delta=1e-6, tolerance=1e-6, verbose=False)
            Force.validate_h_ddqddq(delta=1e-6, tolerance=1e-6, verbose=False)
            Force.validate_h_dudq(delta=1e-6, tolerance=1e-6, verbose=False)
            Force.validate_h_duddq(delta=1e-6, tolerance=1e-6, verbose=False)
            Force.validate_h_dudu(delta=1e-6, tolerance=1e-6, verbose=False)

   :param delta: Amount to add to each configuration, velocity, or
                 input.
   :param tolerance: Acceptable difference between the calculated and
                     approximate derivatives
   :param verbose: Boolean to print error and result messages.
   :rtype: Boolean indicating if all tests passed

   Check the derivatives against the approximate numeric derivative
   calculated from one less derivative (ie, approximate :meth:`f_dq`
   from :meth:`f` and :meth:`f_dudq` from :meth:`f_du`).  

   See :meth:`System.test_derivative_dq`,
   :meth:`System.test_derivative_ddq`, and
   :meth:`System.test_derivative_du` for details of the approximation
   and comparison.


Visualization
-------------

.. method:: Force.opengl_draw()

   Draw a representation of this force in the current OpenGL
   context.  The OpenGL coordinate frame will be in the System's root
   coordinate frame.
   
   This function is called by the automatic visualization tools.  The
   default implementation does nothing.  

