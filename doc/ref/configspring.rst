:class:`ConfigSpring` -- Linear Spring acting on a configuration variable
=========================================================================

.. currentmodule:: trep

Unlike the :class:`LinearSpring` which creates a spring force between
two points in 3D space, a :class:`ConfigSpring` implements a spring
force directly on a generalized coordinate:

.. math::
   F_{q} = -k (q - q_0)

where :math:`k` is the spring stiffness and :math:`q_0` is the neutral
"length" of the spring.  


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

We can create a simple 1D harmonic oscillator using
:class:`ConfigSpring` on a translational configuration variable.  The
same oscillator could be created using a :class:`LinearSpring` as
well.

.. literalinclude:: code_snippets/configspring-ex1.py

A :class:`ConfigSpring` can be used to create a torsional spring on a
rotational configuration variable.  Here we create a pendulum with a
length of 2 and mass of 1.  Instead of gravity, the pendulum is moved
by a single torsional spring.

.. literalinclude:: code_snippets/configspring-ex2.py

ConfigSpring Objects
--------------------

.. class:: ConfigSpring(system, config, k[, q0=0.0, name=None])

   Create a new spring acting on the specified configuration variable.
   The configuration variable must already exist in the system.

.. attribute:: ConfigSpring.config

   The configuration variable that spring depends and acts on.

   *(read only)*

.. attribute:: ConfigSpring.q0

   The "neutral length" of the spring.  When ``self.config.q == q0``,
   the force is zero.
                   
.. attribute:: ConfigSpring.k

   The spring constant of the spring.


