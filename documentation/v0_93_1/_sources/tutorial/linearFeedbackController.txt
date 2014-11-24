Design linear feedback controller
=================================
Let us now create a linear feedback controller that will stabilize the pendulum
system that was created in the last section to its upright, unstable equilibrium.

Create pendulum system
----------------------
Let us start by using the same code from the last section that creates a simple
pendulum system with the addition of just a few lines.

.. literalinclude:: ./code_snippets/linearFeedbackController.py
    :start-after: # linearFeedbackController.py
    :end-before: # Create discrete system
    :emphasize-lines: 5,7,9,18-20

Three additional (highlighted above) modules are imported: ``dot`` from the
``numpy`` module, which we will use for matrix multiplication; ``pylab`` (part
of the `matplotlib project <http://matplotlib.org/>`_, which is used for
plotting; and trep's discrete optimization module, :mod:`trep.discopt`.

In addition, a desired configuration value has been set to the variable
``qBar``. The desired configuration is :math:`\pi`, which is the pendulum's
neutrally-stable, upright configuration. Also, correctly-dimensioned, identity
state and input weighting matrices (``Q`` and ``R`` respectively) have been
created for the optimization of the linear feedback controller.

Create discrete system
----------------------
Trep has a module that provides functionality for solving several linear,
time-varying discrete linear-quadratic regulation problems (see :ref:`this page
<trep_dlqr>`).

.. literalinclude:: ./code_snippets/linearFeedbackController.py
    :start-after: # Create discrete system
    :end-before: # Design linear feedback controller

To do this, we use our system definition, and a prescribed time vector to create
a :class:`trep.discopt.DSystem`. This object is basically a wrapper for
:class:`trep.System` objects and :class:`trep.MidpointVI` objects to represent
the general nonlinear discrete systems as first order discrete nonlinear systems
of the form :math:`X(k+1) = f(X(k), U(k), k)`.

The states :math:`X` and inputs :math:`U` of a :class:`trep.discopt.DSystem`
should be noted. The state consists of the variational integrator’s full
configuration, the dynamic momentum, and the kinematic velocity. The full
configuration consist of both dynamic states and kinematic states. The
difference being that the dynamic states are governed by first-order
differential equations and the kinematic states can be set directly with
"kinematic" inputs. This is equivalent to saying you have "perfect" control over
a dynamic configuration i.e. your system is capable of exerting unlimited force
to drive the configuration to follow any arbitrary trajectory. The kinematic
velocity is calculated as a finite difference of the kinematic
configurations. The :class:`trep.discopt.DSystem` class has a method,
:mod:`trep.discopt.DSystem.build_state`, that will "build" this state vector from
configuration, momentum, and kinematic velocity vectors. "Build" here means
construct a state array of the correct dimension. Anything that is not specified
is set to zero. This is used above to calculated a desired state array from the
desired configuration value.

Using IPython you can display a list of all the properties and methods of the
discrete system.

    >>> dsys.
    dsys.build_input               dsys.fdu                       dsys.save_state_trajectory
    dsys.build_state               dsys.fdudu                     dsys.set
    dsys.build_trajectory          dsys.fdx                       dsys.split_input
    dsys.calc_feedback_controller  dsys.fdxdu                     dsys.split_state
    dsys.check_fdu                 dsys.fdxdx                     dsys.split_trajectory
    dsys.check_fdudu               dsys.k                         dsys.step
    dsys.check_fdx                 dsys.kf                        dsys.system
    dsys.check_fdxdu               dsys.linearize_trajectory      dsys.time
    dsys.check_fdxdx               dsys.load_state_trajectory     dsys.uk
    dsys.convert_trajectory        dsys.nU                        dsys.varint
    dsys.dproject                  dsys.nX                        dsys.xk
    dsys.f                         dsys.project 

Please refer to the :ref:`DSystem <trep_dsystem>` documentation to learn more
about this object.

Design linear feedback controller
---------------------------------
Let us now design a linear feedback controller to stabilize the pendulum to the
desired configuration.

.. literalinclude:: ./code_snippets/linearFeedbackController.py
    :start-after: # Design linear feedback controller
    :end-before: # Simulate the system forward

The ``DSystem`` class has a method
(:mod:`trep.discopt.DSystem.calc_feedback_controller`) for calculating a
stabilizing feedback controller for the system about state and input
trajectories given both state and input weighting functions.

A trajectory of the desired configuration is constructed and then used with the
:mod:`trep.discopt.DSystem.build_trajectory` method to create the desired state
and input trajectories. The weighting functions are created with Python lambda
functions that always output the state and input cost weights that were assigned
to ``Qk`` and ``Rk``. The ``calc_feedback_controller`` method returns the gain
value :math:`K` for each instance in time. To approximate the infinite-horizon
optimal controller only the first value is used.

The discrete system must be reset to the initial state value because during the
optimization the discrete system is integrated and thus set to the final
time. Note, because the discrete system was created from the variational
integrator, which was created from the system; setting the discrete system
states also sets the configuration of variational integrator and the
system. This can be checked by checking the values before and after running the
``set`` method, as shown below.

    >>> dsys.xk
    array([ 3.14159265,  0.        ])
    
    >>> mvi.q1
    array([ 3.14159265])

    >>> system.q
    array([ 3.14159265])

    >>> dsys.set(np.array([q0, 0.]), np.array([0.]), 0) 

    >>> dsys.xk
    array([ 2.35619449,  0.        ])

    >>> mvi.q1
    array([ 2.35619449])

    >>> system.q
    array([ 2.34019535])

    >>> (mvi.q2 + mvi.q1)/2.0
    array([ 2.34019535])

Note that the ``system.q`` returns the current configuration of the *system*.
This is calculated using the midpoint rule and the endpoints of the variational
integrator as seen above.

Simulate the system forward
---------------------------
As was done in the `previous section of this tutorial <pendulumSystem.html>`_ the
system is simulated forward with a while loop, which steps it forward one time
step at a time.

.. literalinclude:: ./code_snippets/linearFeedbackController.py
    :start-after: # Simulate the system forward
    :end-before: # Visualize the system in action

This time the system is stepped forward with the
:mod:`trep.discopt.DSystem.step` method instead of the
:mod:`trep.MidpointVI.step` method. This is done so that the discrete state gets
updated and set as well as the system configurations and momenta. The input is
calcuated with a standard negative feedback of the state error multiplied by the
gain that was found previously. In addition, two more variables are created to
store the state values and the input values throughout the simulation.

Visualize the system in action
------------------------------
The visualization is created in the exact way it was created in the last
section. But this time also plotting state and input verse time.

.. literalinclude:: ./code_snippets/linearFeedbackController.py
    :start-after: # Visualize the system in action

.. image:: linearFeedbackController.gif

.. image:: linearFeedbackControllerPlot.png

From the plot you can see that it requires a large input to stablize the
pendulum. What if the input torque is limited to a smaller value and we want to
bring the pendulum to the upright configuration from any other configuration? By
using a swing-up controller and then switching to this linear feedback
controller we can accomplish this stabilization with a limited input from any
configuration. We will demonstrate how this can be done with trep in the next
two sections.

linearFeedbackController.py code
--------------------------------
Below is the entire script used in this section of the tutorial.

.. literalinclude:: ./code_snippets/linearFeedbackController.py
    :linenos:
