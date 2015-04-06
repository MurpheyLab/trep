Create a pendulum
=================
Now let's create a trep model of a simple single-link pendulum, simulate a
discrete trajectory, and then visualize the results.

Import necessary Python modules
-------------------------------
Let us begin by importing some standard modules including the trep module into
python.

.. literalinclude:: ./code_snippets/pendulumSystem.py
    :start-after: # Import necessary Python modules
    :end-before: # Build a pendulum system
    :emphasize-lines: 5

The line that should be noted here is highlighted above. ``tx, ty, tz, rx, ry,
rz`` are trep methods that make it very easy to create new frames of reference
for your system. They are used in the next section.

Build a pendulum system
-----------------------
Let us now build the pendulum system that corresponds to the figure below.

.. image:: simplePendulum.png

.. literalinclude:: ./code_snippets/pendulumSystem.py
   :start-after: # Build a pendulum system
   :end-before: # Add forces to the system

The first lines of code here define system parameters. Then you can see that a
new trep system is created.

The frames of the system are defined using the methods that were imported
above. The :ref:`Frame <trep_frame>` documentation has a through explanation of
how to create and use these frames. The first frame is created to rotate around
its parent's (``system.world_frame``) X axis. It is defined by the configuration
parameter ``theta`` and is named ``pendulumShoulder``. The second frame is a
translation of fixed amount (``-l``) along its parent's (``pendulumShoulder``) Z
axis and is given a mass of ``m``. The ``system.import_frames`` method is used
to create the frames from the list of frame definations in ``frames``.

Add forces to the system
------------------------
Let us now add forces, potentials, and damping into the system.

.. literalinclude:: ./code_snippets/pendulumSystem.py
    :start-after: # Add forces to the system
    :end-before: # Create and initialize the variational integrator

Gravity can work in any direction relative to the world frame. Here it is
assigned to be parallel to the Z axis and have a value of :math:`-9.8
~m/s^2`. Trep can handle other types of potentials as well.  See the
:ref:`Potential <trep_potential>` documentation for information on the
:class:`trep.Potential` base class, and see the :mod:`trep.potentials`
documentation for a list of the potentials that have been implemented.

Damping is applied to the entire system with the :mod:`trep.forces.Damping`
method. Note that trep can also apply unique damping values to individual
configurations or set default values for all configurations -- see the
:ref:`Damping <trep_force_damping>` documentation.

An input is configured for the system by adding a configuration force with the
:mod:`trep.forces.ConfigForce` method. Specifically this adds an input to the
configuration variable ``theta`` (the configuration variable for the
``pendulumShoulder`` frame) with the name ``theta-torque``.  See :ref:`Forces
<builtin_forces>` for a list of the available force types, and
:class:`trep.Force` for the documentation on the base class.


Create and initialize the variational integrator
------------------------------------------------
Trep uses variational integrators to simulate the dynamics of mechanical
systems. 

.. literalinclude:: ./code_snippets/pendulumSystem.py
    :start-after: # Create and initialize the variational integrator
    :end-before: # Simulate the system forward

Here a new variational integrator object ``mvi`` is created using our ``system``
instance of a :class:`trep.System`. It is then initialized with a set of two
time points and configurations using
:mod:`trep.MidpointVI.initialize_from_configs`. The first two arguments are the
current time and configuration and the next two are the next time and
configuration. Trep calculates the discrete generalized momentum from these two
pairs. You can also initialize the variational integrator with a single time,
configuration, and momentum using the
:mod:`trep.MidpointVI.initialize_from_state` method.

Here is a list of all the properties and methods of the variational integrator.

    >>> mvi.
    mvi.calc_f                   mvi.nk                       mvi.q2_dk2dk2
    mvi.calc_p2                  mvi.nq                       mvi.q2_dp1
    mvi.discrete_fm2             mvi.nu                       mvi.q2_dp1dk2
    mvi.initialize_from_configs  mvi.p1                       mvi.q2_dp1dp1
    mvi.initialize_from_state    mvi.p2                       mvi.q2_dp1du1
    mvi.lambda1                  mvi.p2_dk2                   mvi.q2_dq1
    mvi.lambda1_dk2              mvi.p2_dk2dk2                mvi.q2_dq1dk2
    mvi.lambda1_dk2dk2           mvi.p2_dp1                   mvi.q2_dq1dp1
    mvi.lambda1_dp1              mvi.p2_dp1dk2                mvi.q2_dq1dq1
    mvi.lambda1_dp1dk2           mvi.p2_dp1dp1                mvi.q2_dq1du1
    mvi.lambda1_dp1dp1           mvi.p2_dp1du1                mvi.q2_du1
    mvi.lambda1_dp1du1           mvi.p2_dq1                   mvi.q2_du1dk2
    mvi.lambda1_dq1              mvi.p2_dq1dk2                mvi.q2_du1du1
    mvi.lambda1_dq1dk2           mvi.p2_dq1dp1                mvi.set_midpoint
    mvi.lambda1_dq1dp1           mvi.p2_dq1dq1                mvi.step
    mvi.lambda1_dq1dq1           mvi.p2_dq1du1                mvi.system
    mvi.lambda1_dq1du1           mvi.p2_du1                   mvi.t1
    mvi.lambda1_du1              mvi.p2_du1dk2                mvi.t2
    mvi.lambda1_du1dk2           mvi.p2_du1du1                mvi.tolerance
    mvi.lambda1_du1du1           mvi.q1                       mvi.u1
    mvi.nc                       mvi.q2                       mvi.v2
    mvi.nd                       mvi.q2_dk2 

Simulate the system forward
---------------------------
Let us now simulate this system forward in time.

.. literalinclude:: ./code_snippets/pendulumSystem.py
    :start-after: # Simulate the system forward
    :end-before: # Visualize the system in action

The system is simulated forward in time using a simple while loop. First, two
lists are initialized to hold all of the time values and configuration values
for the simulation. Next, it enters a loop that says to continue until the
variational integrator reaches the final time. In each interation of the loop
the system is integrated forward by one time step. Then the new values are
append to the storage vectors.

The variational integrator object has attributes for times, configurations, and
discrete generalized momenta at both time points of the current integration
(e.g. ``mvi.t1``, ``mvi.q1``, and ``mvi.p1``). The :mod:`trep.MidpointVI.step`
method integrates the system forward from the current time endpoint to the time
given the in the first argument. The second argument specifies the input over
that time period. You can see above that the input is set to zero.

Visualize the system in action
------------------------------
Let us use trep's visualization tools to see the system in action.

.. literalinclude:: ./code_snippets/pendulumSystem.py
    :start-after: # Visualize the system in action

Finally let's create a visualization of the system being simulated. Only the
system object, the list of times, and the list of configurations are needed to
create the visualization.

The output that you should see on your screen is shown below.

.. image:: pendulumSystem.gif

**Note:** *This is an animated gif of captured screen shots. So it is slower and
lower-quality than what you will see.*

pendulumSystem.py code
----------------------
Below is the entire script used in this section of the tutorial.

.. literalinclude:: ./code_snippets/pendulumSystem.py
    :linenos:
