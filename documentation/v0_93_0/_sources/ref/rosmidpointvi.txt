.. _trep_rosmidpointvi:

:class:`ROSMidpointVI` - ROS Midpoint Variational Integrator
============================================================

.. currentmodule:: trep.ros

The :class:`ROSMidpointVI` class wraps the :class:`trep.MidpointVI` class to implement a variational integrator with a few extra features for use in a ROS environment. This class is a superset of the :class:`trep.MidpointVI` class, so see the :class:`trep.MidpointVI` documentation for a full listing of the class description.

Initializing the :class:`ROSMidpointVI` class creates a :class:`rospy` publisher which will automatically publish the configurations to the the */joint_states* topic when :meth:`step` is called.


.. class:: ROSMidpointVI(system, timestep, tolerance=1e-10, num_threads=None)

   Create a new empty mechanical system.  *system* is a valid
   :class:`System` object that will be simulation. 

   *timestep* is a different requirement for the :class:`ROSMidpointVI` class.  This
    sets the fixed timestep for the system simulation.  The value is in seconds, ie. 0.1.

   *tolerance* sets the desired tolerance of the root solver when
    solving the DEL equation to advance the integrator.

    :class:`MidpointVI` makes use of multithreading to speed up the
    calculations.  *num_threads* sets the number of threads used by
    this integrator.  If *num_threads* is :data:`None`, the integrator
    will use the number of available processors reported by Python's
    :mod:`multiprocessing` module.           

Simulation
^^^^^^^^^^

.. method:: ROSMidpointVI.step(u1=tuple(), k2=tuple(), max_iterations=200, lambda1_hint=None, q2_hint=None)

   Step the integrator forward by one timestep .  This advances the time
   and solves the DEL equation.  The current state will become the
   previous state (ie, :math:`t_2 \Rightarrow t_1`, :math:`q_2
   \Rightarrow q_1`, :math:`p_2 \Rightarrow p_1`).  The solution
   will be saved as the new state, available through :attr:`t2`,
   :attr:`q2`, and :attr:`p2`.  :attr:`lambda` will be updated with
   the new constraint force, and :attr:`u1` will be updated with the
   value of *u1*.  The configurations are also published to the /joint_states topic.

   *lambda1* and *q2* can be specified to seed the root solving
   algorithm.  If they are :data:`None`, the previous values will be
   used.
 
   The method returns the number of root solver iterations needed to
   find the solution.  

   Raises a :exc:`ConvergenceError` exception if the root solver
   cannot find a solution after *max_iterations*.   

.. method:: ROSMidpointVI.sleep()

   Calls the :meth:`rospy.Rate.sleep` method set to *timestep* of the :class:`ROSMidpointVI` class.  This method attempts to keep a loop at the specified frequency accounting for the time used by any operations during the loop.  

   Raises a :exc:`rospy.ROSInterruptException` exception if sleep is interrupted by shutdown.
    
 
        
