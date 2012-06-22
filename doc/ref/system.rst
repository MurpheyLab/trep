.. _trep_system:

:class:`System` - A Mechanical System
=====================================

.. currentmodule:: trep

The :class:`System` object is the central component for modeling a
mechanical system, and usually the first :mod:`trep` object you
create.  It contains the entire definition of the system, including
all coordinate frames, configuration variables, constraints, etc.

:class:`System` is responsible for calculating continuous dynamics and
derivative, but it is also used for the underlying Lagrangian
calculations in any discrete dynamics calculations.  

A :class:`System` has a single inertial coordinate frame, accessible
through the :attr:`world_frame` attribute.  Every other coordinate
frame added to the system will be descended from this coordinate
frame.  

.. class:: System()

   Create a new empty mechanical system.   

.. contents::
   :local:


System Components
-----------------

.. attribute:: System.world_frame

   The root spatial frame of the system.  The world frame will always
   exist and cannot be changed other than adding child frames.

   *(read only)*
    
.. attribute:: System.frames
               System.configs    
               System.dyn_configs
               System.kin_configs
               System.potentials
               System.forces
               System.inputs
               System.constraints

   Tuples of all the components in the system.  **The order of
   components in these tuples defines their order throughout**
   :mod:`trep`.  Any vector of configuration values will be the same
   order as :attr:`System.configs`, any vector of constraint forces
   will be the same order as :attr:`System.constraints`, etc.

   For example, any array of numbers representing a configuration will
   be ordered according to :attr:`System.configs`.  An array of
   constraint forces will correspond to the order of
   :attr:`System.constraints`.

   :attr:`configs` is guaranteed to be ordered as the concatenation of
   :attr:`dyn_configs` and :attr:`kin_configs`::

         System.configs = System.dyn_configs + System.kin_configs

   The tuples are read only, but the components in them may be
   modified.


.. attribute:: System.masses

   A tuple of all the :class:`Frame` objects in the system with
   non-zero mass properties.

   *(read only)*

.. attribute:: System.nQ
               
   Number of configuration variables in the system.  Equivalent to
   ``len(system.configs)``.

   *(read only)*
    
.. attribute:: System.nQd

   Number of dynamic configuration variables in the system. Equivalent
   to ``len(system.dyn_configs)``.
    
   *(read only)*
    
.. attribute:: System.nQk

   Number of kinematic configuration variables in the system.
   Equivalent to ``len(system.kin_configs)``.
    
   *(read only)*
    
.. attribute:: System.nu

   Number of inputs in the system.
   Equivalent to ``len(system.inputs)``.
    
   *(read only)*
    
.. attribute:: System.nc  

   Number of constraints in the system.
   Equivalent to ``len(system.constraints)``.

   *(read only)*
    

Finding Specific Components
---------------------------

.. method:: System.get_frame(identifier)
            System.get_config(identifier)
            System.get_potential(identifier)
            System.get_constraint(identifier)
            System.get_force(identifier)
            System.get_input(identifier)

   Find a specific component in the system.  The identifier can be:

   - Integer: Returns the component at that position in the
     corresponding tuple.  
       
       system.get_frame(i) = system.frames[i]

     If the index is invalid, an :exc:`IndexError` exception is
     raised.

   - String: Returns the component with the matching name:

       system.get_config('theta').name == 'theta'

     If no object has a matching name, a :exc:`KeyError` exception is
     raised.

   - Object: Return the identifier unmodified::

       system.get_force(force) == force

     If the object is the incorrect type, a :exc:`TypeError` exception
     is raised::

       >>> config = system.configs[0]
       >>> system.get_frame(config)
       Traceback (most recent call last):
         File "<stdin>", line 1, in <module>
          File "/usr/local/lib/python2.7/dist-packages/trep/system.py", line 106, in get_frame
           return self._get_object(identifier, Frame, self.frames)
         File "/usr/local/lib/python2.7/dist-packages/trep/system.py", line 509, in _get_object
           raise TypeError()
       TypeError
  

Importing and Exporting Frame Definitions
-----------------------------------------

.. method:: System.import_frames(children)

   Adds children to this system's world frame using a special
   frame definition.  See :meth:`Frame.import_frames` for details.

.. method:: System.export_frames(system_name='system', frames_name='frames', tab_size=4)

   Create python source code to define this system's frames.  The code
   is returned as a string.

System State
------------

A :class:`System` is a stateful object that has a current time,
configuration, velocity, and input at all times.  When working
directly with a :class:`System`, you are a responsible for setting the
state.


.. attribute:: System.t

   Current time of the system.
   
The other state information is actually spread out among each
component in the system.  For example, each current configuration
value is stored in :attr:`Config.q`.  These can be modified directly
through each component, but the following attributes are usually more
convenient for reading and writing multiple values at once.

.. attribute:: System.q
               System.dq
               System.ddq

   The value, velocity, and acceleration of the complete
   configuration.

.. attribute:: System.qd
               System.dqd
               System.ddqd

   The value, velocity, and acceleration of the dynamic configuration.

.. attribute:: System.qk
               System.dqk
               System.ddqk

   The value, velocity, and acceleration of the kinematic
   configuration.

.. attribute:: System.u

   The current values of the force inputs.


Reading the each attribute will return a numpy array of the current
values.  

The values can be set with three different methods:

   - A dictionary that maps names to values::
       
       >>> system.q = { 'x' : 1.0, 'theta' : 0.1}

     Any variables that are not named will be unchanged.

   - A array-like list of numbers::

       >>> system.q = [1.0, 0.1, 0.0, 2.3]
       >>> system.q = np.array([1.0, 0.1, 0.0, 2.3])
     
     If the size of the array and the number of variables doesn't
     match, the shorter of the two will be used::

       >>> system.nQ
       4

       # This only sets the first 2 configuration variables!
       >>> system.q = [0.2, 5.0]
       # This ignores the last 2 values!
       >>> system.q = [0.5, 0, 1.1, 2.4, 9.0]

   - A single number::
     
       >>> system.q = 0
       
     This will set the entire configuration to the number.


.. method:: System.set_state(q=None, dq=None, u=None, ddqk=None, t=None)

   Set the current state of the system, not including the "output"
   ddqd.  The types of values accepted are the same as described
   above.


Constraints
-----------

.. method:: System.satisfy_constraints(tolerance=1e-10, verbose=False)

   Modify the current configuration to satisfy the system constraints.
   Letting :math:`q_0` be the system's current configuration, this
   performs the optimization:

   .. math::
      
      q = \arg\min_q  |q-q_0|-2 \quad \mathrm{s.t.}\quad  h(q) = 0

   The new configuration will be set in the system and returned.  If
   the optimization fails, a :exc:`StandardError` exception is raised.

   Setting *verbose* to :data:`True` will make the optimization print
   out information to the console while it is running.


Lagrangian Calculations
-----------------------
    
.. method:: System.total_energy()
            
   Calculate the total energy in the current state.
    
.. method:: System.L()
            System.L_dq(q1)
            System.L_dqdq(q1, q2)
            System.L_dqdqdq(q1, q2, q3)
            System.L_ddq(dq1)
            System.L_ddqdq(dq1, q2)
            System.L_ddqdqdq(dq1, q2, q3)
            System.L_ddqdqdqdq(dq1, q2, q3, q4)    
            System.L_ddqddq(dq1, dq2)
            System.L_ddqddqdq(dq1, dq2, q3)
            System.L_ddqddqdqdq(dq1, dq2, q3, q4)


   Calculate the Lagrangian or it's derivatives at the current state.
   When calculating the Lagrangian derivatives, you must specify a
   variable to take the derivative with respect to.

   Calculating :math:`\deriv[L]{q_0}`::

     >>> q0 = system.configs[0]
     >>> system.L_dq(q0)

   Calculating :math:`\derivII[L]{q_0}{\dq_1}`::

     >>> q0 = system.configs[0]
     >>> q1 = system.configs[1]
     >>> system.L_ddqdq(q1, q0)

   Calculating :math:`\derivII[L]{\dq_0}{\dq_1}`::

     >>> q0 = system.configs[0]
     >>> q1 = system.configs[1]
     >>> system.L_ddqddq(q1, q0)

     # Mixed partials always commute, so we could also do:
     >>> system.L_ddqddq(q0, q1)

   Calculate an entire derivative :math:`\deriv[L]{q}`::

     >>> [system.L_dq(q) for q in system.configs]
     

.. _system_dynamics_functions:

Dynamics
--------

.. method:: System.f(q=None)
   
   :type q: :class:`Config` or :data:`None`
            

   Calculate the dynamics at the current state, :math:`\ddq_d = f(q,\dq,\ddq_k, u, t)`.

   This calculates the second derivative of the dynamic configuration
   variables.  The results are also written to :attr:`Config.ddq`.

   If *q* is :data:`None`, the entire vector :math:`\ddq_d` is
   returned.  The array is in the same order as
   :attr:`System.dyn_configs`.

   If *q* is specified, it must be a dynamic configuration variable,
   and it's second derivative will be returned.  This could also be
   accessed as ``q.ddq``.

 
   Once the dynamics are calculated, the results are saved until the
   system's state changes, so repeated calls will not keep repeating
   work.

.. method:: System.f_dq(q=None, q1=None)
            System.f_ddq(q=None, dq1=None)
            System.f_dddk(q=None, k1=None)
            System.f_du(q=None, u1=None)
            
   Calculate the first derivative of the dynamics with respect to the
   configuration, the configuration velocity, the kinematic
   acceleration, or the force inputs.

   If both parameters are :data:`None`, the entire first derivative is
   returned as a :mod:`numpy` array with the derivatives across the
   rows.  
   
   If any parameters are specified, they must be appropriate objects,
   and the function will return the specific information requested.

   Calculating :math:`\deriv[f]{q_2}`::

     >>> dq = system.configs[2]
     >>> system.f_dq(q1=dq)
     array([-0.076, -0.018, -0.03 , ...,  0.337, -0.098,  0.562])

   Calculating :math:`\deriv[f]{\dq}`::

     >>> system.f_ddq()
     array([[-0.001, -0.   , -0.   , ...,  0.001, -0.001,  0.   ],
     [-0.   , -0.002, -0.   , ..., -0.   , -0.   ,  0.   ],
     [-0.   , -0.   , -0.   , ...,  0.   , -0.   ,  0.   ],
     ..., 
     [-0.   , -0.001, -0.   , ..., -0.003, -0.   ,  0.   ],
     [ 0.001, -0.   ,  0.   , ..., -0.005,  0.008, -0.   ],
     [-0.001, -0.   , -0.   , ...,  0.008, -0.027,  0.   ]])
     >>> system.f_ddq().shape
     (22, 23)

   Calculating :math:`\deriv[\ddq_4]{\ddq_{k0}}`::

     >>> qk = system.kin_configs[0]
     >>> q = system.dyn_configs[4]
     >>> system.f_dddk(q, qk)
     -0.31036045452513322

   The first call to any of these functions will calculate and cache
   the entire first derivative.  Once calculated, subsequent calls
   will not recalculate the derivatives until the system's state is
   changed.


.. method:: System.f_dqdq(q=None, q1=None, q2=None)
            System.f_ddqdq(q=None, dq1=None, q2=None)
            System.f_ddqddq(q=None, dq1=None, dq2=None)
            System.f_dddkdq(q=None, k1=None, q2=None)
            System.f_dudq(q=None, u1=None, q2=None)
            System.f_duddq(q=None, u1=None, dq2=None)
            System.f_dudu(q=None, u1=None, u2=None)

   Calculate second derivatives of the dynamics.  

   If no parameters specified, the entire second derivative is
   returned as a :mod:`numpy` array.  The returned arrays are indexed
   with the two derivatives variables as the first two dimensions and
   the dynamic acceleration as the last dimension.  In other words,
   calling::
      
      system.f_dqdq(q, q1, q2)  

   is equivalent to::
     
     result = system.f_dqdq()
     result[q1.index, q2.index, q.index]

   If any parameters are specified, they must be appropriate objects,
   and the function will return the specific information requested.
   For example::

     system.f_dqdq(q2=q2)

   is equivalent to::

     result = system.f_dqdq()
     result[:, q2.index, :]

   The second derivatives are indexed opposite from the first
   derivatives because they are generally multiplied by an adjoint
   variable in practice.  For example, the quantity:

   .. math::

      z^T \derivII[f]{q}{\dq}

   can be calculated as::

     numpy.inner(system.f_dqddq(), z)

   without having to do a transpose or specify a specific axis.


   The first call to any of these functions will calculate and cache
   the entire second derivative.  Once calculated, subsequent calls
   will not recalculate the derivatives until the system's state is
   changed.


Constraint Forces
-----------------

.. method:: System.lambda_(constraint=None)
            
.. method:: System.lambda_dq(constraint=None, q1=None)
            System.lambda_ddq(constraint=None, dq1=None)
            System.lambda_dddk(constraint=None, k1=None)
            System.lambda_du(constraint=None, u1=None)

.. method:: System.lambda_dqdq(constraint=None, q1=None, q2=None)
            System.lambda_ddqdq(constraint=None, dq1=None, q2=None)
            System.lambda_ddqddq(constraint=None, dq1=None, dq2=None)
            System.lambda_dddkdq(constraint=None, k1=None, q2=None)
            System.lambda_dudq(constraint=None, u1=None, q2=None)
            System.lambda_duddq(constraint=None, u1=None, dq2=None)
            System.lambda_dudu(constraint=None, u1=None, u2=None)

   These are functions for calculating the value of the constraint
   force vector :math:`\lambda` and its derivatives.  They have the
   same behavior and index orders as the :ref:`dynamics <system_dynamics_functions>` functions.


Derivative Testing
------------------


.. method:: System.test_derivative_dq(func, func_dq, delta=1e-6, tolerance=1e-7, verbose=False, \
                                      test_name='<unnamed>')
            System.test_derivative_ddq(func, func_ddq, delta=1e-6, tolerance=1e-7, \
                                       verbose=False, test_name='<unnamed>')

        Test the derivative of a function with respect to a
        configuration variable's time derivative and its numerical
        approximation.

        func -> Callable taking no arguments and returning float or np.array
        
        func_ddq -> Callable taking one configuration variable argument
                   and returning a float or np.array.

        delta -> perturbation to the current configuration to
                 calculate the numeric approximation.

        tolerance -> acceptable difference between the approximation
                     and exact value.  (\|exact - approx\| <= tolerance)

        verbose -> Boolean indicating if a message should be printed for failures.

        name -> String identifier to print out when reporting messages
                when verbose is true.

        Returns False if any tests fail and True otherwise.



Structure Updates 
-----------------

Whenever a system is significantly modified, an internal function is
called to make sure everything is consistent and resize the arrays
used for calculating values as needed.  You can register a function to
be called after the system has been made consistent using the
:meth:`add_structure_changed_func`.  This is useful if you building
your own component that needs to be updated whenever the system's
structure changed (See :class:`Damping` for an example).
        
.. method:: System.add_structure_changed_func(function)

   Register a function to call whenever the system structure changes.
   This includes adding and removing frames, configuration variables,
   constraints, potentials, and forces.

Since every addition to the system triggers a structure update,
building a large system can cause a long delay.  In these cases, it is
useful to stop the structure updates until the system is fully
constructed and then perform the update once.  

.. warning::
   
   Be sure to remove all holds before performing any calculations with system.

.. method:: System.hold_structure_changes()

   Prevent the system from calling :meth:`System._update_structure()`
   (mostly).  This can be called multiple times, but
   :meth:`resume_structure_changes` must be called an equal number of
   times.


.. method:: System.resume_structure_changes()

   Stop preventing the system from calling
   :meth:`System._update_structure()`.  The structure will only be
   updated once every hold has been removed, so calling this does not
   guarantee that the structure will be immediately update.


