:class:`Config` -- Configuration Variables
==========================================

.. currentmodule:: trep

.. class:: Config(system[, name=None, kinematic=False])

   :param system: An instance of :class:`System` to add the variable to.
   :param name: A string that uniquely identifies the configuration variable.
   :param kinematic: True to define a kinematic configuration variable.

   An :class:`Config` instance represents a configuration variable in
   the generalized coordinates, :math:`q`, of a mechanical system.
   Configuration variables primarily parameterize rigid-body
   transformations between coordinate frames in the system, though
   sometimes special configuration variables are used only by
   constraints.

   Configuration variables are created automatically by :class:`Frame`
   and :class:`Constraint` when needed, so you do not need to create
   them directly unless defining a new constraint type.

   If a *name* is provided, it can be used to identify and retrieve
   configuration variables.

   The current values and their derivatives of a configuration
   variable can be accessed directly (:attr:`q`, :attr:`dq`,
   :attr:`ddq`) for an individual variable, or through :class:`System`
   to access all configuration variables at once (:attr:`System.q`,
   :attr:`System.dq`, :attr:`System.ddq`)

   .. warning::

      Currently :mod:`trep` does not enforce unique names for
      configuration variables. It is recommended to provide a unique
      name for every :class:`Config` so they can be unambiguously
      retrieved by :meth:`System.get_config`.

Dynamic and Kinematic Variables
-------------------------------

   A configuration variable can be dynamic or kinematic.  Dynamic
   configuration variables, :math:`q_d`, are traditional configuration
   variables.  Their trajectory is determined by the system dynamics
   :math:`(\ddot{q_d} = f(q(t), \dot{q}(t), u(t)))`.  Dynamic
   configuration variables must parameterize a rigid-body
   transformation, and there must be a coordinate frame with non-zero
   mass that depends on the dynamic variable (directly or indirectly).

   Kinematic configuration variables, :math:`q_k`, are considered
   perfectly controllable.  Their second derivative is specified
   directly as an additional input to the system :math:`(\ddot{q_k} =
   u_k(t))`.  Kinematic configuration variables can parameterize any
   rigid body transformation in a system, regardless of whether or not
   a frame with non-zero mass depends directly or indirectly on the
   variable.  Additionally, kinematic configuration variables that do
   not parameterize a transformation can be defined by constraint
   functions.  For example, the :class:`Distance` constraint uses a
   kinematic configuration variable to maintain a time-varying
   distance between two coordinate frames.


Config Objects
--------------         

.. attribute:: Config.q

   The current value of the configuration variable.  

.. attribute:: Config.dq
               
   The 1st time derivative (velocity) of the configuration variable (i.e, :math:`\dot{q}`).

.. attribute:: Config.ddq

   The 2nd time derivative (acceleration) of the configuration variable (i.e, :math:`\ddot{q}`).

.. attribute:: Config.kinematic

   Boolean indicating if this configuration variable is dynamic or
   kinematic.

   *(read only)*

.. attribute:: Config.system    
                              
   The :class:`System` that this configuration variable belongs to. 
 
   *(read only)*

.. attribute:: Config.frame               
   
   The :class:`Frame` that depends on this configuration variable for
   its transformation, or :data:`None` if it is not used by a
   coordinate frame (e.g, a kinematic configuration variable in a
   constraint).

   *(read only)*

.. attribute:: Config.name

   The name of this configuration variable or :data:`None`.

.. attribute:: Config.index
              
   Index of the configuration variable in :attr:`System.configs`.  

   For dynamic configuration variables, this is also the index of the
   variable in :attr:`System.dyn_configs`.

   *(read only)*

.. attribute:: Config.k_index
               
   For kinematic configuration variables, this is the index of this
   variable in :attr:`System.kin_configs`.

   *(read-only)*

.. attribute:: Config.masses

   A tuple of all :class:`coordinate frames <Frame>` with non-zero
   masses that depend (directly or indirectly) on this configuration
   variable.
   
   *(read-only)*

