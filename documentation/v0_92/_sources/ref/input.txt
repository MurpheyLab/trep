:class:`Input` -- Input Variables
=================================

.. currentmodule:: trep

.. class:: Input(system[, name=None])

   :param system: An instance of :class:`System` to add the variable to.
   :param name: A string that uniquely identifies the input variable.

   An :class:`Input` instance represents a single variable in the
   input vector, :math:`u`, of a mechanical system.  Inputs are used
   by :class:`Force` to apply non-conservative forcing to the system
   (e.g, a torque or body wrench)

   Input variables are created automatically by implementations of
   :class:`Force` when needed, so you do not need to create them
   directly unless defining a new force type.  In that case, they
   should be created using :meth:`Force._create_input()`.

   If a *name* is provided, it can be used to identify and retrieve
   input variables.

   The current value of a single input variable can be accessed
   directly (:attr:`u`) or through :class:`System` to access all input
   variables at once (:attr:`System.u`).
   
   .. warning::

      Currently :mod:`trep` does not enforce unique names for input
      variables. It is recommended to provide a unique name for every
      :class:`Input` so they can be unambiguously retrieved by
      :meth:`System.get_input`.


Input Objects
-------------

.. attribute:: Input.u

   The current value of the input.

.. attribute:: Input.system

   The system that owns this input variable.

   *(read only)*

.. attribute:: Input.force
             
   The force that uses this input variable.

   *(read only)*

.. attribute:: Input.name

   The name of this input variable or :data:`None`.

.. attribute:: Input.index

   The index of the input in :attr:`System.inputs`.

