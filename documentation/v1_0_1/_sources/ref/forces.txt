.. _builtin_forces:

:mod:`trep.forces` - Forces
===========================

.. module:: trep.forces
   :platform: Linux, Mac, Windows
   :synopsis: Built in non-conservative force types.


Non-conservative forces are modeled in trep by deriving from the
:class:`Force` type.  These types of forces include damping and
control forces/torques.  

These are the types of forces currently built in to :mod:`trep`.

..
   :ref:`creating_forces` describes how to create a new type of force in
   Python or C.


.. toctree::
   damping
   lineardamper
   configforce
   bodywrench
   hybridwrench
   spatialwrench

   
