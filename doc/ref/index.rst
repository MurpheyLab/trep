:mod:`trep` Reference
=====================

.. module:: trep
   :platform: Linux, Mac
   :synopsis: Dynamics and Optimal Control Tools

:mod:`trep` is a Python module for simulation and trajectory
optimization of mechanical systems in generalized coordinates.  


The dynamics can be modeled in continuous time as a traditional ODE,
or in discrete time using variational integrators.


The central component of :mod:`trep` is the :class:`System`
class. A :class:`System` is a collection of coordinate frames,
forces, potential energies, and constraints that describe a mechanical
system in generalized coordinates.  The :class:`System` is capable
of calculating the continuous dynamics and the first and second
derivatives of the dynamics.  


.. toctree::
   :maxdepth: 2
   
   core
   potentials
   forces
   constraints
   optimal
   misc
   visualization

