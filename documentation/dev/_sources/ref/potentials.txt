.. _builtin_potential_energies:

:mod:`trep.potentials` - Potential Energies
===========================================

.. module:: trep.potentials
   :platform: Linux, Mac
   :synopsis: Built in potential energy types.



Conservative forces are best modeled in :mod:`trep` as potential
energies.  Every type of potential is derived from :class:`trep.Potential`.

These are the types of potential energy currently built in to
:mod:`trep`.  

.. :ref:`creating_potential_energies` describes how to create new potential
.. energies in Python or C.


.. toctree::
   
   gravity
   linearspring
   configspring
   nonlinearconfigspring
   
