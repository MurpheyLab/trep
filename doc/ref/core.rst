
:mod:`trep` - Core Components
=============================

.. module:: trep
   :platform: Linux, Mac, Windows
   :synopsis: Dynamics and Optimal Control Tools

The central component of :mod:`trep` is the :class:`System`
class. A :class:`System` is a collection of coordinate frames,
forces, potential energies, and constraints that describe a mechanical
system in generalized coordinates.  The :class:`System` is capable
of calculating the continuous dynamics and the first and second
derivatives of the dynamics.  

.. toctree::
   :maxdepth: 1

   system
   frame
   config
   input
   force
   potential
   constraint
   midpointvi
