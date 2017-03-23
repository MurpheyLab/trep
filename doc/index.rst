.. trep documentation master file, created by
   sphinx-quickstart on Sun Mar  4 00:58:26 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Trep: Dynamic Simulation and Optimal Control
============================================

:Release: v\ |version|
:Date: |today|

Trep is a Python module for modeling rigid body mechanical systems in
generalized coordinates.  It provides tools for calculating continuous
and discrete dynamics (based on a midpoint variational integrator),
and the first and second derivatives of both.  Tools for trajectory
optimization and other basic optimal control methods are also available.

You can find detailed `installation instructions
<http://murpheylab.github.io/trep/install/>`_ on our website.
Many examples are included with the source code (`browse online
<https://github.com/MurpheyLab/trep/tree/master/examples>`_).

The :ref:`api-ref` has detailed documentation for each part of
:mod:`trep`. We have also put together a detailed :ref:`tutorial` that gives an
idea of the capabilities and organization of :mod:`trep` by stepping through
several example problems.

If you have any questions or suggestions, please head over to our
`project page <http://murpheylab.github.io/trep/>`_.

.. toctree::
   :hidden:

   Project Homepage <http://murpheylab.github.io/trep/>

.. _api-ref: 

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   Core Components <ref/core>
   Potential Energies <ref/potentials>
   Forces <ref/forces>
   Holonomic Constraints <ref/constraints>
   Discrete Optimal Control <ref/optimal>
   ROS Tools <ref/ros>
   ref/misc

.. _tutorial:

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial/trepHelloWorld
   tutorial/pendulumSystem
   tutorial/linearFeedbackController
   tutorial/energyShapingSwingupController
   tutorial/optimalSwitchingTime
   user_guide/varint_intro

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

