.. _trep_urdf:

URDF Import Tool
====================

.. currentmodule:: trep.ros

The :mod:`trep.ros` module provides several tools for working with the ROS environment.  The URDF import function allows trep to interface directly with the Unified Robot Description Format (URDF). 

.. function:: import_urdf(source, system=None, prefix=None)
   
   :param source: string containing the URDF xml data
   :type filename: string
   :rtype: :mod:`trep.system` class

   This function creates the :mod:`trep.system` class and fills in the frames and joints
   as defined in the URDF xml data.

   *source* is a string containing the URDF data. May be loaded from the parameter server if using a launch file.
    
   *system* is a previously defined :mod:`trep.system` class.  If provided, the URDF will be imported into the existing system.

   *prefix* is a string to be prefixed to all trep frames and configs created when importing the URDF. This is useful if importing the same URDF multiple times in a single system, ie. multiple robots.

   The function returns the trep system defining the URDF.

.. function:: import_urdf_file(filename, system=None, prefix=None)
   
   :param filename: path to URDF
   :type filename: string
   :rtype: :mod:`trep.system` class

   This function creates the :mod:`trep.system` class and fills in the frames and joints
   as defined in the URDF file.

   *filename* is the complete path to the URDF file.

   *system* is a previously defined :mod:`trep.system` class.  If provided, the URDF will be imported into the existing system.

   *prefix* is a string to be prefixed to all trep frames and configs created when importing the URDF. This is useful if importing the same URDF multiple times in a single system, ie. multiple robots.

   The function returns the trep system defining the URDF.
