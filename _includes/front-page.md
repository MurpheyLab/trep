### Trep: Mechanical Simulation and Optimal Control
Trep is a Python module for modeling articulated rigid body mechanical systems in generalized coordinates. Trep supports basic simulation but it is primarily designed to serve as a calculation engine for analysis and optimal control algorithms that require 1st and 2nd derivatives of the system's dynamics.

Get started by [installing](install/) trep and checking out the [documentation](documentation/current/). Source code for trep is available from the [GitHub repository](https://github.com/MurpheyLab/trep).

ROS users can now get trep binaries directly from the ROS repositories. Check the *python_trep* [wiki page](http://wiki.ros.org/python_trep) for more information.

Trep is designed to work with large mechanical systems in generalized coordinates. This video shows the progress of a discrete time trajectory optimization that was created with trep.

Trep currently features:

- Concise System Definitions
- Conservative Forces (Potential Energies)
- Non-Conservative forces
- Holonomic Constraints
- Continuous Dynamics
- Exact 1st and 2nd derivatives of continuous dynamics
- Discrete Dynamics using variational integrators
- Exact 1st and 2nd derivatives of discrete dynamics
- Projection-based Trajectory Optimization in discrete time
- Automatic Visualization Tools
- Excellent Scalability
