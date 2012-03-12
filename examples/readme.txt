This directory contains demos to use for testing installations and
learning how to use trep by example.

IMPORTANT: In the example commands, substitute PYTHON for the name of
  the correct python executable (ie, python 2.5).  This is most likely
  'python' or 'python2.5'.



puppet-interactive - An interactive marionette simulation.  This example
  shows how to create highly customized visualizations and uses
  kinematic configuration variables.  Run using the command:

    PYTHON puppet.py
  	   
scissor - The scissor lift is a nice benchmark system because it can
  be varied in complexity by changing the number of segments in the
  system and contains a large number of holonomic constraints.
  Additionally, we can generate benchmark trajectories in Mathematica
  for comparison (however, this is not done here).  This example shows
  how to build a system through the trep API instead of s-expressions.
  It also shows how to customize the automatic visualization and use a
  2D display.  The default simulation can be run using the command:

     PYTHON scissor.py 

  For a list of options that can be changed, run: 
     
     PYTHON scissor.py --help

pendulum - Simulate a two dimensional N-link pendulum.  This example
  shows how to create a basic simulation and define a system with the
  trep API.  The default simulation can be run as:

     PYTHON pendulum.py

  For a list of options that can be changed, run:
 
     PYTHON pendulum.py --help
  
screw-joint - The screw-joint simulation uses a custom constraint to
  create a screw joint in a system.  This example shows how to create
  new constraints directly in Python (ie, without writing/compiling
  code in C).  Run with the command:
   
    PYTHON screw-joint.py

sexp - This example simulates generic systems that are specified as
  s-expressions.  Systems are loaded from s-expressions, simulated,
  and then automatically visualized.  Two examples are provided: a
  puppet and the closed-chain device.  To run the puppet example:
 
    PYTHON simulate.py puppet.lsp

  To run the planar closed-chain device:

    PYTHON simulate.py pccd.lsp

  To get a list of options that can be changed, run:
    
    PYTHON simulate.py --help

linear_spring - This uses a linear spring as an example of creating a
  custom potential energy type.  The new potential is created purely 
  in Python (ie, no C code is required).  To run the example:

    PYTHON spring.py

