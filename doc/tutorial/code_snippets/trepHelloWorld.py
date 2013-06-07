# trepHelloWorld.py

# Import the Trep module into python
import trep
# Create a new instance of a trep system
system = trep.System()
# Visualize the system with trep's visualization tools
trep.visual.visualize_3d([ trep.visual.VisualItem3D(system, [], []) ])
