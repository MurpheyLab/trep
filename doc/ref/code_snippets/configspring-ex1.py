import trep
from trep import tx,ty,tz,rx,ry,rz

# Create a sytem with one mass that moves in the x direction.
system = trep.System()
system.import_frames([tx('x', mass=1, name='block')])

trep.potentials.ConfigSpring(system, 'x', k=20, q0=0)

