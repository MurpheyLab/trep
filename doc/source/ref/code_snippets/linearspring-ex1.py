import trep
from trep import tx,ty,tz,rx,ry,rz

# Create a sytem with one mass that moves in the x direction.
system = trep.System()
system.import_frames([tx('x', mass=1, name='block')])

trep.potentials.LinearSpring(system, 'World', 'block', k=20, x0=1)

# Remember to avoid x = 0 in simulation.
system.get_config('x').q = 0.5
