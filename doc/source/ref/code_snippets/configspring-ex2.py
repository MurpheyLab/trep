import trep
from trep import tx,ty,tz,rx,ry,rz

system = trep.System()
system.import_frames([
    ry("theta"), [
        tz(2, mass=1)
        ]])

trep.potentials.ConfigSpring(system, 'theta', k=20, q0=0.7)
