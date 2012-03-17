import trep
from trep import tx,ty,tz,rx,ry,rz

system = trep.System()
system.import_frames([
    ry('theta1'), [
        tz(2, mass=1, name='pend1')
        ],
    tx(1), [
        ry('theta2'), [
            tz(2, mass=1, name='pend2')
            ]]
    ])

trep.potentials.LinearSpring(system, 'pend1', 'pend2', k=20, x0=0.9)
