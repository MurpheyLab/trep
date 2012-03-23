# This simulation implements a screw type transformation using a
# constraint.  The same approach can be used to implement other
# mechanical structures like cam-shafts or gears.
import sys
from math import sin, cos, pi as mpi
import trep
from trep import tx, ty, tz, rx, ry, rz
import trep.visual as visual


tf = 10.0
dt = 0.01

class Screw(trep.Constraint):
    def __init__(self, system, angle, offset, pitch=1.0):
        trep.Constraint.__init__(self, system)
        angle = system.get_config(angle)
        offset = system.get_config(offset)
        self.angle_config = angle
        self.offset_config = offset
        self.pitch = float(pitch)
        
    def delete(self):
        self.angle_config = None
        self.offset_config = None
        Constraint.delete(self)
        
    def requires_config(self, frame):
        if frame.config == self.angle_config:
            return True
        elif frame.config == self.offset_config:
            return True
        else:
            return False

    def h(self):
        return self.angle_config.q*self.pitch - self.offset_config.q
        
    def h_dq(self, q_i):
        if q_i == self.angle_config:
            return self.pitch
        elif q_i == self.offset_config:
            return -1.0
        else:
            return 0.0
        
    def h_dqdq(self, q_i, q_j):
        return 0.0

   
system = trep.System()
frames = [
    tz('hel-x', kinematic=True, name='hel-mid', mass=(1,1,1,1)), [
        rz('hel-angle', name='hel-part', mass=(1,1,1,1)), [tx(1)]]
    ]
# Add the frames to the system.
system.import_frames(frames)
# Add gravity 
trep.potentials.Gravity(system, (0, 0, -9.8))

Screw(system, "hel-angle", "hel-x", 1.0/5.0)

# Define a function that we'll use to drive the kinematic variable.
def calc_x(t):
    return 0.75 + 0.75*sin(t)

# Calculate an initial condition that is consistent with the constraints.   
system.q = 0.0
system.get_config('hel-x').q = calc_x(dt)
system.satisfy_constraints()
q0 = system.q

# Create and initialize the variational integrator
mvi = trep.MidpointVI(system)
mvi.initialize_from_configs(0.0, q0, dt, q0)

# This is our simulation loop.  We save the results in two lists.
q = [mvi.q2]
t = [mvi.t2]
while mvi.t1 < tf:
    t2 = mvi.t2 + dt
    mvi.step(t2, (), (calc_x(t2),))
    q.append(mvi.q2)
    t.append(mvi.t2)
    # The Python constraints are much slower, so print out the time
    # occasionally to indicate our progress.
    if abs(mvi.t2 - round(mvi.t2)) < dt/2.0:
        print "t =",mvi.t2

# After the simulation is finished, we can visualize the results.  The
# viewer can automatically draw a primitive representation for
# arbitrary systems from the tree structure.  print_instructions() has
# the viewer print out basic usage information to the console.
visual.visualize_3d([visual.VisualItem3D(system, t, q)])


