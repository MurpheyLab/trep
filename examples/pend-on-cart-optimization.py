import numpy as np
import sys
import trep
from trep import tx, ty, tz, rx, ry, rz
import trep.discopt as discopt
import math
from math import sin, cos
from math import pi as mpi
import trep.visual as visual
from PyQt4.QtCore import Qt, QRectF, QPointF
from PyQt4.QtGui import QColor


class PendCartVisual(visual.VisualItem2D):
    def __init__(self, *args, **kwds):
        
        draw_track = kwds.setdefault('draw_track', True)
        del(kwds['draw_track'])

        super(PendCartVisual, self).__init__(*args, **kwds)

        if draw_track:
            self.attachDrawing(None, self.paint_track)
        self.attachDrawing('Cart', self.paint_cart)
        self.attachDrawing('PendulumBase', self.paint_pend)
        self.attachDrawing('Pendulum', self.paint_mass)

    def paint_track(self, painter):
        rect = QRectF(0, 0, 4.0, 0.05)
        rect.moveCenter(QPointF(0,0))
        painter.fillRect(rect, QColor(100, 100, 100))

    def paint_cart(self, painter):
        rect = QRectF(0, 0, 0.2, 0.07)
        rect.moveCenter(QPointF(0,0))
        painter.fillRect(rect, QColor(200, 200, 200))

    def paint_pend(self, painter):
        rect = QRectF(-0.01, 0, 0.02, -1.0)
        painter.fillRect(rect, QColor(0, 0, 0))

    def paint_mass(self, painter):
        rect = QRectF(0, 0, 0.07, 0.07)
        rect.moveCenter(QPointF(0,0))
        painter.fillRect(rect, QColor(200, 200, 0))


def build_system(torque_force=False):
    cart_mass = 10.0
    pendulum_length = 1.0
    pendulum_mass = 1.0

    system = trep.System()
    frames = [
        tx('x', name='Cart', mass=cart_mass), [
            rz('theta', name="PendulumBase"), [
                ty(-pendulum_length, name="Pendulum", mass=pendulum_mass)]]]
    system.import_frames(frames)
    trep.potentials.Gravity(system, (0, -9.8, 0))
    trep.forces.Damping(system, 0.01)
    trep.forces.ConfigForce(system, 'x', 'x-force')
    if torque_force:
        trep.forces.ConfigForce(system, 'theta', 'theta-force')
    return system

def generate_desired_trajectory(system, t, amp=130*mpi/180):
    qd = np.zeros((len(t), system.nQ))
    theta_index = system.get_config('theta').index
    for i,t in enumerate(t):
        if t >= 3.0 and t <= 7.0:
            qd[i, theta_index] = (1 - cos(2*mpi/4*(t-3.0)))*amp/2
    return qd

def make_state_cost(dsys, base, x, theta):
    weight = base*np.ones((dsys.nX,))
    weight[system.get_config('x').index] = x
    weight[system.get_config('theta').index] = theta
    return np.diag(weight)

def make_input_cost(dsys, base, x, theta=None):
    weight = base*np.ones((dsys.nU,))
    if theta is not None:
        weight[system.get_input('theta-force').index] = theta
    weight[system.get_input('x-force').index] = x
    return np.diag(weight)                    



# Build cart system with torque input on pendulum.
system = build_system(True)
mvi = trep.MidpointVI(system)
t = np.arange(0.0, 10.0, 0.01)
dsys_a = discopt.DSystem(mvi, t)


# Generate an initial trajectory
(X,U) = dsys_a.build_trajectory()
for k in range(dsys_a.kf()):
    if k == 0:
        dsys_a.set(X[k], U[k], 0)
    else:
        dsys_a.step(U[k])
    X[k+1] = dsys_a.f()


# Generate cost function
qd = generate_desired_trajectory(system, t, 130*mpi/180)
(Xd, Ud) = dsys_a.build_trajectory(qd)
Qcost = make_state_cost(dsys_a, 0.01, 0.01, 100.0)
Rcost = make_input_cost(dsys_a, 0.01, 0.01, 0.01)
cost = discopt.DCost(Xd, Ud, Qcost, Rcost)

optimizer = discopt.DOptimizer(dsys_a, cost)

# Perform the first optimization
optimizer.first_method_iterations = 4
finished, X, U = optimizer.optimize(X, U, max_steps=40)

# Increase the cost of the torque input
cost.R = make_input_cost(dsys_a, 0.01, 0.01, 100.0)
optimizer.first_method_iterations = 4
finished, X, U = optimizer.optimize(X, U, max_steps=40)

# We could print a converge plot here if we wanted to.
## dcost = np.array(optimizer.monitor.dcost_history.items()).T
## import pylab
## pylab.semilogy(dcost[0], -dcost[1])
## pylab.show()

# Increase the cost of the torque input
cost.R = make_input_cost(dsys_a, 0.01, 0.01, 1000000.0)
optimizer.first_method_iterations = 4
finished, X, U = optimizer.optimize(X, U, max_steps=40)


# The torque should be really tiny now, so we can hopefully use this
# trajectory as the initial trajectory of the real system.  

# Build a new system without the extra input
system = build_system(False)
mvi = trep.MidpointVI(system)
dsys_b = discopt.DSystem(mvi, t)

# Map the optimized trajectory for dsys_a to dsys_b
(X, U) = dsys_b.convert_trajectory(dsys_a, X, U)

# Simulate the new system starting from the initial condition of our
# last optimization and using the x-force input.
for k in range(dsys_b.kf()):
    if k == 0:
        dsys_b.set(X[k], U[k], 0)
    else:
        dsys_b.step(U[k])
    X[k+1] = dsys_b.f()

# Generate a new cost function for the current system.
qd = generate_desired_trajectory(system, t, 130*mpi/180)
(Xd, Ud) = dsys_b.build_trajectory(qd)
Qcost = make_state_cost(dsys_b, 0.01, 0.01, 100.0)
Rcost = make_input_cost(dsys_b, 0.01, 0.01)
cost = discopt.DCost(Xd, Ud, Qcost, Rcost)

optimizer = discopt.DOptimizer(dsys_b, cost)

# Perform the optimization on the real system
optimizer.first_method_iterations = 4
finished, X, U = optimizer.optimize(X, U, max_steps=40)



if '--novisual' not in sys.argv:

    q,p,v,u,rho = dsys_b.split_trajectory(X, U)

    if False:
        view = Viewer(system, t, q, qd)
        view.main()
    else:
        visual.visualize_2d([
            PendCartVisual(system, t, qd),
            PendCartVisual(system, t, q, draw_track=False)
            ])

