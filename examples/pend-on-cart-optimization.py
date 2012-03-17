import numpy as np

from itertools import product
import sys
import trep
from trep import tx, ty, tz, rx, ry, rz

import math
from math import sin, cos
from math import pi as mpi
import pygame


class Viewer(object):
    def __init__(self, system, t, q, qd):
        self.system = system
        self.t = t
        self.q_index = 0
        self.q = q
        self.qd = qd

        self.window_width = 800
        self.window_height = 600
        self.scale = 200
        self.camera_x = 0

        pygame.init()

        self.recording = False
        self.frame_index = 0
        self.paused = True
        self.clock = pygame.time.Clock()
        self.clock.tick(0)

        self.cart_color = [
            pygame.Color(200, 200, 200),
            pygame.Color(230, 230, 230)
            ]
        self.cart_height = 20
        self.cart_width = 80
        self.link_color = [
            pygame.Color(0, 0, 0),
            pygame.Color(200, 200, 200)
            ]
        self.link_width = 5
        self.weight_color = [
            pygame.Color(0, 66, 66),
            pygame.Color(0, 200, 200)
            ]
            
        self.weight_radius = 20

    def transform(self, point):
        x = point[0] 
        y = point[1]

        x = int(self.window_width/2 + self.scale*(x - self.camera_x))
        y = int(self.window_height/2 - self.scale*y)
        return (x,y)

    def draw_cart(self, q, colors=0):
        self.system.q = q

        cart_pos = self.transform(self.system.get_frame('Cart').p())
        pygame.draw.rect(self.screen, self.cart_color[colors],
                         pygame.Rect(cart_pos[0] - self.cart_width/2,
                                     cart_pos[1] - self.cart_height/2,
                                     self.cart_width, self.cart_height))
        pygame.draw.line(self.screen, self.link_color[colors],
                         self.transform(self.system.get_frame('Cart').p()),
                         self.transform(self.system.get_frame('Pendulum').p()),
                         self.link_width)
        pygame.draw.circle(self.screen, self.link_color[colors],
                           self.transform(self.system.get_frame('Cart').p()),
                           self.link_width/2)
        pygame.draw.circle(self.screen, self.weight_color[colors],
                           self.transform(self.system.get_frame('Pendulum').p()),
                           self.weight_radius)

    def draw_ground(self):
        color1 = pygame.Color(100, 100, 100)
        period = 50.0 # pixels
        color2 = pygame.Color(150, 150, 150)
        duty = 25.0 # pixels
        slant = 5.0 # pixels
        thickness = 10
        top = self.window_height/2 - thickness/2

        pygame.draw.rect(self.screen, color1,
                         pygame.Rect(0, top, self.window_width, thickness))

        # Draw alternating rectangles to give the appearance of movement.
        left_edge = -self.scale*self.camera_x
        i0 = int(math.floor(left_edge/period))
        i1 = int(math.ceil(i0 + self.window_width/period))

        for i in range(i0-1, i1+1):
            #pygame.draw.rect(screen, color2, pygame.Rect(i*period - left_edge, top,
            #                                             duty, thickness))
            x = i*period - left_edge
            pygame.draw.polygon(self.screen, color2, (
                (x, top), (x+duty, top),
                (x+duty+slant, top+thickness-1), (x+slant, top+thickness-1)))

    def draw_time(self):
        # Create a font
        font = pygame.font.Font(None, 17)
        
        # Render the text
        txt = 't = %4.2f' % self.t[self.frame_index]
        text = font.render(txt , True, (0, 0, 0))

        # Create a rectangle
        textRect = text.get_rect()
        
        # Center the rectangle
        textRect.right = self.window_width - 10
        textRect.top = 10
        #textRect.centerx = screen.get_rect().centerx
        #textRect.centery = screen.get_rect().centery
        # Blit the text
        self.screen.blit(text, textRect)

        txt = 'Trajectory %d/%d' % (self.q_index+1, len(self.q))
        text = font.render(txt , True, (0, 0, 0))
        textRect2 = text.get_rect()

        textRect2.right = textRect.right
        textRect2.top = textRect.bottom + 5
        self.screen.blit(text, textRect2)
        
    def togglepause(self):
        if self.recording:
            return
        
        if self.paused:
            self.paused = False
            self.clock.tick(0) # Reset the clock start time
        elif self.paused == False and self.frame_index == len(self.t)-1:
            self.frame_index = 0
        else:
            self.paused = True

    def record(self):
        if self.recording:
            self.recording = False
            return

        self.frame_index = 0
        self.recording = True
        
    def main(self):
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption('trajectory optimization')
        self.screen = pygame.display.get_surface()
        self.run_viewer = True        
        
        while self.run_viewer:
            for event in pygame.event.get(): 
                if event.type == pygame.QUIT or \
                   (event.type == pygame.KEYDOWN and event.key == pygame.K_q) or \
                   (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.run_viewer = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.togglepause()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    if not self.recording:
                        self.frame_index = 0
                if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    self.record()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                    self.q_index = max(0, self.q_index-1)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                    self.q_index = min(len(self.q)-1, self.q_index+1)
                    

            self.screen.fill(pygame.Color('white'))
            self.draw_ground()
            self.draw_cart(self.qd[self.frame_index], 1)
            self.draw_cart(self.q[self.q_index][self.frame_index], 0)
            self.draw_time()
            pygame.display.flip()

            if not self.recording:
                if not self.paused:
                    if self.frame_index < len(self.t)-1:
                        self.frame_index += 1
                        if self.frame_index < len(self.t):
                            delay = (self.t[self.frame_index]-self.t[self.frame_index-1])
                        else:
                            delay = 1.0/60.0
                else:
                    delay = 1/60.0
                self.clock.tick(1.0/delay)
            else:
                pygame.image.save(self.screen, 'frame-%04d.png' % self.frame_index)
                if self.frame_index < len(self.t)-1:
                    self.frame_index += 1
                else:
                    self.recording = False
                    self.frame_index = 0




def build_system(torque_force=False):
    cart_mass = 10.0
    pendulum_length = 1.0
    pendulum_mass = 1.0

    system = trep.System()
    frames = [
        tx('x', name='Cart', mass=cart_mass), [
            rz('theta'), [
                ty(-pendulum_length, name="Pendulum", mass=pendulum_mass)]]]
    system.import_frames(frames)
    trep.potentials.Gravity(system, (0, -9.8, 0))
    #trep.forces.Damping(system, 0.01)
    trep.forces.JointForce(system, 'x', 'x-force')
    if torque_force:
        trep.forces.JointForce(system, 'theta', 'theta-force')
    return system

def generate_desired_trajectory(system, t, amp=130*mpi/180):
    qd = np.zeros((len(t), system.nQ))
    theta_index = system.get_config('theta').index
    for i,t in enumerate(t):
        if t >= 3.0 and t <= 7.0:
            qd[i, theta_index] = (1 - cos(2*mpi/4*(t-3.0)))*amp/2
    return qd

def make_state_cost(base, x, theta):
    weight = base*np.ones((dsys.nX,))
    weight[system.get_config('x').index] = x
    weight[system.get_config('theta').index] = theta
    return np.diag(weight)

def make_input_cost(base, x, theta=None):
    weight = base*np.ones((dsys.nU,))
    if theta is not None:
        weight[system.get_input('theta-force').index] = theta
    weight[system.get_input('x-force').index] = x
    return np.diag(weight)                    


# START OF CODE
system = build_system(True)
mvi = trep.MidpointVI(system)
t = np.arange(0.0, 10.0, 0.01)
dsys = trep.DSystem(mvi, t)

# Generate an initial trajectory
(X,U) = dsys.build_trajectory()
for k in range(dsys.kf()):
    if k == 0:
        dsys.set(X[k], U[k], 0)
    else:
        dsys.step(U[k])
    X[k+1] = dsys.f()
    
## class Monitor(trep.doptimizer.DOptimizerMonitor):
##     def __init__(self):
##         self.dcosts = []
##         self.methods = []
##     def calculated_descent_direction(self, iterations, optimizer, method, dxi, dcost):
##         self.dcosts.append(dcost)
##         self.methods.append(method)


# Generate cost function
qd = generate_desired_trajectory(system, t, 130*mpi/180)
(Xd, Ud) = dsys.build_trajectory(qd)

Qcost = make_state_cost(0.01, 0.01, 100.0)
Rcost = make_input_cost(0.01, 0.01, 0.01)
cost = trep.DCost(Xd, Ud, Qcost, Rcost)

#monitor = Monitor()
optimizer = trep.DOptimizer(dsys, cost)#, monitor)

xi = [(X, U)]

optimizer.first_order_iterations = 6

## cost.R = make_input_cost(0.001, 0.001, 0.0001)
xi.append(optimizer.optimize(xi[-1], 2)[1:])
#print "Cost after first round: ", optimizer.cost_n

## import matplotlib.pyplot as pyplot
## pyplot.subplot(3,1,1)
## optimizer.descent_plot(*xi[-1], method='steepest', legend=False)
## pyplot.subplot(3,1,2)
## optimizer.descent_plot(*xi[-1], method='quasi', legend=False)
## pyplot.subplot(3,1,3)
## optimizer.descent_plot(*xi[-1], method='newton', legend=False)
## pyplot.show()

print "dcost, steepest"
print optimizer.check_dcost(*xi[-1], delta=1e-4)
print optimizer.check_dcost(*xi[-1], delta=1e-5)
print optimizer.check_dcost(*xi[-1], delta=1e-6)
print optimizer.check_dcost(*xi[-1], delta=1e-7)
print 
print "dcost, newton"
print optimizer.check_dcost(*xi[-1], method='newton', delta=1e-4)
print optimizer.check_dcost(*xi[-1], method='newton', delta=1e-5)
print optimizer.check_dcost(*xi[-1], method='newton', delta=1e-6)
print optimizer.check_dcost(*xi[-1], method='newton', delta=1e-7)

print
print "ddcost, steepest"
print optimizer.check_ddcost(*xi[-1], delta=1e-4)
print optimizer.check_ddcost(*xi[-1], delta=1e-5)
print optimizer.check_ddcost(*xi[-1], delta=1e-6)
print optimizer.check_ddcost(*xi[-1], delta=1e-7)
print
print "ddcost, newton"
print optimizer.check_ddcost(*xi[-1], method='newton', delta=1e-4)
print optimizer.check_ddcost(*xi[-1], method='newton', delta=1e-5)
print optimizer.check_ddcost(*xi[-1], method='newton', delta=1e-6)
print optimizer.check_ddcost(*xi[-1], method='newton', delta=1e-7)

## X,U = xi[-1]

## k = len(X)/2
## xk = X[k]
## uk = U[k]

## print 'fdx:   ', dsys.check_fdx(xk, uk, k)
## print 'fdu:   ', dsys.check_fdu(xk, uk, k)

## print 'fdxdx: ', dsys.check_fdxdx(xk, uk, k)
## print 'fdxdu: ', dsys.check_fdxdu(xk, uk, k)
## print 'fdudu: ', dsys.check_fdudu(xk, uk, k)


## (Kproj, dX, dU) = optimizer.calc_descent_direction(X, U)


## q_history = []
## for zi in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
##     (nX, nU) = optimizer.project(X + zi*dX,
##                                U + zi*dU,
##                                Kproj)

##     q,p,v,u,rho = dsys.split_trajectory(nX, nU)
##     q_history.append(q)
    
## view = Viewer(system, t, q_history, qd)
## view.main()






## f = open('converge.csv', 'wt')
## for dcost in monitor.dcosts:
##     f.write('%r\n' % dcost)
## f.close()


## cost.R = matrix(numpy.diag([0.001, 0.01]))
## xi.append(optimizer.optimize(xi[-1], 100))
## print "Cost after second round: ", optimizer.cost_n

## cost.R = matrix(numpy.diag([0.001, 0.1]))
## xi.append(optimizer.optimize(xi[-1], 100))
## print "Cost after third round: ", optimizer.cost_n

## cost.R = matrix(numpy.diag([0.001, 1]))
## xi.append(optimizer.optimize(xi[-1], 100))
## print "Cost after fourth round: ", optimizer.cost_n

## cost.R = matrix(numpy.diag([0.001, 10]))
## xi.append(optimizer.optimize(xi[-1], 100))
## print "Cost after fifth round: ", optimizer.cost_n

## cost.R = matrix(numpy.diag([0.001, 100]))
## xi.append(optimizer.optimize(xi[-1], 100))
## print "Cost after sixth round: ", optimizer.cost_n

## cost.R = matrix(numpy.diag([0.001, 1000]))
## xi.append(optimizer.optimize(xi[-1], 100))
## print "Cost after 7th round: ", optimizer.cost_n

## cost.R = matrix(numpy.diag([0.001, 10000]))
## xi.append(optimizer.optimize(xi[-1], 100))
## print "Cost after 8th round: ", optimizer.cost_n

## cost.R = matrix(numpy.diag([0.001, 100000]))
## xi.append(optimizer.optimize(xi[-1], 200))
## print "Cost after 9th round: ", optimizer.cost_n

## ## ## dump = open('cart-opt-1.txt', 'wt')
## ## ## for (x,u) in zip(*xi[-1]):
## ## ##     dump.write(', '.join(['%f' % xii for xii in (x.T.tolist()[0] + u.T.tolist()[0])]) + '\n')
## ## ## dump.close()

## # Convert the last trajectory 
## (q0, p0, v0, mu0, rho0) = dsys.split_trajectory(*xi[-1])

## # Build a new system without the extra input
## system = build_system(False)
## mvi = trep.MidpointVI(system)
## dsys = trep.DSystem(mvi, t)

## # Build the trajectory for this new system
## mu0 = [ui[:len(system.inputs)] for ui in mu0]  # Take the inputs we still have
## (x0,u) = dsys.build_trajectory(q0, p0, v0, mu0, rho0)

## # Generate an initial trajectory
## x = [x0[0]]
## for k in range(dsys.kf()):
##     if k == 0:
##         dsys.set(x[-1], u[k], 0)
##     else:
##         dsys.step(u[k])
##     x.append(dsys.f())

## xi.append((x,u))

## # Generate cost function
## (xd, ud) = generate_desired_trajectory(system, t, 130*mpi/180.0)
## xi.append( generate_desired_trajectory(system, t, 85*mpi/180.0) )

## weight = [1.0]*dsys.nX
## weight[system.configs.index(system.get_config('x'))] = 0.01
## weight[system.configs.index(system.get_config('theta'))] = 100.0
## Qcost = matrix(numpy.diag(weight))
## Rcost = matrix(numpy.diag([0.001]))
## cost = opt.TrajectoryTrackingCost(xd, ud, Qcost, Rcost)

## optimizer = opt.TrajectoryOptimization(dsys, cost)
## optimizer.shitcock = True
## optimizer.first_order_iterations = 20

## xi.append(optimizer.optimize(xi[-1], 100))
## xi.append(optimizer.optimize(xi[-1], 100))
## print "Cost for true system (1): ", optimizer.cost_n

## ## xi.append(optimizer.optimize(xi[-1], 200))
## ## print "Cost for true system (2): ", optimizer.cost_n
## ## xi.append(optimizer.optimize(xi[-1], 600))
## ## print "Cost for true system (3): ", optimizer.cost_n
## ## xi.append(optimizer.optimize(xi[-1], 1000))
## ## print "Cost for true system (4): ", optimizer.cost_n

## out = open('cost-2.0-order.txt', 'wt')
## for ci in optimizer.cost_n:
##     out.write('%f\n' % ci)
## out.close()

#(xd, ud) = generate_desired_trajectory(system, t)



q_history = []
for X,U in xi:
    q,p,v,u,rho = dsys.split_trajectory(X, U)
    q_history.append(q)

## from viewer import Viewer
## view = Viewer(system, t, q_history, qd)
## view.main()


