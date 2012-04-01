#!/usr/bin/python

# This example models a radial engine with a variable number of
# cylinders.  We use a trivial and unrealistic model to calculate the
# combustion force as a function of the crank shaft angle.

import sys
import trep
from trep import tx, ty, tz, rx, ry, rz
import trep.visual as visual
import math
import numpy as np
from OpenGL.GL import *

from piston import PistonForce
# Force a specific implementation
#import piston.PyPistonForce as PistonForce
#import piston.CPistonForce as PistonForce

tf = 30.0
dt = 0.01


CRANK_OFFSET=0.2
B = 0.1 # Radius of the circle that the rods connect to.
C = 0.2 # Half length of a piston
ROD_LENGTH=0.5

# To make the system definition easier, we just use the same inertia
# properties for every component in the system.
m0=(1,1,1,1)

# This spline calculates the combusion force as a function of the
# crank shaft angle.  See the spline.py example to see how to plot
# this.
piston_force_curve = [
    (0.0*math.pi, 0.0, 0, 0),
    (0.25*math.pi, 1.0),
    (0.75*math.pi, 1.0),
    (1.0*math.pi, 0.0, 0, 0),
    (2.0*math.pi, 0.0, 0, 0),
    ]

    

class RadialEngine(trep.System):
    def __init__(self, cylinders=5):
        super(RadialEngine, self).__init__()

        # This is the spline we use to define a combustion force.
        self.spline = trep.Spline(piston_force_curve)

        # Create the crank shaft and master rod
        frames = [
            rx('crank-theta', mass=m0, name='crank-shaft'), [
                tz(CRANK_OFFSET, name='master-origin'), [
                    rx('rod0-theta', name='master-pivot'), [
                        tz(B), [
                            tz(ROD_LENGTH/2, mass=m0),
                            tz(ROD_LENGTH),               
                            [
                                rx('piston0-theta', name='piston0-base'), [
                                    tz(C, name='piston0-center', mass=m0)
                                    ]]],
                        ],            
                    ],
                ],    
            rx(0.0, name='cylinder0'), [tz(CRANK_OFFSET/2)],
            ]
        # Add the frames to the system.
        self.import_frames(frames)

        # Add some damping
        trep.forces.Damping(self, 0.2)

        # Constrain the master piston
        trep.constraints.PointOnPlane(self, 'cylinder0', (0,1,0), 'piston0-base')
        trep.constraints.PointOnPlane(self, 'cylinder0', (0,1,0), 'piston0-center')

        # Add the combustion force for the master piston.
        PistonForce(self, 'crank-theta', 0.0, 'piston0-center', self.spline)

        # Add the other pistons.
        for i in range(1, cylinders):
            self._add_piston(i,cylinders)



    def _add_piston(self, n, N):
        ang = n*2*math.pi/N
        frames = [
            rx(ang), [
                tz(B), [
                    rx('rod%d-theta' % n, name='rod%d-pivot' % n), [
                        tz(ROD_LENGTH/2, mass=m0),
                        tz(ROD_LENGTH), [
                            rx('piston%d-theta' % n, name='piston%d-base' % n), [
                                tz(C, name='piston%d-center' % n, mass=m0)
                                ]]]]]]
        # Attach the new rod and piston to the master rod's base
        self.get_frame('master-pivot').import_frames(frames)

        # Add a frame to define the axis of this piston's cylinder.
        # The tz() frame is unnecessary, but when using the auto
        # drawing visual model, it gives a visual indication of the
        # cylinder.
        frames = [rx(ang, name='cylinder%d' % n),[tz(CRANK_OFFSET/2)]]
        self.import_frames(frames)

        # Constrain the piston to the cylinder.
        trep.constraints.PointOnPlane(self, 'cylinder%d' % n, (0,1,0), 'piston%d-base' % n)
        trep.constraints.PointOnPlane(self, 'cylinder%d' % n, (0,1,0), 'piston%d-center' % n)

        # Add the combustion force.
        PistonForce(self, 'crank-theta', ang, 'piston%d-center' % n, self.spline)




# Create a radial engine with 5 pistons
system = RadialEngine(5)

# Find a configuratoin that satisfies the constraints.
system.satisfy_constraints()

## print "here"
## for f in system.forces:
##     print f.validate_f_dq(verbose=True)

## sys.exit(1)


# Create and initialize the variational integrator
mvi = trep.MidpointVI(system)
q0 = system.q
mvi.initialize_from_configs(0.0, q0, dt, q0)


# This is our simulation loop.  We save the results in two lists.
q = [mvi.q2]
t = [mvi.t2]
try:
    while mvi.t1 < tf:
        mvi.step(mvi.t2+dt)
        q.append(mvi.q2)
        t.append(mvi.t2)
except trep.ConvergenceError:
    print "simulation failed at %f" % mvi.t1
        

# We can create a custom visualization for the radial engine.
class PistonVisual(visual.VisualItem3D):
    def __init__(self, *args, **kwds):
        super(PistonVisual, self).__init__(*args, **kwds)
        self.density = 5000

# Now we can show the simulation using the builtin basic visualizer.
visual.visualize_3d([PistonVisual(system, t, q)])



