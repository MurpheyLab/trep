import math
import numpy as np
from scipy.interpolate import interp1d
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from OpenGL.GL import *
from OpenGL.GLU import *

__all__ = ['VisualItem', 'VisualItem2D', 'VisualItem3D']

class VisualItem(object):
    def __init__(self, system, time=None, q=None, u=None):

        if time is None: time = []
        if q is None: q = []
        if u is None: u = []
        
        self._system = system
        self._time = np.array(time)
        self._q = np.array(q)
        self._u = np.array(u)
        
        if len(self._q):
            self._q_interp = interp1d(self._time, self._q, axis=0)
        else:
            self._q_interp = None

        if len(self._u):
            self._u_interp = interp1d(self._time[:-1], self._u, axis=0)
        else:
            self._u_interp = None
            
        self._draw_funcs = []


    @property
    def system(self):
        return self._system


    def attachDrawing(self, frame, func):
        self._draw_funcs.append( (self._system.get_frame(frame), func) )

        
    def draw(self, painter=None):
        raise NotImplemented("Please inherit from VisualItem2D or VisualItem3D")


    def setTime(self, t):
        if len(self._time) == 0:
            return

        if t > self._time.max():
            t = self._time.max()
        elif t < self._time.min():
            t = self._time.min()

        self._system.t = t
        if self._q_interp is not None:
            self._system.q = self._q_interp(t)
        if self._u_interp is not None:
            self._system.u = self._u_interp(t)


    def getTimeRange(self):
        if len(self._time) == 0:
            return None

        return (self._time.min(), self._time.max(), self._time)


class VisualItem2D(VisualItem):

    def __init__(self, *args, **kwds):
        plane = kwds.setdefault('plane', 'XY')
        del kwds['plane']
        super(VisualItem2D, self).__init__(*args, **kwds)

        self.plane = plane
        

    def getTransform(self, frame):
        self.density = 5
        if frame is None:
            return QTransform()
        t = frame.g().T

        i1 = "XYZ".index(self.plane[0])
        i2 = "XYZ".index(self.plane[1])
        
        return QTransform(t[i1,i1], t[i1,i2], 0.0,
                          t[i2,i1], t[i2,i2], 0.0,
                          t[ 3,i1], t[ 3,i2], 1.0)

        
    def draw(self, painter):

        if len(self._draw_funcs) > 0:
            base_transform = painter.transform()
            for frame, func in self._draw_funcs:            
                transform = self.getTransform(frame) * base_transform
                painter.setTransform(transform, False)
                func(painter)
            painter.setTransform(base_transform, False)
        else:
            self.auto_draw(painter)


    def auto_draw(self, painter):

        for frame in self.system.frames:

            frame_g = self.getTransform(frame)
            p1 = QPointF(frame_g.dx(), frame_g.dy())
                
            if frame.parent != None:
                parent_g = self.getTransform(frame.parent)
                p2 = QPointF(parent_g.dx(), parent_g.dy())

                painter.drawLine(p1, p2)

            if frame.mass != 0.0:                
                r = (3.0/4.0 * frame.mass / (self.density*math.pi))**(1.0/2.0)
                painter.drawEllipse(p1, r, r)

        ## for part in (self.system.constraints +
        ##              self.system.potentials +
        ##              self.system.forces):
        ##     part.opengl_draw()



class VisualItem3D(VisualItem):
    def __init__(self, *args, **kwds):
        self.density = kwds.setdefault('density', 50)
        del kwds['density']

        self.offset = kwds.setdefault('offset', None)
        del kwds['offset']

        super(VisualItem3D, self).__init__(*args, **kwds)

        self.setOrientation(forward=[1,0,0], up=[0,0,1])

        
    def setOrientation(self, forward, up):
        up = np.array(up, dtype=np.float)
        forward = np.array(forward, dtype=np.float)

        right = np.cross(up, forward)
        forward = np.cross(right, up)

        forward = forward/np.linalg.norm(forward)
        right = right/np.linalg.norm(right)
        up = up/np.linalg.norm(up)

        world = np.eye(4)
        world[:3,0] = forward
        world[:3,1] = right
        world[:3,2] = up
        self._orientation = np.linalg.inv(world)


    def orientation(self):
        return self._orientation

    
    def draw(self):

        glPushMatrix()
        if self.offset is not None:
            glTranslate(*self.offset)

        glMultMatrixf(self._orientation.flatten('F'))

        if len(self._draw_funcs) > 0:
            for frame, func in self._draw_funcs:
                if frame:
                    glPushMatrix()
                    frame_g = frame.g()
                    glMultMatrixf(frame_g.flatten('F'))
                    func()
                    glPopMatrix()
                else:
                    func()
        else:
            self.auto_draw()

        glPopMatrix()


    def auto_draw(self):
        quad = gluNewQuadric()
        gluQuadricNormals(quad, GLU_SMOOTH)

        for frame in self.system.frames:
            frame_g = frame.g()
            if frame.parent != None:
                parent_g = frame.parent.g()

                glPushAttrib(GL_LIGHTING_BIT )
                glDisable(GL_LIGHTING)
                glBegin(GL_LINES)
                glVertex3d(parent_g[0][3],
                           parent_g[1][3],
                           parent_g[2][3])
                glVertex3d(frame_g[0][3],
                           frame_g[1][3],
                           frame_g[2][3])    
                glEnd()
                glPopAttrib()

            if frame.mass != 0.0:
                glPushMatrix()
                glMultMatrixf(frame_g.flatten('F'))
                r = (3.0/4.0 * frame.mass / (self.density*math.pi))**(1.0/3.0)
                gluSphere(quad, r, 10, 10)
                glPopMatrix()

        for part in (self.system.constraints +
                     self.system.potentials +
                     self.system.forces):
            part.opengl_draw()

        gluDeleteQuadric(quad)
