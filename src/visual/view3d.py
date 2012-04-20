import math
import numpy as np
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtOpenGL import *
from OpenGL.GL import *
from OpenGL.GLU import *



def se3_rx(angle):
    return np.array([
        [1, 0, 0, 0],
        [0, math.cos(angle), -math.sin(angle), 0],
        [0, math.sin(angle),  math.cos(angle), 0],
        [0, 0, 0, 1]])
                    
def se3_ry(angle):
    return np.array([
        [ math.cos(angle), 0, math.sin(angle), 0],
        [0, 1, 0, 0],
        [-math.sin(angle), 0, math.cos(angle), 0],
        [0, 0, 0, 1]])

def se3_rz(angle):
    return np.array([
        [math.cos(angle), -math.sin(angle), 0, 0],
        [math.sin(angle),  math.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    
class Camera(object):
    # The local x axis points along the camera's view
    # The local y axis points to the left
    # The local z axis points up

    
    def __init__(self):

        # yaw, pitch, roll
        self._ang = np.array([math.pi, 0.0, 0.0])
        self._pos = np.array([0.0, 0.0, 0.0])

        self._pos = np.array([5.0, 0.0, 0.0])

        # When I stand on our world frame, I want the
        #    X axis to point in front of me
        #    Y axis to point to my left
        #    Z axis to point up
        forward = np.array([1,0,0])
        left = np.array([0,1,0])
        up = np.array([0,0,1])

        # When I stand on the default OpenGL orientation, the
        #    X axis points to my right
        #    Y axis points up
        #    Z axis points behind of me
        world = np.eye(4)
        world[:3,0] = -left
        world[:3,1] = up
        world[:3,2] = -forward

        self._world = np.linalg.inv(world)
    
    def worldFrame(self):
        return self._world

        #self._world = np.linalg.inv(world).flatten('F')

    def orientTransform(self):
        frame = np.eye(4)
        frame = np.dot(frame, se3_rz(self._ang[0]))
        frame = np.dot(frame, se3_ry(self._ang[1]))
        frame = np.dot(frame, se3_rx(self._ang[2]))
        return frame

    def cameraFrame(self):
        frame = np.eye(4)
        frame[:3,3] = self._pos
        frame = np.dot(frame, self.orientTransform())
        return frame

    def adjustYaw(self, amount):
        self._ang[0] += amount

    def adjustPitch(self, amount):
        self._ang[1] += amount

    def adjustRoll(self, amount):
        self._ang[2] += amount

    def move(self, x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0, roll=0.0):
        delta = np.array([x,y,z, 0.0])
        trans = self.orientTransform()
        delta = np.dot(trans, delta)
        self._pos += delta[:3]
        self._ang += [yaw, pitch, roll]
        
    


class View3D(QGLWidget):
    MOVEMENT_KEYS = {
        # Forward/Back
        Qt.Key_W : (0, 1),
        Qt.Key_S : (0,-1),

        # Left/Right
        Qt.Key_A : (1, 1),
        Qt.Key_D : (1,-1),

        # Up/Down
        Qt.Key_Q : (2, 1),
        Qt.Key_E : (2,-1),

        # Yaw
        Qt.Key_Left  : (3, 1),
        Qt.Key_Right : (3,-1),

        # Pitch
        Qt.Key_Down : (4, 1),
        Qt.Key_Up   : (4,-1),
        }

    def __init__(self, parent=None):
        super(View3D, self).__init__(parent)
        self._scene = 0

        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.ClickFocus)
        
        self._viewaspect = 1.0
        
        self.camera = Camera()
        self._moving = [0,0,0,0,0,0]
        self._movingTimerId = None
        self._movingRate = 40
        self._movingSpeed = 1.0

    
        
    def setScene(self, scene):
        self._scene = scene
        self._scene.frameTimeChanged.connect(self.update)
        self._scene.itemsChanged.connect(self.update)


    def scene(self):
        return self._scene


    def screenshot(self):
        return self.renderPixmap()

        
    def initializeGL(self):
        glClearColor(0.4, 0.4, 0.6, 1.0)

        lightPos = (5.0, 8.0, 2.0, 1.0)

        glLightfv(GL_LIGHT0, GL_POSITION, lightPos)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.7, 0.7, 0.7, 1.0))
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_NORMALIZE)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)


    def startMoving(self):
        if self._movingTimerId is not None:
            return
        self._movingTimerId = self.startTimer(self._movingRate)


    def stopMoving(self):
        if any(self._moving) or self._movingTimerId is None:
            return
        self.killTimer(self._movingTimerId)
        self._movingTimerId = None


    def timerEvent(self, event):
        if event.timerId() == self._movingTimerId:
            delta = np.array(self._moving)
            delta = self._movingRate/1000.0 * self._movingSpeed * delta
            self.camera.move(*delta)
            self.update()
            
        
    def keyPressEvent(self, event):
        if not event.isAutoRepeat():
            for key,info in self.MOVEMENT_KEYS.iteritems():
                if event.key() == key:
                    self._moving[info[0]] = info[1]
                    self.startMoving()
                    event.accept()
                    return
        super(View3D, self).keyPressEvent(event)


    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat():
            for key,info in self.MOVEMENT_KEYS.iteritems():
                if event.key() == key and self._moving[info[0]] == info[1]:
                    self._moving[info[0]] = 0
                    self.stopMoving()
                    event.accept()
                    return
        super(View3D, self).keyPressEvent(event)


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._oldPos = event.pos()
            self.grabKeyboard()


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._oldPos = event.pos()
            self.releaseKeyboard()
            self.stopMoving()


    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            diff = event.pos() - self._oldPos
            self._oldPos = event.pos()
            dx = float(diff.x())
            dy = float(diff.y())
            self.camera.adjustYaw(-0.005*dx)
            self.camera.adjustPitch(0.005*dy)
            self.update()
            

    def draw_coord_frame(self):
        glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT )
        glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 1.0)
        glEnd()
        glPopAttrib()


    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, self._viewaspect, 0.01, 100.0)        

        glMatrixMode(GL_MODELVIEW)     
        glLoadIdentity()

        # For developing/debuging camera movement, change this to True
        if False:
            gluLookAt(3,3,5, 0,0,0,  0,1,0)
            glMultMatrixf(self.camera.worldFrame().flatten('F'))
            self.draw_coord_frame()
            glMultMatrixf(self.camera.cameraFrame().flatten('F'))
            self.draw_coord_frame()
        else:
            glMultMatrixf(self.camera.worldFrame().flatten('F'))
            glMultMatrixf(np.linalg.inv(self.camera.cameraFrame()).flatten('F'))

        # Draw the system
        time = self.scene().time()
        for item in self.scene().items():
            item.setTime(time)
            glPushMatrix()
            item.draw()
            glPopMatrix()
        

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        if height:
            self._viewaspect = float(width)/height
        else:
            self._viewaspect = 1.0

