import sys
import math
from math import sin, cos
from math import pi as mpi
import trep


try:
    import Image
except ImportError:
    Image = None

try:
    from OpenGL.GLUT import *
    from OpenGL.GLU import *
    from OpenGL.GL import *
except:
    print "ERROR: PyOpenGL is not installed properly."
    sys.exit()


def gl_flatten_matrix(matrix):
    """
    Flatten a 4x4 matrix into a column-major 16 value array
    representation.
    """
    temp = []
    for c in range(4):
        for r in range(4):
            temp.append(matrix[r][c])
    return temp
        

def draw_coordinate_frame():
    """
    Draw a coordinate frame representation.  Draws a red line in the
    x-axis direction, a green line in the y-axis direction, and a blue
    line in the z-axis direction (RGB->XYZ), all starting at the
    current origin.
    """
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

class Camera_3D:
    def __init__(self):
        self.viewer = None
        
        # Values used for the camera system
        self.camera_pos = [0, -15, 0]
        self.camera_ang = [-90, 0.0, 0.0]
        self.camera_ang_dt = [0.0, 0.0, 0.0]
        self.camera_pos_dt = [0.0, 0.0, 0.0]

        # Damping constant (-c*v model) for camera angular velocity
        self.camera_ang_damping = 12.0
        # When the angular velocity goes below the minimum, it is
        # clamped to 0.0
        self.camera_ang_dt_min = 1.0
        # Angular acceleration rate when the user moves the camera
        self.camera_ang_dt_accel = 1000.0
        # Maximum angular velocity
        self.camera_ang_dt_max = 125.0
        # Damping constant (-c*v model) for camera linear velocity
        self.camera_pos_damping = 6.0
        # Same as angular acceleration values
        self.camera_pos_dt_min = 0.2
        self.camera_pos_dt_accel = 15.0
        self.camera_pos_dt_max = 3.0
      
        # OpenGL-related values
        self.fov = 60.0
        self.clipping_near = 0.1
        self.clipping_far = 100.0
        self.visible = False

        # Setup input variables
        self.last_mouse_pos = None

    def setup_projection_matrix(self):
        """
        Setup the projection matrix.
        """
        gluPerspective(self.fov,
                       float(self.viewer.window_width) /
                         float(self.viewer.window_height),
                       self.clipping_near,
                       self.clipping_far)

    def reshape(self):
        """
        GLUT reshape callback function
        """
        glViewport(0, 0, self.viewer.window_width, self.viewer.window_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        self.setup_projection_matrix()
        glMatrixMode(GL_MODELVIEW)

    def apply_matrix(self):
        # Orient world frame to have x-axis towards viewer, y-axis to the right,and z-axis up
        glRotated(-90.0, 1.0, 0.0, 0.0)
        glRotated(-90.0, 0.0, 0.0, 1.0)
        # Camera transformations
        glRotated(-self.camera_ang[2], 1.0, 0.0, 0.0)
        glRotated(-self.camera_ang[1], 0.0, 1.0, 0.0)
        glRotated(-self.camera_ang[0], 0.0, 0.0, 1.0)
        glTranslated(-self.camera_pos[0],
                     -self.camera_pos[1],
                     -self.camera_pos[2])

    def load_matrix(self):
        glLoadIdentity()
        self.apply_matrix()
        
    def run_physics(self, dt):
        """
        Run the camera physics to determine new camera position.
        """
        if self.viewer.mouse_button_state(GLUT_RIGHT_BUTTON):
            if self.last_mouse_pos == None:
                self.last_mouse_pos = self.viewer.mouse_position
            d_x = (self.viewer.mouse_position[0] -
                   self.last_mouse_pos[0])
            d_y = (self.viewer.mouse_position[1] -
                   self.last_mouse_pos[1])
            self.last_mouse_pos = self.viewer.mouse_position
            
            # Mouse navigation bypasses the physics and directly affects
            # positions.
            self.camera_ang_dt = [0.0, 0.0, 0.0]
            self.camera_ang[0] += -0.3*d_x
            self.camera_ang[1] += -0.3*d_y
        else:
            self.last_mouse_pos = None
    
        # For each input action, we either accelerate and clamp to the
        # maximum range, or just deaccelerate from damping
        if self.viewer.key_state('q'):
            self.camera_pos_dt[2] = min(
                self.camera_pos_dt[2] + self.camera_pos_dt_accel*dt, 
                self.camera_pos_dt_max)
        elif self.viewer.key_state('e'):
            self.camera_pos_dt[2] = max(
                self.camera_pos_dt[2] - self.camera_pos_dt_accel*dt,
                -self.camera_pos_dt_max)
        else:
            self.camera_pos_dt[2] -= (self.camera_pos_damping
                                      *self.camera_pos_dt[2]*dt)

        if self.viewer.key_state('w'):
            self.camera_pos_dt[0] = max(
                self.camera_pos_dt[0] - self.camera_pos_dt_accel*dt,
                -self.camera_pos_dt_max)
        elif self.viewer.key_state('s'):
            self.camera_pos_dt[0] = min(
                self.camera_pos_dt[0] + self.camera_pos_dt_accel*dt,
                self.camera_pos_dt_max)
        else:
            self.camera_pos_dt[0] -= (self.camera_pos_damping
                                      *self.camera_pos_dt[0]*dt)

        if self.viewer.key_state('d'):
            self.camera_pos_dt[1] = min(
                self.camera_pos_dt[1] + self.camera_pos_dt_accel*dt,
                self.camera_pos_dt_max)
        elif self.viewer.key_state('a'):
            self.camera_pos_dt[1] = max(
                self.camera_pos_dt[1] - self.camera_pos_dt_accel*dt,
                -self.camera_pos_dt_max)
        else:
            self.camera_pos_dt[1] -= (self.camera_pos_damping
                                      *self.camera_pos_dt[1]*dt)

        if self.viewer.key_state(GLUT_KEY_UP):
            self.camera_ang_dt[1] = min(
                self.camera_ang_dt[1] + self.camera_ang_dt_accel*dt,
                self.camera_ang_dt_max)
        elif self.viewer.key_state(GLUT_KEY_DOWN):
            self.camera_ang_dt[1] = max(
                self.camera_ang_dt[1] - self.camera_ang_dt_accel*dt,
                -self.camera_ang_dt_max)
        else:
            self.camera_ang_dt[1] -= (self.camera_ang_damping
                                      *self.camera_ang_dt[1]*dt)

        if self.viewer.key_state(GLUT_KEY_RIGHT):
            self.camera_ang_dt[0] = max(
                self.camera_ang_dt[0] - self.camera_ang_dt_accel*dt,
                -self.camera_ang_dt_max)
        elif self.viewer.key_state(GLUT_KEY_LEFT):
            self.camera_ang_dt[0] = min(
                self.camera_ang_dt[0] + self.camera_ang_dt_accel*dt,
                self.camera_ang_dt_max)
        else:
            self.camera_ang_dt[0] -= (self.camera_ang_damping
                                      *self.camera_ang_dt[0]*dt)

        # If velocities are really small, set them to zero.
        for i in xrange(3):
            if abs(self.camera_pos_dt[i]) < self.camera_pos_dt_min:
                self.camera_pos_dt[i] = 0.0
            if abs(self.camera_ang_dt[i]) < self.camera_ang_dt_min:
                self.camera_ang_dt[i] = 0.0

        # Angular positions are simple
        for i in xrange(3):
                self.camera_ang[i] += self.camera_ang_dt[i]*dt

        # Linear velocties have to be converted from body coordinates
        # to world coordinates
        yaw = math.radians(self.camera_ang[0])
        pitch = math.radians(self.camera_ang[1])
        roll = math.radians(self.camera_ang[2])
        x_dir = (cos(pitch)*cos(yaw),
                 cos(pitch)*sin(yaw),
                 -sin(pitch)
                 )
        y_dir = (
            cos(yaw)*sin(pitch)*sin(roll) - cos(roll)*sin(yaw),
            cos(roll)*cos(yaw) + sin(pitch)*sin(roll)*sin(yaw),
            cos(pitch)*sin(roll)
            )
        z_dir = (
            cos(roll)*cos(yaw)*sin(pitch) + sin(roll)*sin(yaw),
            -cos(yaw)*sin(roll) + cos(roll)*sin(pitch)*sin(yaw),
            cos(pitch)*cos(roll)
            )
        for i in xrange(3):
            self.camera_pos[i] += (self.camera_pos_dt[0]*dt*x_dir[i] +
                                   self.camera_pos_dt[1]*dt*y_dir[i] +
                                   self.camera_pos_dt[2]*dt*z_dir[i])


class Camera_2D:
    def __init__(self, **opts):
        self.viewer = None

        # Values used for the camera system
        self.camera_pos = [0.0, 0.0, 30.0]
        self.camera_pos_dt = [0.0, 0.0, 0.0]

        # Damping constant (-c*v model) for camera linear velocity
        self.camera_pos_damping = 6.0
        # When the linear velocity goes below the minimum, it is
        # clamped to 0.0
        self.camera_pos_dt_min = 0.2
        # Linear acceleration rate when the user moves the camera
        self.camera_pos_dt_accel = 15.0
        # Maximum linear velocity
        self.camera_pos_dt_max = 4.0
      
        # OpenGL-related values
        self.visible = False
    
        self.last_mouse_pos = None        
        
    def setup_projection_matrix(self):
        """
        Setup the projection matrix.
        """
        w = self.camera_pos[2]
        h = w*(float(self.viewer.window_height)/
                    float(self.viewer.window_width))
        gluOrtho2D(-w/2.0, w/2.0, -h/2.0, h/2.0)        

    def get_width(self):
        return self.camera_pos[2]

    def get_height(self):
        return  self.camera_pos[2]*(float(self.viewer.window_height)/
                                    float(self.viewer.window_width))
        
    def reshape(self):
        """
        GLUT reshape callback function
        """
        glViewport(0, 0, self.viewer.window_width, self.viewer.window_height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        self.setup_projection_matrix()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def load_matrix(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        self.setup_projection_matrix()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glRotatef(-90, 1.0, 0.0, 0.0)
        glTranslated(-self.camera_pos[0], 0.0,
                     -self.camera_pos[1])

    def run_physics(self, dt):
        """
        Run the camera physics to determine new camera position.
        """

        if self.viewer.mouse_button_state(GLUT_RIGHT_BUTTON):
            if self.last_mouse_pos == None:
                self.last_mouse_pos = self.mouse_position
            d_x = (self.viewer.mouse_current_pos[0] -
                   self.last_mouse_pos[0])
            d_y = (self.viewer.mouse_current_pos[1] -
                   self.last_mouse_pos[1])
            self.last_mouse_pos = self.viewer.mouse_position
            
            # Mouse navigation bypasses the physics and directly affects
            # positions.
            self.camera_pos_dt[0] = 0.0
            self.camera_pos_dt[1] = 0.0
            self.camera_pos[0] += -self.camera_pos[2]/self.viewer.window_width*d_x
            self.camera_pos[1] += self.camera_pos[2]/self.viewer.window_height*d_y
        else:
            self.last_mouse_pos = None

        # For each input action, we either accelerate and clamp to the
        # maximum range, or just deaccelerate from damping
        if self.viewer.key_state('q'):
            self.camera_pos_dt[1] = min(
                self.camera_pos_dt[1] + self.camera_pos_dt_accel*dt, 
                self.camera_pos_dt_max)
        elif self.viewer.key_state('e'):
            self.camera_pos_dt[1] = max(
                self.camera_pos_dt[1] - self.camera_pos_dt_accel*dt,
                -self.camera_pos_dt_max)
        else:
            self.camera_pos_dt[1] -= (self.camera_pos_damping
                                      *self.camera_pos_dt[1]*dt)

        if self.viewer.key_state('w'): 
            self.camera_pos_dt[2] = max(
                self.camera_pos_dt[2] - self.camera_pos_dt_accel*dt,
                -self.camera_pos_dt_max)
        elif self.viewer.key_state('s'):
            self.camera_pos_dt[2] = min(
                self.camera_pos_dt[2] + self.camera_pos_dt_accel*dt,
                self.camera_pos_dt_max)
        else:
            self.camera_pos_dt[2] -= (self.camera_pos_damping
                                      *self.camera_pos_dt[2]*dt)

        if self.viewer.key_state('d'):
            self.camera_pos_dt[0] = min(
                self.camera_pos_dt[0] + self.camera_pos_dt_accel*dt,
                self.camera_pos_dt_max)
        elif self.viewer.key_state('a'):
            self.camera_pos_dt[0] = max(
                self.camera_pos_dt[0] - self.camera_pos_dt_accel*dt,
                -self.camera_pos_dt_max)
        else:
            self.camera_pos_dt[0] -= (self.camera_pos_damping
                                      *self.camera_pos_dt[0]*dt)

        # If velocities are really small, set them to zero.
        for i in xrange(3):
            if abs(self.camera_pos_dt[i]) < self.camera_pos_dt_min:
                self.camera_pos_dt[i] = 0.0

        for i in xrange(3):
            self.camera_pos[i] += self.camera_pos_dt[i]*dt


class OpenGLViewer:
    """
    This class provides a basic OpenGL window with a movable camera.
    """
    
    def __init__(self,
                 display_dt=1/25.0,
                 camera=None,
                 window_width=600,
                 window_height=600):
        self.display_dt = display_dt

        if camera == None:
            self.camera = Camera_3D()
        else:
            self.camera = camera
        self.camera.viewer = self

        self.window_width = window_width
        self.window_height = window_height


        self.key_pressed = set() # Set of currently pressed keys
        self.key_handler = {} # Dictionary of (down, up) functions to
                              # call when a key changes state
        self.mouse_button_pressed = set()
        self.mouse_button_handler = {}
        self.mouse_position = (-1, -1)

        self.set_key_handler('p', lambda : self.screenshot('frame.png'))

        self.background_color = (0.0, 0.0, 0.0, 0.0)

    def set_key_handler(self, key, down_func, up_func=lambda : None):
        self.key_handler[key] = (down_func, up_func)
        
    def del_key_handler(self, key):
        if key in self.key_handler:
            del self.key_handler[key]
            
    def set_mouse_button_handler(self, button, down_func, up_func=lambda x,y: None):
        self.mouse_button_handler[button] = (down_func, up_func)
        
    def del_mouse_button_handler(self, button):
        if key in self.key_handler:
            del self.mouse_button_handler[button]
        
    def initialize_opengl(self):
        """
        Setup OpenGL
        """
        glClearColor (*self.background_color)
        glShadeModel (GL_SMOOTH)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_NORMALIZE)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)

        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0,1.0,1.0,1.0))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (1.0,1.0,1.0,1.0))

    def reshape(self, w,h):
        """
        GLUT reshape callback function
        """
        self.window_width = w
        self.window_height = h
        self.camera.reshape()

    def key_state(self, key):
        return key in self.key_pressed

    def mouse_button_state(self, button):
        return button in self.mouse_button_pressed

    def keyboard(self, key, x, y):
        """
        GLUT keyboard callback function (normal and special keys)
        """
        if isinstance(key, str):
            key = key.lower()
        if key == chr(27): #ESC
            sys.exit(0)
        self.key_pressed.add(key)
        if key in self.key_handler:
            self.key_handler[key][0]()


    def keyboard_up(self, key, x, y):
        """
        GLUT keyboard up callback function (normal and special keys)
        """
        if isinstance(key, str):
            key = key.lower()
        if key in self.key_pressed:
            self.key_pressed.remove(key)
        if key in self.key_handler:
            self.key_handler[key][1]()
            
    def mouse(self, button, state, m_x, m_y):
        """
        GLUT mouse button callback function
        """
        if state == GLUT_DOWN:
            self.mouse_button_pressed.add(button)
            if button in self.mouse_button_handler:
                self.mouse_button_handler[button][0](m_x, m_y)
        elif state == GLUT_UP:
            if button in self.mouse_button_pressed:
                self.mouse_button_pressed.remove(button)
            if button in self.mouse_button_handler:
                self.mouse_button_handler[button][1](m_x, m_y)

    def mouse_active_motion(self, m_x, m_y):
        """
        GLUT mouse active motion callback function
        """
        self.mouse_position = (m_x, m_y)

    def mouse_passive_motion(self, m_x, m_y):
        """
        GLUT mouse passive motion callback function
        """
        self.mouse_position = (m_x, m_y)
        
    def visibility(self, visible):
        """
        GLUT visibility callback function
        """
        if visible == GLUT_VISIBLE:
            self.visible = True
            glutPostRedisplay()
        else:
            self.visible = False

    def _display(self):
        """
        GLUT display callback function
        """
        glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.camera.load_matrix()
        self.display()
        glutSwapBuffers()


    def display(self):
        """
        This function is called to draw the scene.  The color and
        depth buffers are already cleared and the modelview matrix
        will be transformed according to the camera position and
        orientation.  The buffers will be swapped automatically when
        the function the returns.
        """
        draw_coordinate_frame()
        
        
    def run(self):
        # Start GLUT
        glutInitWindowSize(self.window_width, self.window_height)
        glutInit([]) #(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutCreateWindow('OpenGLWindow')
        self.initialize_opengl()

        # Setup callbacks
        glutReshapeFunc(self.reshape)
        glutKeyboardFunc(self.keyboard)
        glutSpecialFunc(self.keyboard)
        glutKeyboardUpFunc(self.keyboard_up)
        glutSpecialUpFunc(self.keyboard_up)
        glutMouseFunc(self.mouse)
        glutMotionFunc(self.mouse_active_motion)
        glutPassiveMotionFunc(self.mouse_passive_motion)
        glutVisibilityFunc(self.visibility)
        glutDisplayFunc(self._display)

        # Setup timers 
        glutTimerFunc(int(self.display_dt*1000), self._display_timer, 0)

        self.pre_mainloop()
        glutMainLoop()


    def pre_mainloop(self):
        """
        This function is called just before the GLUT main loop is started.
        """
        pass


    def _display_timer(self, timer_value):
        """
        The display timer runs the camera physics and updates the display
        """
        glutTimerFunc(int(self.display_dt*1000.0), self._display_timer, 0)
        self.camera.run_physics(self.display_dt)
        self.display_timer()

        if self.visible:
            glutPostRedisplay()


    def display_timer(self):
        """
        This function is called whenever the display timer is triggered.
        """
        pass

    if Image:
        def screenshot(self, filename):
            pixeldata = glReadPixels(0,0,
                                     self.window_width,
                                     self.window_height,
                                     GL_RGBA, GL_UNSIGNED_BYTE)
            im = Image.frombuffer("RGBA", (self.window_width,
                                           self.window_height),
                                  pixeldata, 'raw', "RGBA", 0, -1)
            im.save(filename)
    
    else:
        def screenshot(self, filename):
            raise StandardError("Python Image library is not properly installed.")
            

class SystemPainter:
    def __init__(self, system):
        self.system = system
        self.density = 50.0
        self.auto_draw = True
        self.auto_draw_skip = []
        self.scale = 1.0
        
        self.color = [0.5, 0.5, 0.5]
        
        self.display_funcs = []


    def add_display_func(self, frame, function):
        if frame == None:
            self.display_funcs.append((None, function))
        else:
            self.display_funcs.append((self.system.get_frame(frame), function))

    def draw_system(self):
        glPushAttrib(GL_CURRENT_BIT)            
        glColor3f(self.color[0],
                  self.color[1],
                  self.color[2])

        glPushMatrix()
        glScale(self.scale, self.scale, self.scale)
        
        if self.auto_draw:
            for frame in self.system.frames:
                if frame in self.auto_draw_skip:
                    continue                
                
                frame_g = frame.g()
                if frame.parent != None:
                    parent_g = frame.parent.g()
                    
                    glPushAttrib(GL_LIGHTING_BIT )
                    glDisable(GL_LIGHTING)
                    glBegin(GL_LINES)
                    glVertex3f(parent_g[0][3],
                               parent_g[1][3],
                               parent_g[2][3])
                    glVertex3f(frame_g[0][3],
                               frame_g[1][3],
                               frame_g[2][3])    
                    glEnd()
                    glPopAttrib()
                
                if frame.mass != 0.0:
                    glPushMatrix()
                    glMultMatrixf(gl_flatten_matrix(frame_g))
                    r = (3.0/4.0 * frame.mass / (self.density*mpi))**(1.0/3.0)
                    glutSolidSphere(r, 10, 10)
                    glPopMatrix()
                    
            for part in (self.system.constraints +
                         self.system.potentials +
                         self.system.forces):
                part.opengl_draw()

        for (frame, func) in self.display_funcs:
            if frame:
                glPushMatrix()
                frame_g = frame.g()
                glMultMatrixf(gl_flatten_matrix(frame_g))
                func()
                glPopMatrix()
            else:
                func()

        glPopMatrix()
        glPopAttrib()

            
class SystemViewer(OpenGLViewer, SystemPainter):
    def __init__(self, system,
                 display_dt=1.0/25.0,
                 camera=None,
                 window_width=600,
                 window_height=600):
        OpenGLViewer.__init__(self,
                              display_dt=display_dt,
                              camera=camera,
                              window_width=window_width,
                              window_height=window_height)
        SystemPainter.__init__(self, system)

    def _pre_mainloop(self):
        pass


    def display(self):
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0,1.0,1.0,1.0))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (1.0,1.0,1.0,1.0))

        glPushMatrix()
        glTranslate(5.0, -7.0, 3.0)
        glLightfv(GL_LIGHT0, GL_POSITION, (0.0, 0.0, 0.0, 1.0))
        glPopMatrix()
        
        glPushMatrix()
        glTranslate(-4.0, 10.0, 0.0)
        glLightfv(GL_LIGHT1, GL_POSITION, (0.0, 0.0, 0.0, 1.0))
        glPopMatrix()

        self.draw_system()


class SystemTrajectoryViewer(SystemViewer):
    def __init__(self, system, times, traj, inputs=None,
                 display_dt=1.0/25.0,
                 camera=None,
                 window_width=600,
                 window_height=600):
        OpenGLViewer.__init__(self,
                              display_dt=display_dt,
                              camera=camera,
                              window_width=window_width,
                              window_height=window_height)
        SystemPainter.__init__(self, system)

        self.times = times
        self.traj = traj
        self.inputs = inputs
        self.index = 0
        self.time = 0.0
        self.playing = False
        self.ended = False
        self.filming = False
        self.set_key_handler('r', self.reset)
        self.set_key_handler(',', self.prev_frame)
        self.set_key_handler('.', self.next_frame)
        self.set_key_handler(' ', self.play_pause)
        self.set_key_handler('o', self.save_frames)

    def print_instructions(self):
        print \
"""
System Trajectory Viewer Instructions

Camera Movement:
 W : move foward
 A : move left
 S : move backward
 D : move right
 Q : move up
 E : move down

 To rotate the camera, use the arrow keys or hold down the right mouse
 button and move the mouse.

Other Commands:
 escape : quit
 space : play/pause animation
 , : go back one frame
 . : go forward one frame
 r : reset animation to beginning
 o : play animation and write frames to disk
 p : save current frame to disk
"""

    def reset(self):
        self.time = 0.0
        self.index = 0
        self.update_title()

    def prev_frame(self):
        if self.filming:
            return
        self.playing = False
        self.ended = False
        i = len(self.times)-1
        while i > 0 and self.times[i] >= self.time:
            i -= 1

        self.index = i
        self.time = self.times[i]        
        self.update_title()
        
    def next_frame(self):
        if self.filming:
            return 
        self.playing = False
        for i,t in enumerate(self.times):
            if t > self.time:
                self.index = i
                self.time = t
                break
        self.update_title()
       
    def play_pause(self):
        if self.filming:
            return
        if self.ended:
            self.time = 0.0
            self.index = 0
            self.playing = True
            self.ended = False
        else:
            self.playing = not self.playing

    def save_frames(self):
        if self.filming:
            self.filming = False
            self.playing = False
        else:
            self.filming = True
            self.playing = True
            self.filming_index = 0

    def display_timer(self):
        if self.playing:
            self.time += self.display_dt
            self.update_title()


    def update_title(self):
        glutSetWindowTitle("System BasicViewer - Frame (%d/%d) - Time %f (s)" % (
                self.index, len(self.times), self.times[self.index]))


    def display(self):
        if self.times[self.index] < self.time:
            while self.index < len(self.times) and self.times[self.index] < self.time:
                self.index += 1
            if self.index == len(self.times):
                self.ended = True
                self.filming = False
            self.index -= 1
        else:
            while self.index > 0 and self.times[self.index] > self.time:
                self.index -= 1
        
        self.system.q = self.traj[self.index]
        if self.inputs is not None and self.index < len(self.inputs): 
            self.system.u = self.inputs[self.index]
        SystemViewer.display(self)

        if self.filming:
            self.screenshot("frame-%05d.png" % self.filming_index)
            self.filming_index += 1
