import trep
import trep.sexp
import trep.visual
import trep.constraints
from trep.visual import SystemViewer
from trep.visual import stlmodel
import math
from math import pi as mpi
from math import sin, cos
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *


def xfrange(start, stop=None, step=None):
    """Like range(), but returns list of floats instead
    
    All numbers are generated on-demand using generators
    """
    if stop is None:
        stop = float(start)
        start = 0.0

    if step is None:
        step = 1.0

    cur = float(start)

    while cur <= stop:
        yield cur
        cur += step         


PICK_GLOBAL = 0
PICK_STRING_PLATFORM = 1
PICK_RIGHT_ARM_STRING = 2
PICK_LEFT_ARM_STRING = 3
PICK_RIGHT_LEG_STRING = 4
PICK_LEFT_LEG_STRING = 5

class StringControl:
    def __init__(self, system, x_config, y_config, l_config=None):
        self.x_config = system.get_config(x_config)
        self.y_config = system.get_config(y_config)
        self.l_config = system.get_config(l_config)
        self.initial_position = None

    def store_initial_position(self):
        self.initial_position = [self.x_config.q,
                                 self.y_config.q]

        
   
class InteractiveViewer(SystemViewer):
    def __init__(self, dt=0.02, alpha=0.5, string_plane_z=10, **opts):

        self.dt = dt
        self.alpha = alpha

        # String Plane Grid values
        self.grid_list = None

        exp = trep.sexp.read_sexp(open("puppet.lsp").read())
        self.system = trep.sexp.read_system(exp)

        # Change the string plane height
        self.string_plane_z = string_plane_z
        self.system.get_frame('String Plane').value = self.string_plane_z

        SystemViewer.__init__(self, self.system, **opts)
        self.auto_draw = False

        # Setup puppet stl model bindings
        self.add_display_func('Torso',
                              stlmodel('./stl/torso.stl').draw)
        self.add_display_func('Left Shoulder',
                              stlmodel('./stl/lefthumerus.stl').draw)
        self.add_display_func('Right Shoulder',
                              stlmodel('./stl/righthumerus.stl').draw)
        self.add_display_func('Left Elbow',
                              stlmodel('./stl/leftradius.stl').draw)
        self.add_display_func('Right Elbow',
                              stlmodel('./stl/rightradius.stl').draw)
        self.add_display_func('Right Hip',
                              stlmodel('./stl/femur.stl').draw)
        self.add_display_func('Left Hip',
                              stlmodel('./stl/femur.stl').draw)
        self.add_display_func('Right Knee',
                              stlmodel('./stl/tibia.stl').draw)
        self.add_display_func('Left Knee',
                              stlmodel('./stl/tibia.stl').draw)
        self.add_display_func('Head',
                              stlmodel('./stl/head.stl').draw)
        self.add_display_func('String Platform',
                              lambda : self.draw_pickable_sphere(PICK_STRING_PLATFORM, 0.5))
        self.add_display_func('Left Arm Spindle',
                              lambda : self.draw_pickable_sphere(PICK_LEFT_ARM_STRING, 0.25))
        self.add_display_func('Right Arm Spindle',
                              lambda : self.draw_pickable_sphere(PICK_RIGHT_ARM_STRING, 0.25))
        self.add_display_func('Left Leg Spindle',
                              lambda : self.draw_pickable_sphere(PICK_LEFT_LEG_STRING, 0.25))
        self.add_display_func('Right Leg Spindle',
                              lambda : self.draw_pickable_sphere(PICK_RIGHT_LEG_STRING, 0.25))

        self.simulation_id = 0

        # Simulation-related values.  These are initialized by
        # reset_system()
        self.gmvi = None 
        self.reset_system()

        # String control values
        self.string_controls = {
            PICK_STRING_PLATFORM: StringControl(self.system,
                                                "StringPlatformX",
                                                "StringPlatformY"),
            PICK_RIGHT_ARM_STRING: StringControl(self.system,
                                                 "RArmStringX",
                                                 "RArmStringY",
                                                 "RArmStringL"),
            PICK_LEFT_ARM_STRING: StringControl(self.system,
                                                "LArmStringX",
                                                "LArmStringY",
                                                "LArmStringL"),
            PICK_RIGHT_LEG_STRING: StringControl(self.system,
                                                 "RLegStringX",
                                                 "RLegStringY",
                                                 "RLegStringL"),
            PICK_LEFT_LEG_STRING: StringControl(self.system,
                                                "LLegStringX",
                                                "LLegStringY",
                                                "LLegStringL")
            }
        self.selected_strings = []
        self.selection_start = None

        # Mouse navigation values
        self.navigation_start = None

        self.set_mouse_button_handler(GLUT_LEFT_BUTTON,
                                      self.left_mouse_button_down)
        self.set_mouse_button_handler(3,
                                      lambda x,y : self.mouse_wheel_up)
        self.set_mouse_button_handler(4,
                                      lambda x,y : self.mouse_wheel_down)
                                      

    def initialize_opengl(self):
        """
        Setup OpenGL
        """
        glClearColor (0.0, 0.0, 0.0, 0.0)
        glShadeModel (GL_SMOOTH)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_NORMALIZE)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)

        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)
        glLineWidth(1.5)
        
        glEnable(GL_FOG)
        glFogi (GL_FOG_MODE, GL_LINEAR)
        glFogfv (GL_FOG_COLOR, [0.0, 0.0, 0.0, 1.0])
        glFogf (GL_FOG_DENSITY, 0.95)
        glHint (GL_FOG_HINT, GL_DONT_CARE)
        glFogf (GL_FOG_START, 5.0)
        glFogf (GL_FOG_END, 10.0)
        
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0,1.0,1.0,1.0))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (1.0,1.0,1.0,1.0))


    def reset_system(self):
        for config in self.system.configs:
            config.q = 0
            config.dq = 0
            
        self.system.get_config('TorsoX').q = 1.0
        self.system.get_config('TorsoY').q = 1.0
        self.system.get_config('LElbowTheta').q = -mpi/2.0
        self.system.get_config('RElbowTheta').q = -mpi/2.0
        self.system.get_config('LHipTheta').q = -mpi/10.0
        self.system.get_config('RHipTheta').q = mpi/10.0
        self.system.get_config('LHipPhi').q = -mpi/4.0
        self.system.get_config('RHipPhi').q = -mpi/4.0
        self.system.get_config('LKneeTheta').q = mpi/4.0
        self.system.get_config('RKneeTheta').q = mpi/4.0
        self.system.get_config('LArmStringX').q = 1.0
        self.system.get_config('LArmStringY').q = -1.0
        self.system.get_config('RArmStringX').q = -1.0
        self.system.get_config('RArmStringY').q = -1.0
        self.system.get_config('LLegStringX').q = 1.0
        self.system.get_config('LLegStringY').q = -2.0
        self.system.get_config('RLegStringX').q = -1.0
        self.system.get_config('RLegStringY').q = -2.0
        for c in self.system.constraints:
            c.set_length(c.get_actual_length())
        self.system.satisfy_constraints()
        self.gmvi = trep.GenMidpoint(self.system, alpha=0.5, dt=self.dt)

    def advance_simulation(self):
        qk = [config.q for config in self.system.kin_configs]
        self.gmvi.step(self.dt, tuple(qk))

        ## try:
        ##     simulate()
        ## except StandardError, e:
        ##     print e
        ##     self.simulation_id += 1


    def mouse_active_motion(self, x, y):
        """
        GLUT mouse active motion callback function
        """
        SystemViewer.mouse_active_motion(self, x, y)
        if self.mouse_button_state(GLUT_LEFT_BUTTON):
            self.left_mouse_button_motion(x, y)

    def pre_mainloop(self):
        glutTimerFunc(int(self.dt*1000.0), self.simulation_timer, self.simulation_id)

    def draw_pickable_sphere(self, id, radius):
        glPushAttrib(GL_CURRENT_BIT)
        glPushName(id)
        if self.string_controls[id] in self.selected_strings:
            glColor3f(1.0, 0.0, 0.0)
        else:
            glColor3f(0.5, 0.5, 0.5)
        glutSolidSphere(radius, 10, 10)
        glPopName(id)
        glPopAttrib()
        
    def draw_string_plane(self):
        if self.grid_list:
            glCallList(self.grid_list)
        else:
            self.grid_list = glGenLists(1)
            
            if self.grid_list:
                glNewList(self.grid_list, GL_COMPILE)

            x_range = (-50,50)
            y_range = (-50,50)
            spacing = 0.5

            glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT )
            glDisable(GL_LIGHTING)
            glColor3f(0.27, 0.31, 0.45)

            for x in xfrange(x_range[0], x_range[1], spacing):
                glBegin(GL_LINES)
                glVertex3f(x, y_range[0], self.string_plane_z)
                glVertex3f(x, y_range[1], self.string_plane_z)
                glEnd()

            for y in xfrange(y_range[0], y_range[1], spacing):
                glBegin(GL_LINES)
                glVertex3f(x_range[0], y, self.string_plane_z)
                glVertex3f(x_range[1], y, self.string_plane_z)
                glEnd()

            glPopAttrib()

            if self.grid_list:
                glEndList()
                glCallList(self.grid_list)


    def display(self):
        SystemViewer.display(self)
        for x in self.system.constraints:
            x.opengl_draw()
        self.draw_string_plane()


    def display_timer(self):
        if self.visible:
            glutSetWindowTitle("Interactive Puppet Demo - Time %f (s)" % self.gmvi.t1)


    def simulation_timer(self, id):
        """
        The simulation timer drives the simulation.
        """
        if id != self.simulation_id:
            return
        glutTimerFunc(int(self.dt*1000.0), self.simulation_timer, self.simulation_id)
        self.advance_simulation()        


    def screen_to_world_plane(self, sx, sy, A, B, C, D):
        """
        Project a screen coordinate onto a plane in the world
        """
        (x1, y1, z1) = gluUnProject(sx, self.window_height - sy, 0.0)
        (x2, y2, z2) = gluUnProject(sx, self.window_height - sy, 1.0)

        # Intersect line x1 + (x2 - x1)*t with plane A*x + B*y + C*z + D = 0
        denom = A*(x2-x1) + B*(y2-y1) + C*(z2-z1)
        if denom == 0.0:
            return None       
        t = -(D + A*x1 + B*y1 + C*z1)/denom
        return (x1 + (x2-x1)*t,
                y1 + (y2-y1)*t,
                z1 + (z2-z1)*t)


    def pick_scene(self, m_x, m_y):
        """
        Perform a picking operation at the specified screen
        coordinates.  Returns the list of hit records
        """
        viewport = glGetIntegerv(GL_VIEWPORT)
        glSelectBuffer(512)
        glRenderMode(GL_SELECT)
        
        glInitNames()

        # Setup the picking projection matrix
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluPickMatrix(m_x, viewport[3]-m_y, 5.0, 5.0, viewport)
        self.camera.setup_projection_matrix()
        glMatrixMode(GL_MODELVIEW)

        self.camera.load_matrix()
        self.display()

        # Restore projection matrix
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glFlush()

        # Retreive hits
        return glRenderMode(GL_RENDER)
        

    def left_mouse_button_down(self, m_x, m_y):
        """
        Called when the left mouse button is pressed down
        """
        hits = self.pick_scene(m_x, m_y)
        for hit in hits:
            if hit.names == []:
                continue
            if hit.names[0] in self.string_controls:
                string = self.string_controls[hit.names[0]]
                if glutGetModifiers() &  GLUT_ACTIVE_SHIFT:
                    if string in self.selected_strings:
                        self.selected_strings.remove(string)
                    else:
                        self.selected_strings.append(string)
                else:
                    self.selected_strings = [string]

        for string in self.selected_strings:
            string.store_initial_position()

        if self.selected_strings != []:
            self.selection_start = self.screen_to_world_plane(m_x, m_y,
                                                              0, 0, 1, -self.string_plane_z)
                
    def left_mouse_button_motion(self, m_x, m_y):
        """
        Called when the mouse moves while the left mouse button is down
        """
        if self.selected_strings == []:
            return

        pos = self.screen_to_world_plane(m_x, m_y,
                                         0, 0, 1, -self.string_plane_z)
    
        delta_x = pos[0] - self.selection_start[0]
        delta_y = pos[1] - self.selection_start[1]

        for string in self.selected_strings:
            if string.x_config != None:
                string.x_config.q = string.initial_position[0] + delta_x
            if string.y_config != None:
                string.y_config.q = string.initial_position[1] + delta_y


    def mouse_wheel_up(self):
        """
        Called when the mouse wheel is scrolled up
        """
        print "up!"
        if self.inputs['LEFT_MOUSE_BUTTON']:
            for string in self.selected_strings:
                if string.l_config != None:
                    string.l_config.q -= 0.4

    
    def mouse_wheel_down(self):
        """
        Called when the mouse wheel is scrolled down
        """
        if self.inputs['LEFT_MOUSE_BUTTON']:
            for string in self.selected_strings:
                if string.l_config != None:
                    string.l_config.q += 0.4

                
if __name__ == "__main__":
    viewer = InteractiveViewer()
    viewer.run()


