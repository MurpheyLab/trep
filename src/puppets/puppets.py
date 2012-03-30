import trep
from trep import rx, ry, rz, tx, ty, tz, const_txyz

from OpenGL.GL import *
from OpenGL.GLU import *
from trep.visual import *


# Default dimensions, originated from Pygmalion motion capture data
DEFAULT_DIMENSIONS = {
    'ltibia_length' : 0.4000000081268073,
    #ltibia_length - Distance between the left knee axis and left
    #ankle axis
    'upper_torso_length' : 0.49134332092915983,
    # upper_torso_length - Distance between the center of the hips and
    # center of the shoulders
    'lshoulder_width' : 0.15811387592769632,
    # lshoulder_width - Distance from the center of the shoulders and
    # the left shoulder axis
    'final_head_length' : 0.079999981957211988,
    # final_head_length - Distance between the head's center of mass
    # and the top
    'lradius_length' : 0.25765900905806788,
    # lradius_length - Distance between the left elbow axis and left
    # wrist axis
    'rshoulder_width' : 0.15811388178161004,
    # rshoulder_width - Distance from the center of the shoulders and
    # the right shoulder axis
    'rfoot_length' : 0.20000000361419676,
    # rfoot_length - Length of the right foot
    'rradius_length' : 0.25765900577819512,
    # rradius_length - Distance between the right elbow axis and the
    # right wrist axis
    'lfoot_length' : 0.20298699715253277,
    # lfoot_length - Length of the left foot
    'rtibia_length' : 0.40000001048670952,
    # rtibia_length - Distance between the right knee axis and right
    # ankle axis
    'rhip_width' : 0.10000000072551045,
    # rhip_width - Distance between the center of the hips and the
    # center of the right hip joint
    'neck_length' : 0.04500002226970809,
    # neck_length - Distance between the center of the shoulders and
    # the center of the neck joint
    'lhip_width' : 0.10000000633343277,
    # lhip_width - Distance between the center of the hips and the
    # center of the left hip joint
    'lhumerus_length' : 0.24999999610023391,
    # lhumerus_length - Distance between the center of the left
    # shoulder joint and the elbow axis
    'lower_head_length' : 0.044999991144167348,
    # lower_head_length - Distance between the center of the neck
    # joint and the base of the head
    'rfemur_length' : 0.4000000149386187,
    # rfemur_length - Distance between the center of the right hip
    # joint and the right knee axis
    'upper_head_length' : 0.090000011865579349,
    # upper_head_length - Distance from the base of the head to the
    # center of mass
    'lhand_length' : 0.090000006491074078,
    # lhand_length - Length of the left hand
    'rhumerus_length' : 0.2500000129383001,
    # rhumerus_length - Distance between the center of the right
    # shoulder joint and the elbow axis
    'rhand_length' : 0.089999998759819969,
    # rhand_length - Length of the right hand
    'lfemur_length' : 0.4000000143639284,
    # lfemur_length - Distance between the center of the left hip
    # joint and the left knee axis.

    # Mass properties of the puppet.  Each list is the mass and
    # rotational intertias: [M, Ixx, Iyy, Izz].
    'pelvis_mass' : [10, 1, 1, 1],
    'head_mass' : [0.75, 0.08, 0.08, 0.08],
    'femur_mass' : [1.5, 0.3, 0.3, 0.1],
    'tibia_mass' : [1.5, 0.3, 0.3, 0.1],
    'humerus_mass' : [0.5, 0.05, 0.05, 0.01],
    'radius_mass' : [0.5, 0.05, 0.05, 0.01],

    # Damping constant for the dynamic configuration variables.
    'damping' : 0.2,

    # Height of the string plane along the Z-axis.
    'string_plane_height' : 2,

    # String definitions.  Each entry maps the name of a string to a
    # tuple describing how it attaches to the puppet.  The first entry
    # is the name of a frame in the system.  The second entry is a
    # tuple of the offset (x,y,z translations) of the attachment point
    # to the named frame.
    'strings' : {
        #'upper_torso_string' : ('spine_top', (0, -0.1, 0)),
        #'lower_torso_string' : ('pelvis', (0, -0.1, 0)),
        'upper_torso_string' : ('spine_top', (-0.1, 0, 0)),
        'lower_torso_string' : ('spine_top', (0.1, 0, 0)),
        'left_arm_string' : ('lradius_end', (0, 0.1, 0)),
        'right_arm_string' : ('rradius_end', (0, 0.1, 0)),
        'left_leg_string' : ('lfemur_end', (0, 0.1, 0)),
        'right_leg_string' : ('rfemur_end', (0, 0.1, 0))
        }
    }

# Default joints, a string is the name of the configuration variable
# controlling the joint parameter, a constant is a fixed parameter.
DEFAULT_JOINTS = {
    'torso_tx' : 'torso_tx',
    'torso_ty' : 'torso_ty',
    'torso_tz' : 'torso_tz',
    'torso_rz' : 'torso_rz',
    'torso_ry' : 'torso_ry',
    'torso_rx' : 'torso_rx',
    'lhip_rz' : 'lhip_rz',
    'lhip_ry' : 'lhip_ry',
    'lhip_rx' : 'lhip_rx',
    'lknee_rx' : 'lknee_rx',
    'lfoot_rx' : 0.0,
    'lfoot_ry' : 0.0,
    'lfoot_rz' : 0.0,
    'rhip_rz' : 'rhip_rz',
    'rhip_ry' : 'rhip_ry',
    'rhip_rx' : 'rhip_rx',
    'rknee_rx' : 'rknee_rx',
    'rfoot_rx' : 0.0,
    'rfoot_ry' : 0.0,
    'rfoot_rz' : 0.0,
    'neck_rz' : 0.0,
    'neck_ry' : 0.0,
    'neck_rx' : 0.0,
    'lshoulder_rz' : 'lshoulder_rz',
    'lshoulder_ry' : 'lshoulder_ry',
    'lshoulder_rx' : 'lshoulder_rx',
    'lelbow_rx' : 'lelbow_rx',
    'lhand_rx' : 0.0,
    'lhand_ry' : 0.0,
    'lhand_rz' : 0.0,
    'rshoulder_rz' : 'rshoulder_rz',
    'rshoulder_ry' : 'rshoulder_ry',
    'rshoulder_rx' : 'rshoulder_rx',
    'relbow_rx' : 'relbow_rx',
    'rhand_rx' : 0.0,
    'rhand_ry' : 0.0,
    'rhand_rz' : 0.0
    }    

def fill_dimensions(dimensions={}):
    dim = dimensions.copy()
    for (key, default) in DEFAULT_DIMENSIONS.iteritems():
        if key not in dim:
            dim[key] = default
    return dim

def fill_joints(joints={}):
    joints = joints.copy()
    for (key, default) in DEFAULT_JOINTS.iteritems():
        if key not in joints:
            joints[key] = default
    return joints

def make_skeleton(dimensions={}, joints={}):
    dim = fill_dimensions(dimensions)
    joints = fill_joints(joints)

    frames = [
        tx(joints['torso_tx']), [ty(joints['torso_ty']), [tz(joints['torso_tz']), [
            rz(joints['torso_rz']), [ ry(joints['torso_ry']), [
                rx(joints['torso_rx'], name='pelvis',  mass=dim['pelvis_mass']), [
                    tx(-dim['lhip_width'], name='lhip'), [
                        rz(joints['lhip_rz']), [ry(joints['lhip_ry']), [rx(joints['lhip_rx'], name='lfemur'), [
                            tz(-dim['lfemur_length']/2, name='lfemur_mass', mass=dim['femur_mass']),
                            tz(-dim['lfemur_length'], name='lfemur_end'), [
                                rx(joints['lknee_rx'], name='ltibia'), [
                                    tz(-dim['ltibia_length']/2, name='ltibia_mass', mass=dim['tibia_mass']),
                                    tz(-dim['ltibia_length'], name='ltibia_end'), [
                                        rz(joints['lfoot_rz']), [ry(joints['lfoot_ry']), [
                                            rx(joints['lfoot_rx'], name='lfoot'), [
                                                ty(dim['lfoot_length'], name='lfoot_end')]]]]]]]]]],
                    tx(dim['rhip_width'], name='rhip'), [
                        rz(joints['rhip_rz']), [ry(joints['rhip_ry']), [rx(joints['rhip_rx'], name='rfemur'), [
                            tz(-dim['rfemur_length']/2, name='rfemur_mass', mass=dim['femur_mass']), 
                                tz(-dim['rfemur_length'], name='rfemur_end'), [
                                    rx(joints['rknee_rx'], name='rtibia'), [
                                        tz(-dim['rtibia_length']/2, name='rtibia_mass', mass=dim['tibia_mass']), 
                                    tz(-dim['rtibia_length'], name='rtibia_end'), [
                                        rz(joints['rfoot_rz']), [ry(joints['rfoot_ry']), [
                                            rx(joints['rfoot_rx'], name='rfoot'), [
                                                ty(dim['rfoot_length'], name='rfoot_end')]]]]]]]]]],
                    tz(dim['upper_torso_length'], name='spine_top'), [
                        tz(dim['neck_length'], name='neck'), [
                            rz(joints['neck_rz']), [ry(joints['neck_ry']), [
                                rx(joints['neck_rz'], name='neck_joint'), [
                                    tz(dim['lower_head_length'], name='head'), [
                                        tz(dim['upper_head_length'], name='head_center', mass=dim['head_mass']), [
                                            tz(dim['final_head_length'], name='head_end')]]]]]],
                        tx(-dim['lshoulder_width'], name='lshoulder'), [
                            rz(joints['lshoulder_rz']), [ry(joints['lshoulder_ry']), [
                                rx(joints['lshoulder_rx'], name='lhumerus'), [
                                    tz(-dim['lhumerus_length']/2, name='lhumerus_mass', mass=dim['humerus_mass']), 
                                    tz(-dim['lhumerus_length'], name='lhumerus_end'), [
                                        rx(joints['lelbow_rx'], name='lradius'), [
                                            tz(-dim['lradius_length']/2, name='lradius_mass', mass=dim['radius_mass']), 
                                            tz(-dim['lradius_length'], name='lradius_end'), [
                                                rz(joints['lhand_rz']), [ry(joints['lhand_ry']), [
                                                    rx(joints['lhand_rx'], name='lhand'), [
                                                        tz(-dim['lhand_length'], name='lhand_end')]]]]]]]]]],
                        tx(dim['rshoulder_width'], name='rshoulder'), [
                            rz(joints['rshoulder_rz']), [ry(joints['rshoulder_ry']), [
                                rx(joints['rshoulder_rx'], name='rhumerus'), [
                                    tz(-dim['rhumerus_length']/2, name='rhumerus_mass', mass=dim['humerus_mass']), 
                                    tz(-dim['rhumerus_length'], name='rhumerus_end'), [
                                        rx(joints['relbow_rx'], name='rradius'), [
                                            tz(-dim['rradius_length']/2, name='rradius_mass', mass=dim['radius_mass']), 
                                            tz(-dim['rradius_length'], name='rradius_end'), [
                                                rz(joints['rhand_rz']), [ry(joints['rhand_ry']), [
                                                    rx(joints['rhand_rx'], name='rhand'), [
                                                        tz(-dim['rhand_length'], name='rhand_end')]]]]]]]]]]
                    ]]]]]]]]
    return frames


class Puppet(trep.System):
    def __init__(self, dimensions={}, joints={}, 
                joint_forces=False,
                string_forces=False,
                string_constraints=False):

        trep.System.__init__(self)

        self.string_plane = None
        self.string_hooks = {}
        self.joint_forces = {}
        self.string_forces = {}
        self.string_constraints = {}

        self.dimensions = fill_dimensions(dimensions)
        self.joints = fill_joints(joints)

        # Create the skeleton frames
        self.import_frames(make_skeleton(self.dimensions, self.joints))

        # Add puppet string frames
        self.make_string_frames()    
        
        # Add desired forces/constraints
        if joint_forces:
            self.make_joint_forces()
        if string_forces:
            self.make_string_forces()
        if string_constraints:
            self.make_string_constraints()
                    
        # Add remaining forces/potentials
        trep.potentials.Gravity(self, (0, 0, -9.8))
        trep.forces.Damping(self, self.dimensions['damping'])

        
    def make_string_frames(self):
        # Add the string plane
        self.world_frame.import_frames([
            tz(self.dimensions['string_plane_height'], name='string_plane')])
        self.string_plane = self.get_frame('string_plane')

        # Add the string hook frames
        self.string_hooks = {}    
        for name, hook_location in self.dimensions['strings'].iteritems():
            self.string_hooks[name] = name + '_hook'
            self.get_frame(hook_location[0]).import_frames([
                const_txyz(hook_location[1], name=self.string_hooks[name])])

        
    def make_joint_forces(self):
        for config in self.dyn_configs:
            self.joint_forces[config.name] = \
                trep.forces.ConfigForce(self, config, config.name, config.name)
            

    def make_string_forces(self):
        for name, hook_point in self.string_hooks.iteritems():
            force = {
                'name' : name,
                'x' : name + '-x',
                'y' : name + '-y',
                'z' : name + '-z',
                'hook' : hook_point,
                }
            trep.forces.HybridWrench(self, force['hook'],
                                     (force['x'], force['y'], force['z'], 0, 0, 0),
                                     name=name)
            self.string_forces[name] = force


    def make_string_constraints(self):
        for name, hook_point in self.string_hooks.iteritems():
            info = {
                'name' : name,
                'x' : name + '-x',            # Name of X kinematic config variable
                'y' : name + '-y',            # Name of Y kinematic config variable
                'length' : name + '-length',  # Name of length kinematic config variable
                # Name of the frames connected by the strings
                'control_hook' : name + '_control',
                'hook' : hook_point
                }
            # Add frames from the control_hook
            self.string_plane.import_frames([
                tx(info['x'], kinematic=True), [
                    ty(info['y'], kinematic=True, name=info['control_hook'])
                    ]])
            trep.constraints.Distance(self, info['hook'],
                                      info['control_hook'], info['length'],
                                      name=name)
            self.string_constraints[name] = info


    def project_string_controls(self, strings=None):
        """
        Sets the location of each string control point to be directly
        above the corresponding puppet hook in the current
        configuration.  The string lengths are corrected as well.

        By default, it affects all the strings in the puppet.  Specify
        a list of string names to only project some strings.
        """
        if strings is None:
            infos = self.string_constraints.values()
        else:
            infos = [self.string_constraints[name] for name in strings]
        
        for info in infos:
            x_var = self.get_config(info['x'])
            y_var = self.get_config(info['y'])
            pos = self.get_frame(info['hook']).p()
            x_var.q = pos[0]
            y_var.q = pos[1]

        self.correct_string_lengths(strings)


    def correct_string_lengths(self, strings=None):
        """
        Sets the length of each string to the correct length.

        By default, it affects all the strings in the puppet.  Specify
        a list of string names to only project some strings.
        """
        if strings is None:
            infos = self.string_constraints.values()
        else:
            infos = [self.string_constraints[name] for name in strings]
        
        for info in infos:
            length_var = self.get_config(info['length'])
            length_var.q = self.get_constraint(info['name']).get_actual_distance()


class PuppetVisual(VisualItem3D):
    def __init__(self, *args, **kwds):
        super(PuppetVisual, self).__init__(*args, **kwds)

        self.setOrientation(forward=[0,-1,0], up=[0,0,1])

        self.color=(0.2, 0.2, 0.2, 1.0)
        self.puppet = self._system
        self.attachDrawing('pelvis', self.draw_torso)
        self.attachDrawing('lhumerus',  self.draw_left_humerus)
        self.attachDrawing('lradius',   self.draw_left_radius)
        self.attachDrawing('lhand',     self.draw_left_hand)
        self.attachDrawing('rhumerus', self.draw_right_humerus)
        self.attachDrawing('rradius',  self.draw_right_radius)
        self.attachDrawing('rhand',    self.draw_right_hand)
        self.attachDrawing('lfemur',    self.draw_left_femur)
        self.attachDrawing('ltibia',    self.draw_left_tibia)
        self.attachDrawing('lfoot',     self.draw_left_foot)
        self.attachDrawing('rfemur',   self.draw_right_femur)
        self.attachDrawing('rtibia',   self.draw_right_tibia)
        self.attachDrawing('rfoot',    self.draw_right_foot)
        self.attachDrawing('head_center',   self.draw_head)
        self.attachDrawing('neck_joint', self.draw_neck_joint)
        self.attachDrawing(None, self.draw_globals)
        
        self.quad = gluNewQuadric()
        gluQuadricNormals(self.quad, GLU_SMOOTH)
        
        
    def draw_globals(self):
        glPushAttrib(GL_CURRENT_BIT)            
        glColor4f(*self.color)

        glPushMatrix()
        #glScale(self.scale, self.scale, self.scale)
        
        for part in (self.puppet.constraints +
                     self.puppet.potentials +
                     self.puppet.forces):
            part.opengl_draw()

        glPopMatrix()
        glPopAttrib()

    def draw_neck_joint(self):
        radius = self.puppet.dimensions['neck_length']/2
        gluSphere(self.quad, radius, 10, 10)
        
    def draw_head(self):
        radius = self.puppet.dimensions['final_head_length']

        glPushMatrix()
        glScalef(0.5, 0.8, 1.0)
        gluSphere(self.quad, radius, 10, 10)
        glPopMatrix()
        
    def draw_torso(self):
        radius1 = (self.puppet.dimensions['lfemur_length'] +
                  self.puppet.dimensions['rfemur_length'])/16*0.9

        radius2 = (self.puppet.dimensions['lhumerus_length'] +
                  self.puppet.dimensions['rhumerus_length'])/16*0.9
        
        glPushMatrix()
        glRotatef(90, 0, 1, 0)
        glTranslatef(0, 0, -self.puppet.dimensions['lhip_width'])
        gluCylinder(self.quad, radius1, radius1,
                    self.puppet.dimensions['lhip_width'] +
                    self.puppet.dimensions['rhip_width'], 10, 1)
        glPopMatrix()

        gluCylinder(self.quad, radius1, radius2,
                    self.puppet.dimensions['upper_torso_length'], 10, 1)

        glTranslatef(0, 0, self.puppet.dimensions['upper_torso_length'])
            
        glPushMatrix()
        glRotatef(90, 0, 1, 0)
        glTranslatef(0, 0, -self.puppet.dimensions['lshoulder_width'])
        gluCylinder(self.quad, radius2, radius2,
                    self.puppet.dimensions['lshoulder_width'] +
                    self.puppet.dimensions['rshoulder_width'], 10, 1)
        glPopMatrix()

    def draw_left_humerus(self):
        radius = self.puppet.dimensions['lhumerus_length']/8
        
        gluSphere(self.quad, radius, 10, 10)
        glRotatef(180, 1, 0, 0)
        gluCylinder(self.quad, radius*0.9, radius*0.6,
                    self.puppet.dimensions['lhumerus_length'], 10, 1)
        
    def draw_left_radius(self):
        radius = self.puppet.dimensions['lhumerus_length']/8*0.6

        glPushMatrix()
        glRotatef(90, 0, 1, 0)
        glTranslatef(0, 0, -radius)
        gluCylinder(self.quad, radius, radius, 2*radius, 10, 1)
        gluDisk(self.quad, 0, radius, 10, 1)
        glTranslatef(0, 0, 2*radius)
        gluDisk(self.quad, 0, radius, 10, 1)
        glPopMatrix()
        glRotatef(180, 1, 0, 0)
        gluCylinder(self.quad, radius*0.9, radius*0.6,
                    self.puppet.dimensions['lradius_length'], 10, 1)
        
    def draw_left_hand(self):
        radius = self.puppet.dimensions['lhumerus_length']/8*0.6*0.6

        glPushMatrix()
        glRotatef(90, 0, 1, 0)
        glTranslatef(0, 0, -radius)
        gluCylinder(self.quad, radius, radius, 2*radius, 10, 1)
        gluDisk(self.quad, 0, radius, 10, 1)
        glTranslatef(0, 0, 2*radius)
        gluDisk(self.quad, 0, radius, 10, 1)
        glPopMatrix()
        glRotatef(180, 1, 0, 0)
        gluCylinder(self.quad, radius*0.9, radius*0.6,
                    self.puppet.dimensions['lhand_length'], 10, 1)
        glTranslatef(0,0,self.puppet.dimensions['lhand_length'])
        gluDisk(self.quad, 0, radius*0.6, 10, 1)
        
    def draw_right_humerus(self):
        radius = self.puppet.dimensions['rhumerus_length']/8
        
        gluSphere(self.quad, radius, 10, 10)
        glRotatef(180, 1, 0, 0)
        gluCylinder(self.quad, radius*0.9, radius*0.6,
                    self.puppet.dimensions['rhumerus_length'], 10, 1)
        
    def draw_right_radius(self):
        radius = self.puppet.dimensions['rhumerus_length']/8*0.6

        glPushMatrix()
        glRotatef(90, 0, 1, 0)
        glTranslatef(0, 0, -radius)
        gluCylinder(self.quad, radius, radius, 2*radius, 10, 1)
        gluDisk(self.quad, 0, radius, 10, 1)
        glTranslatef(0, 0, 2*radius)
        gluDisk(self.quad, 0, radius, 10, 1)
        glPopMatrix()
        glRotatef(180, 1, 0, 0)
        gluCylinder(self.quad, radius*0.9, radius*0.6,
                    self.puppet.dimensions['rradius_length'], 10, 1)
        
    def draw_right_hand(self):
        radius = self.puppet.dimensions['rhumerus_length']/8*0.6*0.6

        glPushMatrix()
        glRotatef(90, 0, 1, 0)
        glTranslatef(0, 0, -radius)
        gluCylinder(self.quad, radius, radius, 2*radius, 10, 1)
        gluDisk(self.quad, 0, radius, 10, 1)
        glTranslatef(0, 0, 2*radius)
        gluDisk(self.quad, 0, radius, 10, 1)
        glPopMatrix()
        glRotatef(180, 1, 0, 0)
        gluCylinder(self.quad, radius*0.9, radius*0.6,
                    self.puppet.dimensions['rhand_length'], 10, 1)
        glTranslatef(0,0,self.puppet.dimensions['rhand_length'])
        gluDisk(self.quad, 0, radius*0.6, 10, 1)

    def draw_left_femur(self):
        radius = self.puppet.dimensions['lfemur_length']/8
        
        gluSphere(self.quad, radius, 10, 10)
        glRotatef(180, 1, 0, 0)
        gluCylinder(self.quad, radius*0.9, radius*0.6,
                    self.puppet.dimensions['lfemur_length'], 10, 1)
        
    def draw_left_tibia(self):
        radius = self.puppet.dimensions['lfemur_length']/8*0.6

        glPushMatrix()
        glRotatef(90, 0, 1, 0)
        glTranslatef(0, 0, -radius)
        gluCylinder(self.quad, radius, radius, 2*radius, 10, 1)
        gluDisk(self.quad, 0, radius, 10, 1)
        glTranslatef(0, 0, 2*radius)
        gluDisk(self.quad, 0, radius, 10, 1)
        glPopMatrix()
        glRotatef(180, 1, 0, 0)
        gluCylinder(self.quad, radius*0.9, radius*0.6,
                    self.puppet.dimensions['ltibia_length'], 10, 1)
        
    def draw_left_foot(self):
        radius = self.puppet.dimensions['lfemur_length']/8*0.6*0.6

        glPushMatrix()
        glRotatef(90, 0, 1, 0)
        glTranslatef(0, 0, -radius)
        gluCylinder(self.quad, radius, radius, 2*radius, 10, 1)
        gluDisk(self.quad, 0, radius, 10, 1)
        glTranslatef(0, 0, 2*radius)
        gluDisk(self.quad, 0, radius, 10, 1)
        glPopMatrix()
        glRotatef(-90, 1, 0, 0)
        gluCylinder(self.quad, radius*0.9, radius*0.6,
                    self.puppet.dimensions['lfoot_length'], 10, 1)
        glTranslatef(0,0,self.puppet.dimensions['lfoot_length'])
        gluDisk(self.quad, 0, radius*0.6, 10, 1)
        
    def draw_right_femur(self):
        radius = self.puppet.dimensions['rfemur_length']/8
        
        gluSphere(self.quad, radius, 10, 10)
        glRotatef(180, 1, 0, 0)
        gluCylinder(self.quad, radius*0.9, radius*0.6,
                    self.puppet.dimensions['rfemur_length'], 10, 1)
        
    def draw_right_tibia(self):
        radius = self.puppet.dimensions['rfemur_length']/8*0.6

        glPushMatrix()
        glRotatef(90, 0, 1, 0)
        glTranslatef(0, 0, -radius)
        gluCylinder(self.quad, radius, radius, 2*radius, 10, 1)
        gluDisk(self.quad, 0, radius, 10, 1)
        glTranslatef(0, 0, 2*radius)
        gluDisk(self.quad, 0, radius, 10, 1)
        glPopMatrix()
        glRotatef(180, 1, 0, 0)
        gluCylinder(self.quad, radius*0.9, radius*0.6,
                    self.puppet.dimensions['rtibia_length'], 10, 1)
        
    def draw_right_foot(self):
        radius = self.puppet.dimensions['rfemur_length']/8*0.6*0.6

        glPushMatrix()
        glRotatef(90, 0, 1, 0)
        glTranslatef(0, 0, -radius)
        gluCylinder(self.quad, radius, radius, 2*radius, 10, 1)
        gluDisk(self.quad, 0, radius, 10, 1)
        glTranslatef(0, 0, 2*radius)
        gluDisk(self.quad, 0, radius, 10, 1)
        glPopMatrix()
        glRotatef(-90, 1, 0, 0)
        gluCylinder(self.quad, radius*0.9, radius*0.6,
                    self.puppet.dimensions['rfoot_length'], 10, 1)
        glTranslatef(0,0,self.puppet.dimensions['rfoot_length'])
        gluDisk(self.quad, 0, radius*0.6, 10, 1)
    

        


if __name__ == '__main__':
    puppet = Puppet(joint_forces=False, string_forces=False, string_constraints=True)
    puppet.get_config('torso_rx').q = 0.1
    puppet.get_config('torso_ry').q = 0.1
    puppet.project_string_controls()

    q0 = puppet.get_q()
    u0 = tuple([0.0]*len(puppet.inputs))
    qk2 = puppet.get_qk()
    dt = 0.01
    # Create and initialize a variational integrator for the system.
    gmvi = trep.MidpointVI(puppet)
    gmvi.initialize_from_configs(0.0, q0, dt, q0)


    q = [gmvi.q2]
    t = [gmvi.t2]
    while gmvi.t1 < 10.0:
        gmvi.step(gmvi.t2+dt, u0, qk2)
        q.append(gmvi.q2)
        t.append(gmvi.t2)
        # The puppet can take a while to simulate, so print out the time
        # occasionally to indicate our progress.
        if abs(gmvi.t2 - round(gmvi.t2)) < dt/2.0:
            print "t =",gmvi.t2


    visualize_3d([PuppetVisual(puppet, t, q)])

    
