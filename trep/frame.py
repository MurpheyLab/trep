import trep
import _trep
from _trep import _Frame
from config import Config
import numpy as np
import numpy.linalg
import inspect
from math import sin, cos
from itertools import chain
    
class FrameDef(object):
    def __init__(self, transform_type, param, name, kinematic, mass):
        self.transform_type = transform_type
        self.param = param
        self.name = name
        self.kinematic = kinematic
        self.mass = mass
    def __repr__(self):
        return "<FrameDef %s>" % (self.transform_type)
        
def tx(param, name=None, kinematic=False, mass=0.0):
    return FrameDef(trep.TX, param, name, kinematic, mass)
def ty(param, name=None, kinematic=False, mass=0.0):
    return FrameDef(trep.TY, param, name, kinematic, mass)
def tz(param, name=None, kinematic=False, mass=0.0):
    return FrameDef(trep.TZ, param, name, kinematic, mass)
def rx(param, name=None, kinematic=False, mass=0.0):
    return FrameDef(trep.RX, param, name, kinematic, mass)
def ry(param, name=None, kinematic=False, mass=0.0):
    return FrameDef(trep.RY, param, name, kinematic, mass)
def rz(param, name=None, kinematic=False, mass=0.0):
    return FrameDef(trep.RZ, param, name, kinematic, mass)
def const_se3(se3, name=None, kinematic=False, mass=0.0):
    return FrameDef(trep.CONST_SE3, se3, name, kinematic, mass)
def const_txyz(xyz, name=None, kinematic=False, mass=0.0):
    param = [[1,0,0],[0,1,0],[0,0,1],xyz]
    return FrameDef(trep.CONST_SE3, param, name, kinematic, mass)


def rotation_matrix(theta, axis):
    """
    Build a 4x4 SE3 matrix corresponding to a rotation of theta
    radians around axis.
    """
    axis = np.array(axis)
    axis /= np.linalg.norm(axis)
    w_hat = np.array( ((0, -axis[2], axis[1]),
                       (axis[2], 0, -axis[0]),
                       (-axis[1], axis[0], 0)) )
    rot = np.eye(3) + w_hat * sin(theta) + w_hat*w_hat*(1-cos(theta))
    se3 = np.zeros((4,4))
    se3[:3,:3] = rot
    return se3


def check_and_sort_configs(fail_value, skip_sort=0):
    ## This a decorator for all the frame derivative functions.  It
    ## adds a wrapper around the function to make sure that the frame
    ## depends on every configuration variable passed to the function.
    ## If not, the function will return fail_value instead of calling
    ## the derivative function.  Otherwise, it will sort the
    ## configuration variables so they access the correct part of the
    ## cache (only the 'upper triangular' part of the cache is filled
    ## out for symmetric values.

    def decorator(func):
        def check_and_sort(self, *configs):
            for q in configs:
                assert isinstance(q, _trep._Config)
                if not self.uses_config(q):
                    return fail_value
            # All configs affect this frame, convert to indices and
            # sort.
            unsorted = [q._config_gen for q in configs[:skip_sort]]
            sorted_ =  [q._config_gen for q in configs[skip_sort:]]
            indices = unsorted + sorted(sorted_)
            return func(self, *indices)

        # At this point, we could return check_and_sort() and be done,
        # but the resulting functions would all be be called
        # "check_and_sort", have the same anonymous structure, and no
        # doc string.  Instead, we want the returned function to
        # appear identical to the original.  We use inspect to find
        # out the arguments of func and then create a new function
        # with the same name and arguments that calls check_and_sort.
        # Basically a wrapper for our wrapper.
                
        spec = inspect.getargspec(func)
        if spec.varargs != None:
            raise TypeError("variable args are not supported")
        if spec.keywords != None:
            raise TypeError("keywords are not supported")
        if spec.defaults is not None:
            raise TypeError("default arguments not supported")

        signature = ", ".join(spec.args)
        src =  "def %s(%s):\n" % (func.__name__, signature)
        src += "    return check_and_sort(%s)" % ','.join(spec.args)

        context = {'func' : func, 'check_and_sort' : check_and_sort}
        exec src in context

        wrapper = context[func.__name__]
        wrapper.__dict__ = func.__dict__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


class Frame(_Frame):
    """
    Frame represents a coordinate frame in a mechanical system.  Each
    frame is defined as a child of a parent frame and is located by a
    rigid body transformation from the parent frame's coordinate
    system.
    """
    
    def __init__(self, parent, transform, param, name=None, kinematic=False, mass=0.0):
        _Frame.__init__(self)
        # Initialize _Frame components
        self._system = None
        self._config = None
        self._parent = None
        self._transform = None
        self._children = tuple()
        self._cache_size = 0
        self._allocated_cache_size = 0
        self._cache_index = tuple()

        self.name = name
        if transform == trep.WORLD:
            assert isinstance(parent, _trep._System)
            self._system = parent
            self._transform = trep.WORLD
            self._update_transform()
            self._parent = None
        elif transform in [trep.TX, trep.TY, trep.TZ, trep.RX, trep.RY, trep.RZ]:
            assert isinstance(parent, _trep._Frame)
            self._system = parent.system
            self._transform = transform
            self._update_transform()
            self._parent = parent
            parent._add_child(self)

            if isinstance(param, str):
                self._config = Config(self._system, param, kinematic=kinematic)
            else:
                self._value = float(param)
        elif transform == trep.CONST_SE3:
            assert isinstance(parent, _trep._Frame)
            self._system = parent.system
            self._transform = transform
            self._update_transform()
            self._parent = parent
            parent._add_child(self)
            
            self.set_SE3(param[0], param[1], param[2], param[3])
        else:
            raise StandardError('Unknown frame transform: %r' % transform)

        self.set_mass(mass)

    def __repr__(self):
        if self.transform_type == trep.WORLD:
            return "<Frame '%s'>" % self.name
        elif self.config == None:
            return "<Frame '%s' %s(%s) %f>" % (
                self.name, self.transform_type, self.transform_value, self.mass)
        else:
            return "<Frame '%s' %s(%s)>" % (
                self.name, self.transform_type, self.config.name)


    def tree_view(self, indent=0):
        """
        Return a string that visually describes this frame and it's
        descendants.
        """
        view = [indent*' ' + ('%r' % self)]
        view += [c.tree_view(indent+3) for c in self.children]
        return '\n'.join(view)


    def uses_config(self, q):
        """Return True if the frame depends on q"""
        return self._cache_index[q._config_gen] == q


    def flatten_tree(self):
        """
        Appends the frame and all of its descendants to frame_list.
        If frame_list is None, a new list is created.    
        """
        flat = [self]
        for child in self.children:
            flat += child.flatten_tree()
        return flat

                
    def import_frames(self, children):
        """
        Import a tree of frames from a tree description.
        """
        self.system.hold_structure_changes()
        while children:
            info = children[0]
            children = children[1:]
            if not isinstance(info, trep.frame.FrameDef):
                raise TypeError('Frame definition expected instead of: %r', info)

            frame = trep.Frame(self, info.transform_type, info.param,
                               name=info.name,
                               kinematic=info.kinematic,
                               mass=info.mass)

            if children and isinstance(children[0], list):
                frame.import_frames(children[0])
                children = children[1:]
        self.system.resume_structure_changes()


    def export_frames(self, tabs=0, tab_size=4):
        """
        Export a frame and it's children into (a plain-text
        definition of) the python-list format used by
        trep.Frame.import_frames to define frame trees.
        """
        frame_def_mapping = {
            trep.TX : 'tx',
            trep.TY : 'ty',
            trep.TZ : 'tz',
            trep.RX : 'rx',
            trep.RY : 'ry',
            trep.RZ : 'rz',
            trep.CONST_SE3 : 'const_se3'
            }

        txt = ' '*tab_size*tabs + '%s(' % frame_def_mapping[self.transform_type]
        params = []
        if self.transform_type == trep.CONST_SE3:
            mat = self.lg()
            columns = [ '[' + ', '.join(['%s' % i for i in mat[:,j][:3]]) + ']' for j in range(4)]
            params.append('[%s, %s, %s, %s]' % tuple(columns))
        else:
            if self.config == None:
                params.append('%s' % self.transform_value)
            else:
                params.append("'%s'" % self.config.name)
                if self.config.kinematic:
                    params.append('kinematic=True')
        if self.name:
            params.append("name='%s'" % self.name)
        if self.mass:
            if self.Ixx or self.Iyy or self.Izz:
                params.append('mass=[%s, %s, %s, %s]' % (self.mass, self.Ixx, self.Iyy, self.Izz))
            else:
                params.append('mass=%s' % self.mass)
        txt += ', '.join(params)     
        txt += ')'
        if self.children:
            txt += ', [\n' + ',\n'.join([child.export_frames(tabs+1, tab_size) for child in self.children])
            txt += ']'        
        return txt


    @property
    def system(self):
        """System that the frame is a part of."""
        return self._system


    @property 
    def config(self):
        """
        Configuration variable that drives the frame's transformation.

        None for constant-transformation frames.
        """
        return self._config


    @property
    def parent(self):
        """Parent of the frame.  None for the World Frame."""
        return self._parent


    @property
    def children(self):
        """Tuple of the frame's child frames."""
        return self._children


    @property
    def transform_type(self):
        """Transformation type of the coordinate frame."""
        return self._transform


    @property
    def transform_value(self):
        """
        Current value of the frame's transformation parameters.  This
        will either be the fixed transformation parameter or the value
        of the frame's configuration variable.
        """
        if self.config:
            return self.config.q
        else:
            return self._value

    @transform_value.setter
    def transform_value(self, value):
        if self.config:
            self.config.q = value
        else:
            self._value = value


    def set_SE3(self, Rx=(1,0,0), Ry=(0,1,0), Rz=(0,0,1), p=(0,0,0)):
        """
        Set the SE3 transformation for a const_SE3 frame.
        """        
        # Do a really stupid normalization for now.
        Rx = np.array(Rx)
        Rx = Rx/np.linalg.norm(Rx)
        Rz = np.cross(Rx, Ry)
        Rz = Rz/np.linalg.norm(Rz)
        Ry = np.cross(Rz, Rx)
        Ry = Ry/np.linalg.norm(Ry)
     
        self._set_SE3(tuple(Rx[0:3]), tuple(Ry[0:3]), tuple(Rz[0:3]), tuple(p))
    
        
    def set_mass(self, mass, Ixx=0.0, Iyy=0.0, Izz=0.0):
        """
        Set the system's inertial properies.

        mass -> scalar or sequence of 4 scalars
        """
        try:
            Ixx = mass[1]
            Iyy = mass[2]
            Izz = mass[3]
            mass = mass[0]
        except TypeError, e:
            pass
        # Convert to floats here so we don't get an exception in the
        # middle of changing the system.
        mass = float(mass)
        Ixx = float(Ixx)
        Iyy = float(Iyy)
        Izz = float(Izz)
        self._mass = mass
        self._Ixx = Ixx
        self._Iyy = Iyy
        self._Izz = Izz
        self.system._structure_changed()   

    @property
    def mass(self):
        "Mass of the frame."
        return self._mass
    @mass.setter
    def mass(self, mass):
        self._mass = mass
        self.system._structure_changed()   

    @property
    def Ixx(self):
        "Rotational inertia about the frame's X axis."        
        return self._Ixx
    @Ixx.setter
    def Ixx(self, Ixx):
        self._Ixx = Ixx
        self.system._structure_changed()   

    @property
    def Iyy(self):
        "Rotational inertia about the frame's Y axis."        
        return self._Iyy
    @Iyy.setter
    def Iyy(self, Iyy):
        self._Iyy = Iyy
        self.system._structure_changed()   

    @property
    def Izz(self):
        "Rotational inertia about the frame's Z axis."        
        return self._Izz
    @Izz.setter
    def Izz(self, Izz):
        self._Izz = Izz
        self.system._structure_changed()   


    def lg(self):
        """
        Local coordinate transformation of the frame (ie,
        transformation from parent frame.
        """
        return self._lg()

    def lg_dq(self):
        """
        Derivative of the local coordinate transformation with respect
        to the frame's transformation parameter.
        """
        return self._lg_dq()

    def lg_dqdq(self):
        """
        Second derivative of the local coordinate transformation with
        respect to the frame's transformation parameter.
        """
        return self._lg_dqdq()
    
    def lg_dqdqdq(self):
        """
        Third derivative of the local coordinate transformation with
        respect to the frame's transformation parameter.
        """
        return self._lg_dqdqdq()
    
    def lg_dqdqdqdq(self):
        """
        Fourth derivative of the local coordinate transformation with
        respect to the frame's transformation parameter.
        """
        return self._lg_dqdqdqdq()

    def lg_inv(self):
        """Inverse of the local coordinate transformation."""
        return self._lg_inv()

    def lg_inv_dq(self):
        """
        First derivative of the inverse of the local coordinate
        transformation with respect to the frame's transformation
        parameter.
        """
        return self._lg_inv_dq()
    
    def lg_inv_dqdq(self):
        """
        First derivative of the inverse of the local coordinate
        transformation with respect to the frame's transformation
        parameter.
        """
        return self._lg_inv_dqdq()
    
    def lg_inv_dqdqdq(self):
        """
        Third derivative of the inverse of the local coordinate
        transformation with respect to the frame's transformation
        parameter.
        """
        return self._lg_inv_dqdqdq()
    
    def lg_inv_dqdqdqdq(self):
        """
        Fourth derivative of the inverse of the local coordinate
        transformation with respect to the frame's transformation
        parameter.
        """
        return self._lg_inv_dqdqdqdq()
        
    def twist_hat(self):
        return self._twist_hat()
    
    def g(self):
        """The frame's current global position (in SE(3) )."""
        return self._g()

    @check_and_sort_configs(np.zeros((4,4)))
    def g_dq(self, q1):
        """
        The derivative of frame's current global position (in SE(3) )
        with respect to the value of q1.
        """
        self.system._update_cache(_trep.SYSTEM_CACHE_G_DQ)
        return self._g_dq[q1]

    @check_and_sort_configs(np.zeros((4,4)))
    def g_dqdq(self, q1, q2):
        """
        The second derivative of frame's current global position (in
        SE(3) ) with respect to the values of q1 and q2.
        """
        self.system._update_cache(_trep.SYSTEM_CACHE_G_DQDQ)
        return self._g_dqdq[q1, q2]

    @check_and_sort_configs(np.zeros((4,4)))
    def g_dqdqdq(self, q1, q2, q3):
        """
        The third derivative of frame's current global position (in
        SE(3) ) with respect to the values of q1, q2, and q3.
        """
        self.system._update_cache(_trep.SYSTEM_CACHE_G_DQDQDQ)
        return self._g_dqdqdq[q1, q2, q3]

    @check_and_sort_configs(np.zeros((4,4)))
    def g_dqdqdqdq(self, q1, q2, q3, q4):
        """
        The fourth derivative of frame's current global position (in
        SE(3) ) with respect to the values of q1, q2, q3, and q4.
        """
        self.system._update_cache(_trep.SYSTEM_CACHE_G_DQDQDQDQ)
        return self._g_dqdqdqdq[q1, q2, q3, q4]
    
    def g_inv(self):
        """
        The inverse of the frame's current global position (in SE(3)).
        """
        return self._g_inv()

    @check_and_sort_configs(np.zeros((4,4)))
    def g_inv_dq(self, q1):
        """
        The derivative of the inverse of the frame's current global
        position (in SE(3) ) with respect to the value of q1.
        """
        self.system._update_cache(_trep.SYSTEM_CACHE_G_INV_DQ)
        return self._g_inv_dq[q1]

    @check_and_sort_configs(np.zeros((4,4)))
    def g_inv_dqdq(self, q1, q2):
        """
        The second derivative of the inverse of the frame's current
        global position (in SE(3) ) with respect to the values of q1
        and q2.
        """
        self.system._update_cache(_trep.SYSTEM_CACHE_G_INV_DQDQ)
        return self._g_inv_dqdq[q1, q2]
    
    def p(self):
        """The frame's current global position (in R^3 )."""
        return self._p()

    @check_and_sort_configs(np.zeros((4,)))
    def p_dq(self, q1):
        """
        The derivative of frame's current global position (in R^3 )
        with respect to the value of q1.
        """
        self.system._update_cache(_trep.SYSTEM_CACHE_G_DQ)
        return self._p_dq[q1]

    @check_and_sort_configs(np.zeros((4,)))
    def p_dqdq(self, q1, q2):
        """
        The second derivative of frame's current global position (in
        R^3 ) with respect to the values of q1 and q2.
        """
        self.system._update_cache(_trep.SYSTEM_CACHE_G_DQDQ)
        return self._p_dqdq[q1, q2]
    
    @check_and_sort_configs(np.zeros((4,)))
    def p_dqdqdq(self, q1, q2, q3):
        """
        The third derivative of frame's current global position (in
        R^3 ) with respect to the values of q1, q2, and q3.
        """
        self.system._update_cache(_trep.SYSTEM_CACHE_G_DQDQDQ)
        return self._p_dqdqdq[q1, q2, q3]
    
    @check_and_sort_configs(np.zeros((4,)))
    def p_dqdqdqdq(self, q1, q2, q3, q4):
        """
        The fourth derivative of frame's current global position (in
        R^3 ) with respect to the values of q1, q2, q3, and q4.
        """
        self.system._update_cache(_trep.SYSTEM_CACHE_G_DQDQDQDQ)
        return self._p_dqdqdqdq[q1, q2, q3, q4]
    
    def vb(self):
        """The body velocity of the frame (in se(3))."""
        return self._vb()

    @check_and_sort_configs(np.zeros((4,4)))
    def vb_dq(self, q1):
        """
        The derivative of the body velocity of the frame (in se(3))
        with respect to the value of q1.
        """
        self.system._update_cache(_trep.SYSTEM_CACHE_VB_DQ)
        return self._vb_dq[q1]

    @check_and_sort_configs(np.zeros((4,4)))
    def vb_dqdq(self, q1, q2):
        """
        The second derivative of the body velocity of the frame (in
        se(3)) with respect to the values of q1 and q2.
        """
        self.system._update_cache(_trep.SYSTEM_CACHE_VB_DQDQ)
        return self._vb_dqdq[q1, q2]

    @check_and_sort_configs(np.zeros((4,4)))
    def vb_dqdqdq(self, q1, q2, q3):
        """
        The third derivative of the body velocity of the frame (in
        se(3)) with respect to the values of q1, q2, and q3.
        """
        self.system._update_cache(_trep.SYSTEM_CACHE_VB_DQDQDQ)
        return self._vb_dqdqdq[q1, q2, q3]
    
    @check_and_sort_configs(np.zeros((4,4)))
    def vb_ddq(self, dq1):
        """
        The derivative of the body velocity of the frame (in se(3))
        with respect to the velocity of dq1.
        """
        self.system._update_cache(_trep.SYSTEM_CACHE_VB_DDQ)
        return self._vb_ddq[dq1]
    
    @check_and_sort_configs(np.zeros((4,4)), skip_sort=1)
    def vb_ddqdq(self, dq1, q2):
        """
        The second derivative of the body velocity of the frame (in
        se(3)) with respect to the velocity of dq1 and the value of
        q2.
        """
        self.system._update_cache(_trep.SYSTEM_CACHE_VB_DDQDQ)
        return self._vb_ddqdq[dq1, q2]
    
    @check_and_sort_configs(np.zeros((4,4)), skip_sort=1)
    def vb_ddqdqdq(self, dq1, q2, q3):
        """
        The third derivative of the body velocity of the frame (in
        se(3)) with respect to the velocity of dq1 and the values of
        q2 and q3.
        """
        self.system._update_cache(_trep.SYSTEM_CACHE_VB_DDQDQDQ)
        return self._vb_ddqdqdq[dq1, q2, q3]

    @check_and_sort_configs(np.zeros((4,4)), skip_sort=1)
    def vb_ddqdqdqdq(self, dq1, q2, q3, q4):
        """
        The fourth derivative of the body velocity of the frame (in
        se(3)) with respect to the velocity of dq1 and the values of
        q2, q3, and q4.
        """
        self.system._update_cache(_trep.SYSTEM_CACHE_VB_DDQDQDQDQ)
        return self._vb_ddqdqdqdq[dq1, q2, q3, q4]

    def _add_child(self, child):
        """
        Internal use only.

        _add_child(child) -> None
        
        Append child to the Frame's child list.
        """
        assert isinstance(child, _trep._Frame)
        self._children += (child, )
    
    def _structure_changed(self):
        """
        Internal use only.

        _structure_changed() -> None

        _structure_changed() rebuilds the values/arrays used for the caching
        implementation.  This involves allocated buffers to save the
        results in and creating an index of the configuration
        variables that the Frame depends on.  This should be called,
        for example, when Frames' parameters are modified.
        """

        # Update cache_index

        # cache_index is a tuple of all the configuration variables
        # from the world frame to this frame.  Then it is padded with
        # None until it has a length of nQ+1.  This is an optimizatoin
        # that allows us to quickly determine if a frame depends on a
        # configuration variable, which was an observed bottleneck in
        # profiling.


        
        # Update cache_index then fix the buffers
        configs = [None]*(len(self.system.configs)+1)
        frame = self
        while frame != None:
            if frame.config != None:
                configs.insert(0, frame.config)
            frame = frame.parent
        configs = configs[:len(self.system.configs)+1]
        self._cache_size = configs.index(None)
        self._cache_index = tuple(configs)

        # Update frame cache structures and anything
        if self._allocated_cache_size != self._cache_size:
            # The np arrays MUST be contiguous C-style arrays of doubles.
            n = self._cache_size
            self._g_dq         = np.zeros(      (n,4,4), np.double, 'C')
            self._g_inv_dq     = np.zeros(      (n,4,4), np.double, 'C')
            self._g_inv_dqdq   = np.zeros(    (n,n,4,4), np.double, 'C')
            self._p_dq         = np.zeros(        (n,4), np.double, 'C')
            self._vb_dq        = np.zeros(      (n,4,4), np.double, 'C')
            self._vb_ddq       = np.zeros(      (n,4,4), np.double, 'C')
            self._allocated_cache_size = n

        # These are automatically resized as needed.
        n = 1
        self._g_dqdq       = np.zeros(    (n,n,4,4), np.double, 'C')
        self._p_dqdq       = np.zeros(      (n,n,4), np.double, 'C')
        self._g_dqdqdq     = np.zeros(  (n,n,n,4,4), np.double, 'C')
        self._p_dqdqdq     = np.zeros(    (n,n,n,4), np.double, 'C')
        self._g_dqdqdqdq   = np.zeros((n,n,n,n,4,4), np.double, 'C')
        self._p_dqdqdqdq   = np.zeros(  (n,n,n,n,4), np.double, 'C')

        self._vb_dqdq      = np.zeros(    (n,n,4,4), np.double, 'C')
        self._vb_dqdqdq    = np.zeros(  (n,n,n,4,4), np.double, 'C')
        self._vb_ddqdq     = np.zeros(    (n,n,4,4), np.double, 'C')
        self._vb_ddqdqdq   = np.zeros(  (n,n,n,4,4), np.double, 'C')
        self._vb_ddqdqdqdq = np.zeros((n,n,n,n,4,4), np.double, 'C')

        for child in self.children:
            child._structure_changed()

