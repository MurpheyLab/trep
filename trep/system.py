import math
import inspect
import numpy as np
import numpy.linalg as linalg
import scipy as sp
import scipy.optimize
import scipy.io
from itertools import product

import trep
import _trep
from _trep import _System
from frame import Frame
from finput import Input
from config import Config
from force import Force
from constraint import Constraint
from potential import Potential

from util import dynamics_indexing_decorator

class System(_System):
    """
    The System class represents a complete mechanical system
    comprising coordinate frames, configuration variables, potential
    energies, constraints, and forces.
    """
    def __init__(self):
        """
        Create a new mechanical system.
        """
        _System.__init__(self)
        # _System variables need to be initialized (cleaner here than in C w/ ref counting)
        self._frames = tuple()
        self._configs = tuple()
        self._dyn_configs = tuple()
        self._kin_configs = tuple()
        self._potentials = tuple()
        self._forces = tuple()
        self._inputs = tuple()
        self._constraints = tuple()
        self._masses = tuple()

        self._hold_structure_changes = 0
        self._structure_changed_funcs = []

        # Hold off the initial structure update until we have a world
        # frame.
        self._hold_structure_changes = 1
        self._world_frame = Frame(self, trep.WORLD, None, name="World")
        self._hold_structure_changes = 0
        self._structure_changed()

    def __repr__(self):
        return '<System %d configs, %d frames, %d potentials, %d constraints, %d forces, %d inputs>' % (
            len(self.configs),
            len(self.frames),
            len(self.potentials),
            len(self.constraints),
            len(self.forces),
            len(self.inputs))

    @property
    def nQ(self):
        """Number of configuration variables in the system."""
        return len(self.configs)

    @property
    def nQd(self):
        """Number of dynamic configuration variables in the system."""
        return len(self.dyn_configs)

    @property
    def nQk(self):
        """Number of kinematic configuration variables in the system."""
        return len(self.kin_configs)

    @property
    def nu(self):
        """Number of inputs in the system."""
        return len(self.inputs)

    @property
    def nc(self):
        """Number of constraints in the system."""
        return len(self.constraints)

    @property
    def t(self):
        """Current time of the system."""
        return self._time

    @t.setter
    def t(self, value):
        self._clear_cache()
        self._time = value

    def get_frame(self, identifier):
        """
        get_frame(identifier) -> Frame,None

        Return the first frame with the matching identifier.  The
        identifier can be the frame name, index, or the frame itself.
        Raise an exception if no match is found.
        """
        return self._get_object(identifier, Frame, self.frames)

    def get_config(self, identifier):
        """
        get_config(identifier) -> Config,None

        Return the first configuration variable with the matching
        identifier.  The identifier can be the config name, index, or
        the config itself.  Raise an exception if no match is found.
        """
        return self._get_object(identifier, Config, self.configs)

    def get_potential(self, identifier):
        """
        get_potential(identifier) -> Potential,None

        Return the first potential with the matching identifier.  The
        identifier can be the constraint name, index, or the
        constraint itself.  Raise an exception if no match is found.
        """
        return self._get_object(identifier, Potential, self.potentials)

    def get_constraint(self, identifier):
        """
        get_constraint(identifier) -> Constraint,None

        Return the first constraint with the matching identifier.  The
        identifier can be the constraint name, index, or the
        constraint itself.  Raise an exception if no match is found.
        """
        return self._get_object(identifier, Constraint, self.constraints)

    def get_force(self, identifier):
        """
        get_force(identifier) -> Force,None

        Return the first force with the matching identifier.  The
        identifier can be the force name, index, or the
        force itself.  Raise an exception if no match is found.
        """
        return self._get_object(identifier, Force, self.forces)

    def get_input(self, identifier):
        """
        get_input(identifier) -> Input,None

        Return the first input with the matching identifier.  The
        identifier can be the input name, index, or the
        input itself.  Raise an exception if no match is found.
        """
        return self._get_object(identifier, Input, self.inputs)

    def satisfy_constraints(self, tolerance=1e-10, verbose=False,
                            keep_kinematic=False, constant_q_list=None):
        """
        Modify the current configuration to satisfy the system
        constraints.

        The configuration velocity (ie, config.dq) is simply set to
        zero.  This should be fixed in the future.

        Passing True to keep_kinematic will not allow method to modify
        kinematic configuration variables.

        Passing a list (or tuple) of configurations to constant_q_list
        will keep all elements in list constant.  The method uses
        trep.System.get_config so the list may contain configuration
        objects, indices in Q, or names.  Passing anything for
        constant_list_q will overwrite value for keep_kinematic.
        """
        self.dq = 0
        if keep_kinematic:
            names = [q.name for q in self.dyn_configs]
            q0 = self.qd
        else:
            names = [q.name for q in self.configs]
            q0 = self.q
        if constant_q_list:
            connames = [self.get_config(q).name for q in constant_q_list]
            names = []
            for q in self.configs:
                if q.name not in connames:
                    names.append(q.name)
            q0 = np.array([self.q[self.get_config(name).index] for name in names])

        def func(q):
            v = (q - q0)
            return np.dot(v,v)

        def fprime(q):
            return 2*(q-q0)

        def f_eqcons(q):
            self.q = dict(zip(names,q))
            return np.array([c.h() for c in self.constraints])

        def fprime_eqcons(q):
            self.q = dict(zip(names,q))
            return np.array([[c.h_dq(self.get_config(q)) for q in names] for c in self.constraints])

        (q_opt, fx, its, imode, smode) = sp.optimize.fmin_slsqp(func, q0, f_eqcons=f_eqcons,
                                                                fprime=fprime, fprime_eqcons=fprime_eqcons,
                                                                acc=tolerance, iter=100*self.nQ,
                                                                iprint=0, full_output=True)
        if imode != 0:
            raise StandardError("Minimization failed: %s" % smode)
        self.q = dict(zip(names,q_opt))
        return self.q

    def minimize_potential_energy(self, tolerance=1e-10, verbose=False,
                                  keep_kinematic=False, constant_q_list=None):
        """
        Find a nearby configuration where the potential energy is
        minimized.  Useful for finding nearby equilibrium points.
        If minimum is found, all constraints will be found as well

        The configuration velocity (ie, config.dq) is set to
        zero which ensures the kinetic energy is zero.

        Passing True to keep_kinematic will not allow method to modify
        kinematic configuration variables.

        Passing a list (or tuple) of configurations to constant_q_list
        will keep all elements in list constant.  The method uses
        trep.System.get_config so the list may contain configuration
        objects, indices in Q, or names.  Passing anything for
        constant_list_q will overwrite value for keep_kinematic.
        """
        self.dq = 0
        if keep_kinematic:
            names = [q.name for q in self.dyn_configs]
            q0 = self.qd
        else:
            names = [q.name for q in self.configs]
            q0 = self.q
        if constant_q_list:
            connames = [self.get_config(q).name for q in constant_q_list]
            names = []
            for q in self.configs:
                if q.name not in connames:
                    names.append(q.name)
            q0 = np.array([self.q[self.get_config(name).index] for name in names])


        def func(q):
            self.q = dict(zip(names,q))
            return -self.L()

        def fprime(q):
            return [-self.L_dq(self.get_config(name)) for name in names]

        def f_eqcons(q):
            self.q = dict(zip(names,q))
            return np.array([c.h() for c in self.constraints])

        def fprime_eqcons(q):
            self.q = dict(zip(names,q))
            return np.array([[c.h_dq(self.get_config(q)) for q in names] for c in self.constraints])

        (q_opt, fx, its, imode, smode) = sp.optimize.fmin_slsqp(func, q0, f_eqcons=f_eqcons,
                                                                fprime=fprime, fprime_eqcons=fprime_eqcons,
                                                                acc=tolerance, iter=100*self.nQ,
                                                                iprint=0, full_output=True)
        if imode != 0:
            raise StandardError("Minimization failed: %s" % smode)
        self.q = dict(zip(names,q_opt))
        return self.q


    def set_state(self, q=None, dq=None, u=None, ddqk=None, t=None):
        """
        Set the current state of the system, not including the "output" ddqd.
        """
        if q is not None: self.q = q
        if dq is not None: self.dq = dq
        if u is not None: self.u = u
        if ddqk is not None: self.ddqk = ddqk
        if t is not None: self.t = t

    def import_frames(self, children):
        """
        Adds children to this system's world frame using a special
        frame definition.  See Frame.import_frames() for details.
        """
        self.world_frame.import_frames(children)

    def export_frames(self, system_name='system', frames_name='frames', tab_size=4):
        """
        Create python source code to define this system's frames.
        """
        txt = ''
        txt += '#'*80 + '\n'
        txt += '# Frame tree definition generated by System.%s()\n\n' % inspect.stack()[0][3]
        txt += 'from trep import %s\n' % ', '.join(sorted(trep.frame.frame_def_mapping.values()))
        txt += '%s = [\n' % frames_name
        txt += ',\n'.join([child.export_frames(1, tab_size) for child in self.world_frame.children]) + '\n'
        txt += ' '*tab_size + ']\n'
        txt += '%s.import_frames(%s)\n' % (system_name, frames_name)
        txt += '#'*80 + '\n'
        return txt


    @property
    def q(self):
        """Current configuration of the system."""
        return np.array([q.q for q in self.configs])


    @q.setter
    def q(self, value):
        # Writing c.q will clear system cache
        if isinstance(value, (int, float)):
            for q in self.configs:
                q.q = value
        elif isinstance(value, dict):
            for name, v in value.iteritems():
                self.get_config(name).q = v
        else:
            for q,v in zip(self.configs, value):
                q.q = v


    @property
    def dq(self):
        """ Current configuration velocity of the system """
        return np.array([q.dq for q in self.configs])

    @dq.setter
    def dq(self, value):
        # Writing c.dq will clear system cache
        if isinstance(value, (int, float)):
            for q in self.configs:
                q.dq = value
        elif isinstance(value, dict):
            for name, v in value.iteritems():
                self.get_config(name).dq = v
        else:
            for q,v in zip(self.configs, value):
                q.dq = v

    @property
    def ddq(self):
        """ Current configuration acceleration of the system """
        return np.array([q.ddq for q in self.configs])

    @ddq.setter
    def ddq(self, value):
        # Writing c.ddq will clear system cache
        if isinstance(value, (int, float)):
            for q in self.configs:
                q.ddq = value
        elif isinstance(value, dict):
            for name, v in value.iteritems():
                self.get_config(name).ddq = v
        else:
            for q,v in zip(self.configs, value):
                q.ddq = v


    @property
    def qd(self):
        """Dynamic part of the system's current configuration."""
        return np.array([q.q for q in self.dyn_configs])

    @qd.setter
    def qd(self, value):
        # Writing c.q will clear system cache
        if isinstance(value, (int, float)):
            for q in self.dyn_configs:
                q.q = value
        elif isinstance(value, dict):
            for name, v in value.iteritems():
                self.get_config(name).q = v
        else:
            for q,v in zip(self.dyn_configs, value):
                q.q = v


    @property
    def dqd(self):
        """Dynamic part of the system's current configuration velocity."""
        return np.array([q.dq for q in self.dyn_configs])

    @dqd.setter
    def dqd(self, value):
        # Writing c.q will clear system cache
        if isinstance(value, (int, float)):
            for q in self.dyn_configs:
                q.dq = value
        elif isinstance(value, dict):
            for name, v in value.iteritems():
                self.get_config(name).dq = v
        else:
            for q,v in zip(self.dyn_configs, value):
                q.dq = v


    @property
    def ddqd(self):
        """Dynamic part of the system's current configuration acceleration."""
        return np.array([q.ddq for q in self.dyn_configs])

    @ddqd.setter
    def ddqd(self, value):
        # Writing c.q will clear system cache
        if isinstance(value, (int, float)):
            for q in self.dyn_configs:
                q.ddq = value
        elif isinstance(value, dict):
            for name, v in value.iteritems():
                self.get_config(name).ddq = v
        else:
            for q,v in zip(self.dyn_configs, value):
                q.ddq = v


    @property
    def qk(self):
        """Kinematic part of the system's current configuration."""
        return np.array([q.q for q in self.kin_configs])

    @qk.setter
    def qk(self, value):
        # Writing c.q will clear system cache
        if isinstance(value, (int, float)):
            for q in self.kin_configs:
                q.q = value
        elif isinstance(value, dict):
            for name, v in value.iteritems():
                self.get_config(name).q = v
        else:
            for q,v in zip(self.kin_configs, value):
                q.q = v


    @property
    def dqk(self):
        """Kinematic part of the system's current configuration velocity."""
        return np.array([q.dq for q in self.kin_configs])

    @dqk.setter
    def dqk(self, value):
        # Writing c.q will clear system cache
        if isinstance(value, (int, float)):
            for q in self.kin_configs:
                q.dq = value
        elif isinstance(value, dict):
            for name, v in value.iteritems():
                self.get_config(name).dq = v
        else:
            for q,v in zip(self.kin_configs, value):
                q.dq = v


    @property
    def ddqk(self):
        """Kinematic part of the system's current configuration acceleration."""
        return np.array([q.ddq for q in self.kin_configs])

    @ddqk.setter
    def ddqk(self, value):
        # Writing c.ddq will clear system cache
        if isinstance(value, (int, float)):
            for q in self.kin_configs:
                q.ddq = value
        elif isinstance(value, dict):
            for name, v in value.iteritems():
                self.get_config(name).ddq = v
        else:
            for q,v in zip(self.kin_configs, value):
                q.ddq = v


    @property
    def u(self):
        """Current input vector of the system."""
        return np.array([u.u for u in self.inputs])

    @u.setter
    def u(self, value):
        # Writing u.u will clear system cache
        if isinstance(value, (int, float)):
            for u in self.inputs:
                u.u = value
        elif isinstance(value, dict):
            for name, v in value.iteritems():
                self.get_input(name).u = v
        else:
            for u,v in zip(self.inputs, value):
                u.u = v


    @property
    def world_frame(self):
        "The root spatial frame of the system."
        return self._world_frame

    @property
    def frames(self):
        "Tuple of all the frames in the system."
        return self._frames

    @property
    def configs(self):
        """
        Tuple of all the configuration variables in the system.

        This is always equal to self.dyn_configs + self.kin_configs
        """
        return self._configs

    @property
    def dyn_configs(self):
        """
        Tuple of all the dynamic configuration variables in the system.
        """
        return self._dyn_configs

    @property
    def kin_configs(self):
        """
        Tuple of all the kinematic configuration variables in the
        system.
        """
        return self._kin_configs

    @property
    def potentials(self):
        "Tuple of all the potentials in the system."
        return self._potentials

    @property
    def forces(self):
        "Tuple of all the forces in the system."
        return self._forces

    @property
    def inputs(self):
        "Tuple of all the input variables in the system."
        return self._inputs

    @property
    def constraints(self):
        "Tuple of all the constraints in the system."
        return self._constraints

    @property
    def masses(self):
        "Tuple of all the frames with non-zero inertias."
        return self._masses


    def _clear_cache(self):
        """Clear the system cache."""
        self._cache = 0
        self._state_counter += 1


    def _get_object(self, identifier, objtype, array):
        """
        _get_object(identifier, objtype, array) -> object,None

        Return the first item in array with a matching identifier.
        The type of 'identifier' defines how the object is identified.

        type(identifier) -> how identifier is used

        None -> return None
        int -> return array[identifier]
        name -> return item in array such that item.name == identifier
        objtype -> return identifier

        Raise an exception if 'identifier' is a different type or
        there is an error/no match.
        """
        if identifier == None:
            return None
        elif isinstance(identifier, objtype):
            return identifier
        elif isinstance(identifier, int):
            return array[identifier]
        elif isinstance(identifier, str):
            for item in array:
                if item.name == identifier:
                    return item
            raise KeyError("%s with name '%s' not found" % (objtype, identifier))
        else:
            raise TypeError()


    def _add_kin_config(self, config):
        """
        _add_kin_config(config) -> Append config to the kin_configs
            tuple.
        """
        assert isinstance(config, trep.Config)
        self._kin_configs += (config,)

    def _add_dyn_config(self, config):
        """
        _add_dyn_config(config) -> Append config to the dyn_configs
            tuple.
        """
        assert isinstance(config, trep.Config)
        self._dyn_configs += (config,)

    def _add_constraint(self, constraint):
        """
        _add_constraint(constraint) -> Append constraint to the
            constraint tuple.
        """
        assert isinstance(constraint, trep.Constraint)
        self._constraints += (constraint,)

    def _add_potential(self, potential):
        """
        _add_potential(potential) -> Append potential to the
            potentials tuple.
        """
        assert isinstance(potential, trep.Potential)
        self._potentials += (potential,)

    def _add_input(self, finput):
        """
        _add_input(finput) -> Append input to the inputs tuple.
        """
        assert isinstance(finput, trep.Input)
        self._inputs += (finput,)

    def _add_force(self, force):
        """
        _add_force(force) -> Append force to the forces tuple.
        """
        assert isinstance(force, trep.Force)
        self._forces += (force,)

    def add_structure_changed_func(self, function):
        """
        Register a function to call whenever the system structure
        changes.  This includes adding and removing frames,
        configuration variables, constraints, potentials, and forces.
        """
        self._structure_changed_funcs.append(function)

    def hold_structure_changes(self):
        """
        Prevent the system from calling System._update_structure()
        (mostly).  Useful when building a large system to avoid
        needlessly allocating and deallocating memory.
        """
        self._hold_structure_changes += 1

    def resume_structure_changes(self):
        """
        Stop preventing the system from calling
        System._update_structure().  The structure will only be
        updated once every hold has been removed, so calling this does
        not guarantee that the structure will be immediately upated.
        """
        if self._hold_structure_changes == 0:
            raise StandardError("System.resume_structure_changes() called" \
                                " when _hold_structure_changes is 0")
        self._hold_structure_changes -= 1
        if self._hold_structure_changes == 0:
            self._structure_changed()

    def _structure_changed(self):
        """
        Updates variables so that System is internally consistent.

        There is a lot of duplicate information throughout a System,
        for either convenience or performance reasons.  For duplicate
        information, one place is considered the 'master'.  These are
        places that other functions manipulate.  The other duplicates
        are created from the 'master'.

        The variables controlled by this function include:

        system.frames - This tuple is built by descending the frames
          tree and collecting each frame.

        system.configs - This tuple is built by concatenating
          system.dyn_configs and system.kin_configs.

        config.config_gen - config_gen is set by descending down the
          tree while keeping track of how many configuration variables
          have been seen.

        config.index - 'index' is set using the config's index in
          system.configs

        config.k_index - 'k_index' is set using the config's index in
          system.kin_configs or to -1 for dynamic configuration
          variables.

        system.masses - This tuple is set by running through
          system.frames and collecting any frame that has non-zero
          inertia properties.

        frame.cache_index - Built for each frame by descending up the
          tree and collecting every configuration variable that is
          encountered.  This is set in Frame._structure_changed()

        config.masses - Built for each config by looking at each frame
          in self.masses and collecting those that depend on the config.

        Finally, we call all the registered structure update functions
        for any external objects that need to update their own
        structures.
        """

        # When we build big systems, we waste a lot of time building
        # the cache over and over again.  Instead, we can turn off the
        # updating for a bit, and then do it once when we're
        # done.
        if self._hold_structure_changes != 0:
            return

        # Cache value dependencies:
        # system.frames :depends on: none
        # system.configs :depends on: none
        # config.config_gen :depends on: none
        # config.index :depend on: system.configs
        # system.masses :depends on: none
        # frame.cache_index :depends on: config.config_gen
        # config.masses :depends on: frame.cache_index, system.masses

        self._frames = tuple(self.world_frame.flatten_tree())

        self._configs = self.dyn_configs + self.kin_configs

        # Initialize config_gens to be N+1. Configs that do not drive
        # frame transformations will retain this value
        for config in self.configs:
            config._config_gen = len(self._configs)

        def update_config_gen(frame, index):
            if frame.config != None:
                frame.config._config_gen = index;
                index += 1
            for child in frame.children:
                update_config_gen(child, index)
        update_config_gen(self.world_frame, 0)

        for (i, config) in enumerate(self.configs):
            config._index = i
            config._k_index = -1
        for (i, config) in enumerate(self.kin_configs):
            config._k_index = i
        for (i, constraint) in enumerate(self.constraints):
            constraint._index = i
        for (i, finput) in enumerate(self.inputs):
            finput._index = i

        # Find all frames with non-zero masses
        self._masses = tuple([f for f in self.frames
                        if f.mass != 0.0
                        or f.Ixx != 0.0
                        or f.Iyy != 0.0
                        or f.Izz != 0.0])

        self.world_frame._structure_changed()

        for config in self.configs:
            config._masses = tuple([f for f in self._masses
                                    if config in f._cache_index])


        # Create numpy arrays used for calculation and storage
        self._f = np.zeros( (self.nQd,), np.double, 'C')
        self._lambda = np.zeros( (self.nc,), np.double, 'C')
        self._D = np.zeros( (self.nQd,), np.double, 'C')

        self._Ad = np.zeros((self.nc, self.nQd), np.double, 'C')
        self._AdT = np.zeros((self.nQd, self.nc), np.double, 'C')
        self._M_lu = np.zeros((self.nQd, self.nQd), np.double, 'C')
        self._M_lu_index = np.zeros((self.nQd,), np.int, 'C')
        self._A_proj_lu = np.zeros((self.nc, self.nc), np.double, 'C')
        self._A_proj_lu_index = np.zeros((self.nc, ), np.int, 'C')

        self._Ak = np.zeros( (self.nc, self.nQk), np.double, 'C')
        self._Adt = np.zeros( (self.nc, self.nQ), np.double, 'C')
        self._Ad_dq = np.zeros( (self.nQ, self.nc, self.nQd), np.double, 'C')
        self._Ak_dq = np.zeros( (self.nQ, self.nc, self.nQk), np.double, 'C')
        self._Adt_dq = np.zeros( (self.nQ, self.nc, self.nQ), np.double, 'C')
        self._D_dq = np.zeros( (self.nQ, self.nQd), np.double, 'C')
        self._D_ddq = np.zeros( (self.nQ, self.nQd), np.double, 'C')
        self._D_du = np.zeros( (self.nu, self.nQd), np.double, 'C')
        self._D_dk = np.zeros( (self.nQk, self.nQd), np.double, 'C')
        self._f_dq = np.zeros( (self.nQ, self.nQd), np.double, 'C')
        self._f_ddq = np.zeros( (self.nQ, self.nQd), np.double, 'C')
        self._f_du = np.zeros( (self.nu, self.nQd), np.double, 'C')
        self._f_dk = np.zeros( (self.nQk, self.nQd), np.double, 'C')
        self._lambda_dq = np.zeros( (self.nQ, self.nc), np.double, 'C')
        self._lambda_ddq = np.zeros( (self.nQ, self.nc), np.double, 'C')
        self._lambda_du = np.zeros( (self.nu, self.nc), np.double, 'C')
        self._lambda_dk = np.zeros( (self.nQk, self.nc), np.double, 'C')
        self._Ad_dqdq = np.zeros( (self.nQ, self.nQ, self.nc, self.nQd), np.double, 'C')
        self._Ak_dqdq = np.zeros( (self.nQ, self.nQ, self.nc, self.nQk), np.double, 'C')
        self._Adt_dqdq = np.zeros( (self.nQ, self.nQ, self.nc, self.nQ), np.double, 'C')

        self._D_dqdq = np.zeros( (self.nQ, self.nQ, self.nQd), np.double, 'C')
        self._D_ddqdq = np.zeros( (self.nQ, self.nQ, self.nQd), np.double, 'C')
        self._D_ddqddq = np.zeros( (self.nQ, self.nQ, self.nQd), np.double, 'C')
        self._D_dkdq = np.zeros( (self.nQk, self.nQ, self.nQd), np.double, 'C')
        self._D_dudq = np.zeros( (self.nu, self.nQ, self.nQd), np.double, 'C')
        self._D_duddq = np.zeros( (self.nu, self.nQ, self.nQd), np.double, 'C')
        self._D_dudu = np.zeros( (self.nu, self.nu, self.nQd), np.double, 'C')

        self._f_dqdq = np.zeros( (self.nQ, self.nQ, self.nQd), np.double, 'C')
        self._f_ddqdq = np.zeros( (self.nQ, self.nQ, self.nQd), np.double, 'C')
        self._f_ddqddq = np.zeros( (self.nQ, self.nQ, self.nQd), np.double, 'C')
        self._f_dkdq = np.zeros( (self.nQk, self.nQ, self.nQd), np.double, 'C')
        self._f_dudq = np.zeros( (self.nu, self.nQ, self.nQd), np.double, 'C')
        self._f_duddq = np.zeros( (self.nu, self.nQ, self.nQd), np.double, 'C')
        self._f_dudu = np.zeros( (self.nu, self.nu, self.nQd), np.double, 'C')

        self._lambda_dqdq = np.zeros( (self.nQ, self.nQ, self.nc), np.double, 'C')
        self._lambda_ddqdq = np.zeros( (self.nQ, self.nQ, self.nc), np.double, 'C')
        self._lambda_ddqddq = np.zeros( (self.nQ, self.nQ, self.nc), np.double, 'C')
        self._lambda_dkdq = np.zeros( (self.nQk, self.nQ, self.nc), np.double, 'C')
        self._lambda_dudq = np.zeros( (self.nu, self.nQ, self.nc), np.double, 'C')
        self._lambda_duddq = np.zeros( (self.nu, self.nQ, self.nc), np.double, 'C')
        self._lambda_dudu = np.zeros( (self.nu, self.nu, self.nc), np.double, 'C')

        self._temp_nd = np.zeros( (self.nQd,), np.double, 'C')
        self._temp_ndnc = np.zeros( (self.nQd, self.nc), np.double, 'C')

        self._M_dq = np.zeros( (self.nQ, self.nQ, self.nQ), np.double, 'C')
        self._M_dqdq = np.zeros( (self.nQ, self.nQ, self.nQ, self.nQ), np.double, 'C')

        self._clear_cache()

        for func in self._structure_changed_funcs:
            func()



    def total_energy(self):
        """Calculate the total energy in the current state."""
        return self._total_energy()

    def L(self):
        """Calculate the Lagrangian at the current state."""
        return self._L()

    def L_dq(self, q1):
        """
        Calculate the derivative of the Lagrangian with respect to the
        value of q1.
        """
        assert isinstance(q1, _trep._Config)
        return self._L_dq(q1)

    def L_dqdq(self, q1, q2):
        """
        Calculate the second derivative of the Lagrangian with respect
        to the value of q1 and the value of q2.
        """
        assert isinstance(q1, _trep._Config)
        assert isinstance(q2, _trep._Config)
        return self._L_dqdq(q1, q2)

    def L_dqdqdq(self, q1, q2, q3):
        """
        Calculate the third derivative of the Lagrangian with respect
        to the value of q1, the value of q2, and the value of q3.
        """
        assert isinstance(q1, _trep._Config)
        assert isinstance(q2, _trep._Config)
        assert isinstance(q3, _trep._Config)
        return self._L_dqdqdq(q1, q2, q3)

    def L_ddq(self, dq1):
        """
        Calculate the derivative of the Lagrangian with respect
        to the velocity of dq1.
        """
        assert isinstance(dq1, _trep._Config)
        return self._L_ddq(dq1)

    def L_ddqdq(self, dq1, q2):
        """
        Calculate the second derivative of the Lagrangian with respect
        to the velocity of dq1 and the value of q2.
        """
        assert isinstance(dq1, _trep._Config)
        assert isinstance(q2,  _trep._Config)
        return self._L_ddqdq(dq1, q2)

    def L_ddqdqdq(self, dq1, q2, q3):
        """
        Calculate the third derivative of the Lagrangian with respect
        to the velocity of dq1, the value of q2, and the value of q3.
        """
        assert isinstance(dq1, _trep._Config)
        assert isinstance(q2,  _trep._Config)
        assert isinstance(q3,  _trep._Config)
        return self._L_ddqdqdq(dq1, q2, q3)

    def L_ddqdqdqdq(self, dq1, q2, q3, q4):
        """
        Calculate the fourth derivative of the Lagrangian with respect
        to the velocity of dq1, the value of q2, the value of q3, and
        the value of q4.
        """
        assert isinstance(dq1, _trep._Config)
        assert isinstance(q2, _trep._Config)
        assert isinstance(q3, _trep._Config)
        assert isinstance(q4, _trep._Config)
        return self._L_ddqdqdqdq(dq1, q2, q3, q4)

    def L_ddqddq(self, dq1, dq2):
        """
        Calculate the second derivative of the Lagrangian with respect
        to the velocity of dq1 and the velocity of dq2.
        """
        assert isinstance(dq1, _trep._Config)
        assert isinstance(dq2, _trep._Config)
        return self._L_ddqddq(dq1, dq2)

    def L_ddqddqdq(self, dq1, dq2, q3):
        """
        Calculate the third derivative of the Lagrangian with respect
        to the velocity of dq1, the velocity of dq2, and the value of
        q3.
        """
        assert isinstance(dq1, _trep._Config)
        assert isinstance(dq2, _trep._Config)
        assert isinstance( q3, _trep._Config)
        return self._L_ddqddqdq(dq1, dq2, q3)

    def L_ddqddqdqdq(self, dq1, dq2, q3, q4):
        """
        Calculate the fourth derivative of the Lagrangian with respect
        to the velocity of dq1, the velocity of dq2, the value of q3,
        and the value of q4.
        """
        assert isinstance(dq1, _trep._Config)
        assert isinstance(dq2, _trep._Config)
        assert isinstance( q3, _trep._Config)
        assert isinstance( q4, _trep._Config)
        return self._L_ddqddqdqdq(dq1, dq2, q3, q4)


    @dynamics_indexing_decorator('d')
    def f(self, q=None):
        """
        Calculate the dynamics at the current state.

        See documentation for details.
        """
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS)
        return self._f[q].copy()

    @dynamics_indexing_decorator('dq')
    def f_dq(self, q=None, q1=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV1)
        return self._f_dq[q1, q].T.copy()

    @dynamics_indexing_decorator('dq')
    def f_ddq(self, q=None, dq1=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV1)
        return self._f_ddq[dq1, q].T.copy()

    @dynamics_indexing_decorator('dk')
    def f_dddk(self, q=None, k1=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV1)
        return self._f_dk[k1, q].T.copy()

    @dynamics_indexing_decorator('du')
    def f_du(self, q=None, u1=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV1)
        return self._f_du[u1, q].T.copy()

    @dynamics_indexing_decorator('dqq')
    def f_dqdq(self, q=None, q1=None, q2=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV2)
        return self._f_dqdq[q1, q2, q].copy()

    @dynamics_indexing_decorator('dqq')
    def f_ddqdq(self, q=None, dq1=None, q2=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV2)
        return self._f_ddqdq[dq1, q2, q].copy()

    @dynamics_indexing_decorator('dqq')
    def f_ddqddq(self, q=None, dq1=None, dq2=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV2)
        return self._f_ddqddq[dq1, dq2, q].copy()

    @dynamics_indexing_decorator('dkq')
    def f_dddkdq(self, q=None, k1=None, q2=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV2)
        return self._f_dkdq[k1, q2, q].copy()

    @dynamics_indexing_decorator('duq')
    def f_dudq(self, q=None, u1=None, q2=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV2)
        return self._f_dudq[u1, q2, q].copy()

    @dynamics_indexing_decorator('duq')
    def f_duddq(self, q=None, u1=None, dq2=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV2)
        return self._f_duddq[u1, dq2, q].copy()

    @dynamics_indexing_decorator('duu')
    def f_dudu(self, q=None, u1=None, u2=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV2)
        return self._f_dudu[u1, u2, q].copy()


    @dynamics_indexing_decorator('c')
    def lambda_(self, constraint=None):
        """
        Calculate the constraint forces at the current state.
        """
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS)
        return self._lambda[constraint].copy()

    @dynamics_indexing_decorator('cq')
    def lambda_dq(self, constraint=None, q1=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV1)
        return self._lambda_dq[q1, constraint].T.copy()

    @dynamics_indexing_decorator('cq')
    def lambda_ddq(self, constraint=None, dq1=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV1)
        return self._lambda_ddq[dq1, constraint].T.copy()

    @dynamics_indexing_decorator('ck')
    def lambda_dddk(self, constraint=None, k1=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV1)
        return self._lambda_dk[k1, constraint].T.copy()

    @dynamics_indexing_decorator('cu')
    def lambda_du(self, constraint=None, u1=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV1)
        return self._lambda_du[u1, constraint].T.copy()

    @dynamics_indexing_decorator('cqq')
    def lambda_dqdq(self, constraint=None, q1=None, q2=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV2)
        return self._lambda_dqdq[q1, q2, constraint].copy()

    @dynamics_indexing_decorator('cqq')
    def lambda_ddqdq(self, constraint=None, dq1=None, q2=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV2)
        return self._lambda_ddqdq[dq1, q2, constraint].copy()

    @dynamics_indexing_decorator('cqq')
    def lambda_ddqddq(self, constraint=None, dq1=None, dq2=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV2)
        return self._lambda_ddqddq[dq1, dq2, constraint].copy()

    @dynamics_indexing_decorator('ckq')
    def lambda_dddkdq(self, constraint=None, k1=None, q2=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV2)
        return self._lambda_dkdq[k1, q2, constraint].copy()

    @dynamics_indexing_decorator('cuq')
    def lambda_dudq(self, constraint=None, u1=None, q2=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV2)
        return self._lambda_dudq[u1, q2, constraint].copy()

    @dynamics_indexing_decorator('cuq')
    def lambda_duddq(self, constraint=None, u1=None, dq2=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV2)
        return self._lambda_duddq[u1, dq2, constraint].copy()

    @dynamics_indexing_decorator('cuu')
    def lambda_dudu(self, constraint=None, u1=None, u2=None):
        self._update_cache(_trep.SYSTEM_CACHE_DYNAMICS_DERIV2)
        return self._lambda_dudu[u1, u2, constraint].copy()

    def test_derivative_dq(self, func, func_dq, delta=1e-6, tolerance=1e-7,
                           verbose=False, test_name='<unnamed>'):
        """
        Test the derivative of a function with respect to a
        configuration variable value against its numerical
        approximation.

        func -> Callable taking no arguments and returning float or np.array

        func_dq -> Callable taking one configuration variable argument
                   and returning a float or np.array.

        delta -> perturbation to the current configuration to
                 calculate the numeric approximation.

        Returns stuff
        """
        q0 = self.q

        tests_total = 0
        tests_failed = 0

        for q in self.configs:
            self.q = q0
            dy_exact = func_dq(q)

            delta_q = q0.copy()
            delta_q[q.index] -= delta
            self.q = delta_q
            y0 = func()

            delta_q = q0.copy()
            delta_q[q.index] += delta
            self.q = delta_q
            y1 = func()

            dy_approx = (y1 - y0)/(2*delta)

            error = np.linalg.norm(dy_exact - dy_approx)
            tests_total += 1
            if math.isnan(error) or error > tolerance:
                tests_failed += 1
                if verbose:
                    print "Test '%s' failed for dq derivative of '%s'." % (test_name, q)
                    print "  Error: %f > %f" % (error, tolerance)
                    print "  Approx dy: %s" % dy_approx
                    print "   Exact dy: %s" % dy_exact

        if verbose:
            if tests_failed == 0:
                print "%d tests passing." % tests_total
            else:
                print "%d/%d tests FAILED.  <#######" % (tests_failed, tests_total)

        # Reset configuration
        self.q = q0
        return not tests_failed

    def test_derivative_ddq(self, func, func_ddq, delta=1e-6, tolerance=1e-7,
                            verbose=False, test_name='<unnamed>'):
        """
        Test the derivative of a function with respect to a
        configuration variable's time derivative and its numerical
        approximation.

        func -> Callable taking no arguments and returning float or np.array

        func_ddq -> Callable taking one configuration variable argument
                   and returning a float or np.array.

        delta -> perturbation to the current configuration to
                 calculate the numeric approximation.

        tolerance -> acceptable difference between the approximation
                     and exact value.  (|exact - approx| <= tolerance)

        verbose -> Boolean indicating if a message should be printed for failures.

        name -> String identifier to print out when reporting messages
                when verbose is true.

        Returns False if any tests fail and True otherwise.
        """
        dq0 = self.dq

        tests_total = 0
        tests_failed = 0

        for q in self.configs:
            self.dq = dq0
            dy_exact = func_ddq(q)

            delta_dq = dq0.copy()
            delta_dq[q.index] -= delta
            self.dq = delta_dq
            y0 = func()

            delta_dq = dq0.copy()
            delta_dq[q.index] += delta
            self.dq = delta_dq
            y1 = func()

            dy_approx = (y1 - y0)/(2*delta)

            error = np.linalg.norm(dy_exact - dy_approx)
            tests_total += 1
            if math.isnan(error) or error > tolerance:
                tests_failed += 1
                if verbose:
                    print "Test '%s' failed for dq derivative of '%s'." % (test_name, q)
                    print "  Error: %f > %f" % (error, tolerance)
                    print "  Approx dy: %f" % dy_approx
                    print "   Exact dy: %f" % dy_exact

        if verbose:
            if tests_failed == 0:
                print "%d tests passing." % tests_total
            else:
                print "%d/%d tests FAILED.  <#######" % (tests_failed, tests_total)

        # Reset velocity
        self.dq = dq0
        return not tests_failed


# Supressing a scipy.io.savemat warning.
import warnings
warnings.simplefilter("ignore", FutureWarning)

def save_trajectory(filename, system, t, Q=None, p=None, v=None, u=None, rho=None):
    # Save trajectory to a matlab file.  t is a 1D numpy array.
    # q,p,u,and rho are expected to be numpy arrays of the appropriate
    # dimensions or None

    t = np.array(t)

    data = { 'time' : np.array(t) }
    if Q is not None: data['Q'] = np.array(Q)
    if p is not None: data['p'] = np.array(p)
    if v is not None: data['v'] = np.array(v)
    if u is not None: data['u'] = np.array(u)
    if rho is not None: data['rho'] = np.array(rho)

    # Build indices - Convert to cells so they are well behaved in matlab
    data['Q_index'] = np.array([q.name for q in system.configs], dtype=np.object)
    data['p_index'] = np.array([q.name for q in system.dyn_configs], dtype=np.object)
    data['v_index'] = np.array([q.name for q in system.kin_configs], dtype=np.object)
    data['u_index'] = np.array([u.name for u in system.inputs], dtype=np.object)
    data['rho_index'] = np.array([q.name for q in system.kin_configs], dtype=np.object)

    sp.io.savemat(filename, data)


def load_trajectory(filename, system=None):

    data = sp.io.loadmat(filename)

    # Load time as a 1D array
    t = data['time'].squeeze()

    Q_in = data.get('Q', None)
    p_in = data.get('p', None)
    v_in = data.get('v', None)
    u_in = data.get('u', None)
    rho_in = data.get('rho', None)

    Q_index = [str(s[0]).strip() for s in data['Q_index'].ravel()]
    p_index = [str(s[0]).strip() for s in data['p_index'].ravel()]
    v_index = [str(s[0]).strip() for s in data['v_index'].ravel()]
    u_index = [str(s[0]).strip() for s in data['u_index'].ravel()]
    rho_index = [str(s[0]).strip() for s in data['rho_index'].ravel()]

    # If no system was given, just return the data as it was along
    # with the indices.
    if system is None:
        return (t,
                (Q_index, Q_in),
                (p_index, p_in),
                (v_index, v_in),
                (u_index, u_in),
                (rho_index, rho_in))
    else:
        # If a system was specified, reorganize the data to match the
        # system's layout.
        if Q_in is not None:
            Q = np.zeros((len(t), system.nQ))
            for config in system.configs:
                if config.name in Q_index:
                    Q[:,config.index] = Q_in[:, Q_index.index(config.name)]
        else:
            Q = None

        if p_in is not None:
            p = np.zeros((len(t), system.nQd))
            for config in system.dyn_configs:
                if config.name in p_index:
                    p[:,config.index] = p_in[:, p_index.index(config.name)]
        else:
            p = None

        if v_in is not None:
            v = np.zeros((len(t), system.nQk))
            for config in system.kin_configs:
                if config.name in v_index:
                    v[:,config.k_index] = v_in[:, v_index.index(config.name)]
        else:
            v = None

        if u_in is not None:
            u = np.zeros((len(t)-1, system.nu))
            for finput in system.inputs:
                if finput.name in u_index:
                    u[:,finput.index] = u_in[:, u_index.index(finput.name)]
        else:
            u = None

        if rho_in is not None:
            rho = np.zeros((len(t)-1, system.nQk))
            for config in system.kin_configs:
                if config.name in rho_index:
                    rho[:,config.k_index] = rho_in[:, rho_index.index(config.name)]
        else:
            rho = None

        return t,Q,p,v,u,rho
