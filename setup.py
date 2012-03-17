#!/usr/bin/python

from distutils.core import setup
from distutils.extension import Extension
import numpy 

include_dirs = [
    'src/_trep',
    numpy.get_include()  
    ]
cflags=[]
ldflags=[]
define_macros=[]

# Fast indexing results in significant speed ups for second
# derivatives.  Turning it off will force fast index accesses to use
# the normal indexing functions where the array can be tested for the
# correct dimensions for debugging and development.
define_macros += [("TREP_FAST_INDEXING", None)]

_trep = Extension('trep._trep',
                  include_dirs=include_dirs,
                  define_macros=define_macros,
                  extra_compile_args=cflags,
                  extra_link_args=ldflags,
                  sources = [
                      'src/_trep/midpointvi.c',
                      'src/_trep/system.c',
                      'src/_trep/math-code.c',
                      'src/_trep/frame.c',
                      'src/_trep/_trep.c',
                      'src/_trep/config.c',
                      'src/_trep/potential.c',
                      'src/_trep/force.c',
                      'src/_trep/input.c',
                      'src/_trep/constraint.c',
                      'src/_trep/frametransform.c',
                      'src/_trep/spline.c',
                      'src/_trep/framesequence.c',
                      
                      # Constraints
                      'src/_trep/constraints/distance.c',
                      'src/_trep/constraints/point.c',
                      
                      # Potentials
                      'src/_trep/potentials/gravity.c',
                      'src/_trep/potentials/linearspring.c',
                      'src/_trep/potentials/configspring.c',
                      'src/_trep/potentials/nonlinear_config_spring.c',
                      
                      # Forces
                      'src/_trep/forces/damping.c',
                      'src/_trep/forces/jointforce.c',
                      'src/_trep/forces/bodywrench.c',
                      'src/_trep/forces/hybridwrench.c', 
                      'src/_trep/forces/spatialwrench.c',
                      ])

## _polyobject = Extension('_polyobject',
##                     extra_compile_args=[],
##                     extra_link_args=['-lGL'],
##                     include_dirs = ['/usr/local/include'],
##                     sources = ['src/newvisual/_polyobject.c'])

cmd_class = {}
cmd_options = {}


# Try to add support to build the documentation is Sphinx is
# installed.
try:
    from sphinx.setup_command import BuildDoc
    cmd_class['build_sphinx'] = BuildDoc
    # See docstring for BuildDoc on how to set default options here.
except ImportError:
    pass


setup (name = 'trep',
       version = '0.91',
       description = 'trep is used to simulate mechanical systems.',
       author = ['Elliot Johnson'],
       author_email = 'elliot.r.johnson@gmail.com',
       url = 'http://trep.sourceforge.net/',
       package_dir = {'' : 'src', 'trep': 'src'},
       packages=['trep',
                 'trep.constraints',
                 'trep.potentials',
                 'trep.forces',
                 'trep.visual',
                 'trep.puppets',
                 ],
       ext_modules = [_trep,
                      #_polyobject
                      ],
       cmdclass=cmd_class,
       command_options=cmd_options)
