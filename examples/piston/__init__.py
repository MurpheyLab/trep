
from pypiston import PyPistonForce

PistonForce = PyPistonForce

try:
    from cpiston import CPistonForce
    PistonForce = CPistonForce
except ImportError:
    print """
You must compile the piston extension to use the C implementation of
the piston force:

    cd piston
    python setup.py build_ext --inplace

Defaulting to Python implementation.
"""
    PistonForce = PyPistonForce

