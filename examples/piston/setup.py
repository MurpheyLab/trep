#!/usr/bin/python

import sys
from distutils.core import setup, Extension

_piston = Extension('_piston',
                    extra_compile_args=[],
                    extra_link_args=[],
                    sources = ['piston.c'])

setup (name = '_piston',
       ext_modules = [_piston])

