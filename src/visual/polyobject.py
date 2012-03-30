import numpy as np
from OpenGL.GL import *


class _PyPolyObject(object):

    def __init__(self):
        self._vertices = None
        self._triangles = None
        self._normals = None
        self._compiled = False
        self._gl_list = 0


    def draw(self):
        if not self._compiled:
            self._compile()
        if self._gl_list:
            glCallList(self._gl_list)


    def _compile(self):

        if self._gl_list == 0:
            self._gl_list = glGenLists(1)

        # If we don't get a valid list, just go ahead and draw
        # anyways.  The program will run slower, but will still work
        # fine.
        if self._gl_list:
            # Using GL_COMPILE_AND_EXECUTE raises an invalid operation
            # error at glEndList() on some opengl implementations.
            # Don't know what's wrong...
            glNewList(self._gl_list, GL_COMPILE)
            
        glBegin(GL_TRIANGLES)
        for triangle in self._triangles:
            for index in triangle:            
                if self._normals is not None:                
                    glNormal3dv(self._normals[index])
                glVertex3dv(self._vertices[index])
        glEnd()

        if self._gl_list:
            glEndList()
            self._compiled = True

try:
    from _polyobject import _PolyObject
except:
    print "C _polyobject not found.  Using Python implementation instead."
    _PolyObject = _PyPolyObject


class PolyObject(_PolyObject):
    def __init__(self, vertices=None, normals=None, triangles=None):
        super(PolyObject, self).__init__()
        
        self._vertices = np.zeros((0,3), dtype=np.double)
        self._triangles = np.zeros((0,3), dtype=np.int)
        self._normals = None

        if vertices is not None:
            self.vertices = vertices
        if normals is not None:
            self.normals = normals
        if triangles is not None:
            self.triangles = triangles

    @property
    def vertices(self):
        return self._vertices.copy()

    @vertices.setter
    def vertices(self, value):
        value = np.array(value, np.double)
        if len(value.shape) != 2 or value.shape[1] != 3:
            raise ValueError("vertices must be a Nx3 array (shape is %s)" % (value.shape,))
        self._vertices = value
        self.check_indices()

    @property
    def normals(self):
        if self._normals is None:
            return None
        return self._normals.copy()
        

    @normals.setter
    def normals(self, value):
        if value is not None:
            value = np.array(value, np.double)
            if len(value.shape) != 2 or value.shape[1] != 3:
                raise ValueError("normals must be a Nx3 array (shape is %s)" % (value.shape,))
            self._normals = value
        else:
            self._normals = None
        self.check_indices()
                            
    @property
    def triangles(self):
        return self._triangles.copy()

    @triangles.setter
    def triangles(self, value):
        value = np.array(value, np.int32)
        if len(value.shape) != 2 or value.shape[1] != 3:
            raise ValueError("triangles must be a Nx3 array (shape is %s)" % (value.shape,))
        self._triangles = value
        self.check_indices()

    def check_indices(self):
        if len(self.triangles) == 0:
            return
        max_index = self._triangles.max()
        if max_index >= len(self._vertices):
            raise IndexError("Not enough vertices for these triangles (max_index=%d)" % max_index)
        if self._normals is not None:
            if max_index >= len(self._normals):
                raise IndexError("Not enough normals for these triangles (max_index=%d)" % max_index)

    def draw(self):
        """
        Draw the model in the current OpenGL context.
        """
        _PolyObject.draw(self)
