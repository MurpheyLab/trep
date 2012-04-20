from OpenGL.GL import *
from polyobject import PolyObject
import numpy as np
import struct


class stlmodel(PolyObject):
    """
    The stlmodel class loads an STL model from a file and provides
    basic facilities for drawing the model in OpenGL.  
    """

    def __set_wireframe(self, value):
        self._wireframe = value
        self._compiled = False
    wireframe = property(lambda self : self._wireframe, __set_wireframe)

    def __set_color(self, value):
        self._color = value
        self._compiled = False
    color = property(lambda self : self._color, __set_color)

    def __set_scale(self, value):
        self._scale = value
        self.compiled = False
    scale = property(lambda self : self._scale, __set_scale)
    
    def __init__(self, filename, wireframe=False, normals=False, color=None, scale=(1.0, 1.0, 1.0)):
        """
        stlmodel(filename, wireframe=False, normals=False, color=None)

        Load an STL model.  The model can be drawn as a solid or
        wireframe, with or without normals, and in a specific color.
        color should be the desired color (list of three numbers
        between 0 and 1), or None to use the current color.
        """
        super(stlmodel, self).__init__()
        
        self._wireframe = wireframe
        self._compiled = False
        self._color = color
        self._scale = scale
        self.gl_list = 0
        
        # STL FIle Format:
	# STL File Format:
	# 80 byte -ASCII Header - Ignore
	# 4 byte  - unsigned long int - Number of Polygons	
	# 
	# For each polygon:
	# 4 byte - float - normal i
	# 4 byte - float - normal j
	# 4 byte - float - normal k
	#
	# 4 byte - float - vertex 1 x
	# 4 byte - float - vertex 1 y
	# 4 byte - float - vertex 1 z
	#
	# 4 byte - float - vertex 2 x
	# 4 byte - float - vertex 2 y
	# 4 byte - float - vertex 2 z
	#     
	# 4 byte - float - vertex 3 x
	# 4 byte - float - vertex 3 y
	# 4 byte - float - vertex 3 z
	#                    
	# 2 byte - unsigned int - attribute byte count

        # This is pretty slow, it could probably be sped up a lot.
        src = open(filename, 'rb')
        src.read(80)
    
        count = struct.unpack('<L', src.read(4))[0]

        vertices = np.zeros((count*3, 3), np.double)
        normals = np.zeros((count*3, 3), np.double)
        triangles = np.zeros((count, 3), np.int)

        for i in xrange(count):
            norm = struct.unpack('<fff', src.read(12))
            vert1 = np.array(struct.unpack('<fff', src.read(12)))
            vert2 = np.array(struct.unpack('<fff', src.read(12)))
            vert3 = np.array(struct.unpack('<fff', src.read(12)))
            src.read(2)

            # Blender doesn't export the normal vectors,
            # so we recompute them here.
            v = vert1 - vert2
            z = vert3 - vert2
            norm = np.cross(z, v)

            vertices[3*i+0,:] = vert1
            vertices[3*i+1,:] = vert2
            vertices[3*i+2,:] = vert3
            
            normals[3*i+0,:] = norm
            normals[3*i+1,:] = norm
            normals[3*i+2,:] = norm

            triangles[i,0] = 3*i+0
            triangles[i,1] = 3*i+1
            triangles[i,2] = 3*i+2

        self.vertices = vertices
        self.normals = normals
        self.triangles = triangles

        src.close()

    def draw(self):
        """
        Draw the model in the current OpenGL context.
        """
        glPushMatrix()
        glPushAttrib(GL_POLYGON_BIT | GL_CURRENT_BIT)
        if self._wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        if self._color:
            glColor3fv(self._color)
        glScalef(*self._scale)
        PolyObject.draw(self)
        glPopAttrib()
        glPopMatrix()
            

        
