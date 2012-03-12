from OpenGL.GL import *
import struct


class stlmodel:
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
        self._wireframe = wireframe
        self._compiled = False
        self._color = color
        self._scale = scale
        self.gl_list = 0
        
        self.triangles = []
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
        
        src = open(filename, 'rb')
        src.read(80)
    
        count = struct.unpack('<L', src.read(4))[0]

        for i in xrange(count):
            norm = struct.unpack('<fff', src.read(12))
            vert1 = struct.unpack('<fff', src.read(12))
            vert2 = struct.unpack('<fff', src.read(12))
            vert3 = struct.unpack('<fff', src.read(12))
            src.read(2)

            # Blender doesn't export the normal vectors,
            # so we recompute them here.
            v = (vert1[0]-vert2[0], vert1[1]-vert2[1], vert1[2]-vert2[2])
            z = (vert3[0]-vert2[0], vert3[1]-vert2[1], vert3[2]-vert2[2])
            norm = (-(v[1]*z[2]-v[2]*z[1]),
                    -(v[2]*z[0]-v[0]*z[2]),
                    -(v[0]*z[1] - v[1]*z[0]))
        
                      
            self.triangles.append((norm,
                                   vert1,
                                   vert2,
                                   vert3))
        src.close()

    def draw(self):
        """
        Draw the model in the current OpenGL context.
        """
        if not self._compiled:
            self._compile()
        if self.gl_list:
            glCallList(self.gl_list)
            
    def _compile(self):
        """
        Internal use only.

        Draw and compile the model into a display list for fast
        drawing later.
        """
        if self.gl_list == 0:
            self.gl_list = glGenLists(1)

        # If we don't get a valid list, just go ahead and draw
        # anyways.  The program will run slower, but will still work
        # fine.
        if self.gl_list:
            # Using GL_COMPILE_AND_EXECUTE raises an invalid operation
            # error at glEndList() on some opengl implementations.
            # Don't know what's wrong...
            glNewList(self.gl_list, GL_COMPILE)
            
        glPushAttrib(GL_POLYGON_BIT | GL_CURRENT_BIT)
        if self._wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        if self._color:
            glColor3fv(self._color)

        glBegin(GL_TRIANGLES)
        for tri in self.triangles:
            glNormal3fv(tri[0])
            glVertex3fv([xi*si for (xi,si) in zip(tri[1], self._scale)])
            glVertex3fv([xi*si for (xi,si) in zip(tri[2], self._scale)])
            glVertex3fv([xi*si for (xi,si) in zip(tri[3], self._scale)])
        glEnd()

        glPopAttrib()

        if self.gl_list:
            glEndList()
            self._compiled = True
            

        
