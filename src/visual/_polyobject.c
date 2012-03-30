#include <Python.h>
#include "structmember.h"
#include <GL/gl.h>
#include <numpy/arrayobject.h>



typedef struct {
    PyObject_HEAD
    
    PyObject *vertices;  // numpy array  N x 3 double
    PyObject *normals;   // numpy array  N x 3 double  or NULL/None
    PyObject *triangles; // numpy array  M x 3 int
} PolyObject;


typedef int int3[3];
typedef double double3[3];

static void dealloc(PolyObject *self)
{
    Py_CLEAR(self->vertices);
    Py_CLEAR(self->normals);
    Py_CLEAR(self->triangles);
    self->ob_type->tp_free((PyObject*)self);
}

static int init(PolyObject *self, PyObject *args, PyObject *kwds)
{
    return 0;
}

static PyObject* draw(PolyObject *self)
{
    int3 *triangles = NULL;
    double3 *vertices = NULL;
    double3 *normals = NULL;
    int num_triangles = 0;
    int i,j;
    
    if(self->vertices == NULL || self->normals == NULL || self->triangles == NULL)
        Py_RETURN_NONE;

    num_triangles = PyArray_DIM(self->triangles, 0);
    triangles = (int3*)PyArray_DATA(self->triangles);
    vertices = (double3*)PyArray_DATA(self->vertices);
    
    if(self->normals != NULL && self->normals != Py_None) 
        normals = (double3*)PyArray_DATA(self->normals);
    

    glBegin(GL_TRIANGLES);
    for(i = 0; i < num_triangles; i++)
    {
        for(j = 0; j < 3; j++) {
            if(normals)
                glNormal3dv(normals[triangles[i][j]]);
            glVertex3dv(vertices[triangles[i][j]]);
        }
    }
    glEnd();       
    Py_RETURN_NONE;
}


static PyMethodDef methods_list[] = {
    {"draw", (PyCFunction)draw, METH_NOARGS, "internal use only"},
    {NULL}  /* Sentinel */
};

static PyMemberDef members_list[] = {
    {"_vertices", T_OBJECT_EX, offsetof(PolyObject, vertices), 0, "internal use only"},
    {"_normals", T_OBJECT_EX, offsetof(PolyObject, normals), 0, "internal use only"},
    {"_triangles", T_OBJECT_EX, offsetof(PolyObject, triangles), 0, "internal use only"},    
    {NULL}  /* Sentinel */
};

PyTypeObject PolyObjectType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_polyobject._PolyObject", /*tp_name*/
    sizeof(PolyObject),        /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)dealloc,       /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "internal use only",       /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    methods_list,              /* tp_methods */
    members_list,              /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)init,            /* tp_init */
    0,                         /* tp_alloc */
    0,                         /* tp_new */
};



#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC __attribute__((visibility("default"))) init_polyobject(void) 
{
    PyObject* m;

    import_array()

    m = Py_InitModule3("_polyobject", NULL, "C back-end for drawing a triangle lists in OpenGL.");
    if(m == NULL)
      return;
    
    PolyObjectType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&PolyObjectType) < 0)
        return;

    Py_INCREF(&PolyObjectType);
    PyModule_AddObject(m, "_PolyObject", (PyObject *)(&PolyObjectType));    
}
