#include <Python.h>
#define TREP_MODULE
#include "trep.h"

extern PyTypeObject FrameTransformType;

PyDoc_STRVAR(FrameTransform_doc,
	     "FrameTransforms are static variables created by trep\n"
	     "to enumerate the frame transformation types.  The\n"
	     "available transformation types are: \n"
	     "  trep.WORLD - reserved for a system's spatial frame. \n"
	     "  trep.TX - Translation along parent's X axis.\n"
	     "  trep.TY - Translation along parent's Y axis.\n"
	     "  trep.TZ - Translation along parent's Z axis.\n"
	     "  trep.RX - Rotation about parent's X axis.\n"
	     "  trep.RY - Rotation about parent's Y axis.\n"
	     "  trep.RZ - Rotation about parent's Z axis.\n");
	     
FrameTransform *TREP_WORLD = NULL;
FrameTransform *TREP_TY = NULL;
FrameTransform *TREP_TX = NULL;
FrameTransform *TREP_TZ = NULL;
FrameTransform *TREP_RX = NULL;
FrameTransform *TREP_RY = NULL;
FrameTransform *TREP_RZ = NULL;
FrameTransform *TREP_CONST_SE3 = NULL;

static int
create_FrameTransform(PyObject *module, FrameTransform **dest, char *name)
{
    FrameTransform *transform = NULL;

    transform = (FrameTransform*)FrameTransformType.tp_alloc(
	&FrameTransformType, 0);
    if (transform == NULL)
	return 0;

    transform->name = PyBytes_FromString(name);
    if (transform->name == NULL)
    {
	Py_DECREF(transform);
	return 0;
    }

    PyModule_AddObject(module, name, (PyObject*)transform);
    *dest = transform;
    return 1;
}


static void
dealloc(FrameTransform *self)
{
    Py_DECREF(self->name);
    /* self->ob_type->tp_free((PyObject*)self); */
	Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyObject*
repr(FrameTransform *self)
{
    Py_INCREF(self->name);
    return self->name;
}


PyTypeObject FrameTransformType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    /* 0,                         /\*ob_size*\/ */
    "_trep.FrameTransform",    /*tp_name*/
    sizeof(FrameTransform),    /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)dealloc,       /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    (reprfunc)repr,            /*tp_repr*/
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
    FrameTransform_doc,        /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    0,                         /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    0,                         /* tp_new */
};


void initialize_transform_types(PyObject *module)
{
    create_FrameTransform(module, &TREP_WORLD, "WORLD");
    create_FrameTransform(module, &TREP_TX, "TX");
    create_FrameTransform(module, &TREP_TY, "TY");
    create_FrameTransform(module, &TREP_TZ, "TZ");
    create_FrameTransform(module, &TREP_RX, "RX");
    create_FrameTransform(module, &TREP_RY, "RY");
    create_FrameTransform(module, &TREP_RZ, "RZ");
    create_FrameTransform(module, &TREP_CONST_SE3, "CONST_SE3");
}
