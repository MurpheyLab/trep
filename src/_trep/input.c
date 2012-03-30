#include <Python.h>
#include "structmember.h"
#define TREP_MODULE
#include "trep.h"

static void dealloc(Input *self)
{
    Py_CLEAR(self->system);
    Py_CLEAR(self->force);
    self->ob_type->tp_free((PyObject*)self);
}

static int init(Input *self, PyObject *args, PyObject *kwds)
{
    self->u = 0.0;
    self->index = -1;
    return 0;
}

static PyMemberDef members_list[] = {
    {"_system", T_OBJECT_EX, offsetof(Input, system), 0, trep_internal_doc},
    {"_index", T_INT, offsetof(Input, index), 0, trep_internal_doc},
    {"_force", T_OBJECT_EX, offsetof(Input, force), 0, trep_internal_doc},
    {"_u", T_DOUBLE, offsetof(Input, u), 0, trep_internal_doc},
    {NULL}  /* Sentinel */
};

PyTypeObject InputType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_trep._Input",            /*tp_name*/
    sizeof(Input),             /*tp_basicsize*/
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
    trep_internal_doc,         /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    0,                         /* tp_methods */
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
