#include <Python.h>
#include "structmember.h"
#include "../trep_internal.h"


/* Replace 'Template' with your constraint name.
 * For example:  Template -> Gravity
 */

typedef struct
{
    Potential potential; // Inherits from Potential
    /* Custom variables go here */
    double parameter;
} TemplatePotential;

static double V(TemplatePotential *self)
{
    return 0.0;
}

static double V_dq(TemplatePotential *self, Config *q1)
{
    return 0.0;
}

static double V_dqdq(TemplatePotential *self, Config *q1, Config *q2)
{
    return 0.0;
}

static double V_dqdqdq(TemplatePotential *self, Config *q1, Config *q2, Config *q3)
{
    return 0.0
}


static void dealloc(TemplatePotential *self)
{
    ((PyObject*)self)->ob_type->tp_free((PyObject*)self);
}

static int init(TemplatePotential *self, PyObject *args, PyObject *kwds)
{
    // Note that we do not called Potential.__init__ here.  It will
    // be called by Potential.__init__.

    /* Initialize your custom parameters here */
    self->parameter = 0.0;

    /* Don't touch these though */
    self->potential.V = (PotentialFunc_V)&V;
    self->potential.V_dq = (PotentialFunc_V_dq)&V_dq;
    self->potential.V_dqdq = (PotentialFunc_V_dqdq)&V_dqdq;
    self->potential.V_dqdqdq = (PotentialFunc_V_dqdqdq)&V_dqdqdq;

    return 0;
}

/* Make custom parameters available to python here. */
static PyMemberDef members_list[] = {
    {"_parameter", T_DOUBLE, offsetof(TemplatePotential, parameter), 0, trep_internal_doc},
    {NULL}  /* Sentinel */
};


extern PyTypeObject PotentialType;
PyTypeObject TemplatePotentialType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_trep._TemplatePotential", /*tp_name*/
    sizeof(TemplatePotential),  /*tp_basicsize*/
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
    0,                         /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    0,                         /* tp_methods */
    members_list,              /* tp_members */
    0,                         /* tp_getset */
    &PotentialType,            /* tp_base */   
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)init,            /* tp_init */
    0,                         /* tp_alloc */
    0,                       /* tp_new */
};


