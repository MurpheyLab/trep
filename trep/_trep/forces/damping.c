#include <Python.h>
#include "structmember.h"
#define TREP_MODULE
#include "../trep.h"

typedef struct
{
    Force force; // Inherits from Force
    PyArrayObject *coefficients;
} DampingForce;


static double f(DampingForce *self, Config *config)
{
    return -IDX1_DBL(self->coefficients, config->index)*config->dq;
}

static double f_dq(DampingForce *self, Config *q, Config *q1) { return 0.0; }

static double f_ddq(DampingForce *self, Config *q, Config *dq1)
{
    if(q == dq1)
        return -IDX1_DBL(self->coefficients, q->index);
    else
	return 0.0;
}

static double f_du(DampingForce *self, Config *config, Input *u1) { return 0.0; }
static double f_dqdq(DampingForce *self, Config *q, Config *q1, Config *q2) { return 0.0; }
static double f_ddqdq(DampingForce *self, Config *q, Config *dq1, Config *q2) { return 0.0; }
static double f_ddqddq(DampingForce *self, Config *q, Config *dq1, Config *dq2) { return 0.0; }
static double f_dudq(DampingForce *self, Config *config, Input *u1, Config *q2) { return 0.0; }
static double f_duddq(DampingForce *self, Config *config, Input *u1, Config *dq2) { return 0.0; }
static double f_dudu(DampingForce *self, Config *config, Input *u1, Input *u2) { return 0.0; }

static void dealloc(DampingForce *self)
{
    Py_CLEAR(self->coefficients);
    //((PyObject*)self)->ob_type->tp_free((PyObject*)self);
}

static int init(DampingForce *self, PyObject *args, PyObject *kwds)
{
    // Note that we do not called Force.__init__ here.  It will
    // be called by Force.__init__.

    self->force.f = (ForceFunc_f)&f;
    self->force.f_dq = (ForceFunc_f_dq)&f_dq;
    self->force.f_dqdq = (ForceFunc_f_dqdq)&f_dqdq;
    self->force.f_ddq = (ForceFunc_f_ddq)&f_ddq;
    self->force.f_ddqdq = (ForceFunc_f_ddqdq)&f_ddqdq;
    self->force.f_ddqddq = (ForceFunc_f_ddqddq)&f_ddqddq;
    self->force.f_du = (ForceFunc_f_du)&f_du;
    self->force.f_dudq = (ForceFunc_f_dudq)&f_dudq;
    self->force.f_duddq = (ForceFunc_f_duddq)&f_duddq;
    self->force.f_dudu = (ForceFunc_f_dudu)&f_dudu;
    return 0;
}

static PyMemberDef members_list[] = {
    {"_coefficients", T_OBJECT_EX, offsetof(DampingForce, coefficients), 0, trep_internal_doc},
    {NULL}  /* Sentinel */
};

extern PyTypeObject ForceType;
PyTypeObject DampingForceType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_trep._DampingForce",  /*tp_name*/
    sizeof(DampingForce),   /*tp_basicsize*/
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
    &ForceType,                /* tp_base */   
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)init,            /* tp_init */
    0,                         /* tp_alloc */
    0,                       /* tp_new */
};

