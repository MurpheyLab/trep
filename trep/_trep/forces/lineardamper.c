#include <Python.h>
#include "structmember.h"
#define TREP_MODULE
#include "../trep.h"

typedef struct
{
    Force force; // Inherits from Force
    double coefficient;
    TapeMeasure *path;
} LinearDamperForce;


static double f(LinearDamperForce *self, Config *q)
{
    double v, dx_dq = 0.0;

    if(!TapeMeasure_USES_CONFIG(self->path, q))
        return 0.0;

    v = TapeMeasure_velocity(self->path);
    dx_dq = TapeMeasure_length_dq(self->path, q);

    return -self->coefficient * v * dx_dq;
}

static double f_dq(LinearDamperForce *self, Config *q, Config *q1)
{
    double v, dv_dq1, dx_dq, dx_dqdq1;

    if(!TapeMeasure_USES_CONFIG(self->path, q))
        return 0.0;
    if(!TapeMeasure_USES_CONFIG(self->path, q1))
        return 0.0;

    v = TapeMeasure_velocity(self->path);
    dv_dq1 = TapeMeasure_velocity_dq(self->path, q1);
    dx_dq = TapeMeasure_length_dq(self->path, q);
    dx_dqdq1 = TapeMeasure_length_dqdq(self->path, q, q1);

    return -self->coefficient * (dv_dq1 * dx_dq + v * dx_dqdq1); 

}

static double f_ddq(LinearDamperForce *self, Config *q, Config *dq1)
{
    double dv_ddq1, dx_dq;

    if(!TapeMeasure_USES_CONFIG(self->path, q))
        return 0.0;
    if(!TapeMeasure_USES_CONFIG(self->path, dq1))
        return 0.0;

    dv_ddq1 = TapeMeasure_velocity_ddq(self->path, dq1);
    dx_dq = TapeMeasure_length_dq(self->path, q);

	return -self->coefficient * dv_ddq1 * dx_dq;
}

static double f_du(LinearDamperForce *self, Config *config, Input *u1) { return 0.0; }

static double f_dqdq(LinearDamperForce *self, Config *q, Config *q1, Config *q2)
{
    double v, dv_dq1, dv_dq2, dv_dq1dq2, dx_dq, dx_dqdq1, dx_dqdq2, dx_dqdq1dq2;

    if(!TapeMeasure_USES_CONFIG(self->path, q))
        return 0.0;
    if(!TapeMeasure_USES_CONFIG(self->path, q1))
        return 0.0;
    if(!TapeMeasure_USES_CONFIG(self->path, q2))
        return 0.0;

    v = TapeMeasure_velocity(self->path);
    dv_dq1 = TapeMeasure_velocity_dq(self->path, q1);
    dv_dq2 = TapeMeasure_velocity_dq(self->path, q2);
    dv_dq1dq2 = TapeMeasure_velocity_dqdq(self->path, q1, q2);
    dx_dq = TapeMeasure_length_dq(self->path, q);
    dx_dqdq1 = TapeMeasure_length_dqdq(self->path, q, q1);
    dx_dqdq2 = TapeMeasure_length_dqdq(self->path, q, q2);
    dx_dqdq1dq2 = TapeMeasure_length_dqdqdq(self->path, q, q1, q2);

    return -self->coefficient * (dv_dq1dq2*dx_dq+dv_dq1*dx_dqdq2+dv_dq2*dx_dqdq1+v*dx_dqdq1dq2); 
}

static double f_ddqdq(LinearDamperForce *self, Config *q, Config *dq1, Config *q2)
{
    double dv_ddq1, dv_ddq1dq2, dx_dq, dx_dqdq2;

    if(!TapeMeasure_USES_CONFIG(self->path, q))
        return 0.0;
    if(!TapeMeasure_USES_CONFIG(self->path, dq1))
        return 0.0;
    if(!TapeMeasure_USES_CONFIG(self->path, q2))
        return 0.0;

    dv_ddq1 = TapeMeasure_velocity_ddq(self->path, dq1);
    dv_ddq1dq2 = TapeMeasure_velocity_ddqdq(self->path, dq1, q2);
    dx_dq = TapeMeasure_length_dq(self->path, q);
    dx_dqdq2 = TapeMeasure_length_dq(self->path, q2);

    return -self->coefficient*(dv_ddq1dq2*dx_dq+dv_ddq1*dx_dqdq2);
}

static double f_ddqddq(LinearDamperForce *self, Config *q, Config *dq1, Config *dq2) { return 0.0; }
static double f_dudq(LinearDamperForce *self, Config *config, Input *u1, Config *q2) { return 0.0; }
static double f_duddq(LinearDamperForce *self, Config *config, Input *u1, Config *dq2) { return 0.0; }
static double f_dudu(LinearDamperForce *self, Config *config, Input *u1, Input *u2) { return 0.0; }

static void dealloc(LinearDamperForce *self)
{
    Py_CLEAR(self->path);
    ((PyObject*)self)->ob_type->tp_free((PyObject*)self);
}

static int init(LinearDamperForce *self, PyObject *args, PyObject *kwds)
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
    {"_coefficient", T_DOUBLE, offsetof(LinearDamperForce, coefficient), 0, trep_internal_doc},
    {"_path", T_OBJECT_EX, offsetof(LinearDamperForce, path), 0, "internal use only"},
    {NULL}  /* Sentinel */
};

extern PyTypeObject ForceType;
PyTypeObject LinearDamperForceType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_trep._LinearDamperForce",  /*tp_name*/
    sizeof(LinearDamperForce),   /*tp_basicsize*/
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

