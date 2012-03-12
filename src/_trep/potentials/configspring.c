#include <Python.h>
#include "structmember.h"
#include "trep.h"


typedef struct
{
    Potential potential; // Inherits from Potential
    Config *config;
    double k;
    double q0;
} ConfigSpringPotential;

static double V(ConfigSpringPotential *self)
{
    double diff;
    diff = self->config->q - self->q0;
    return 0.5 * self->k * diff * diff;
}

static double V_dq(ConfigSpringPotential *self, Config *q1)
{
    double diff;
    if(q1 == self->config) {
        diff = self->config->q - self->q0;
        return self->k * diff;
    } else {
        return 0.0;
    }
}

static double V_dqdq(ConfigSpringPotential *self, Config *q1, Config *q2)
{
    if(q1 == self->config && q2 == self->config)
        return self->k;
    else
        return 0.0;
}

static double V_dqdqdq(ConfigSpringPotential *self, Config *q1, Config *q2, Config *q3)
{
    return 0.0;
}


static void dealloc(ConfigSpringPotential *self)
{
    Py_CLEAR(self->config);
    ((PyObject*)self)->ob_type->tp_free((PyObject*)self);
}

static int init(ConfigSpringPotential *self, PyObject *args, PyObject *kwds)
{
    // Note that we do not called Potential.__init__ here.  It will
    // be called by Potential.__init__.
    self->k = 0.0;
    self->q0 = 0.0;

    /* Don't touch these though */
    self->potential.V = (PotentialFunc_V)&V;
    self->potential.V_dq = (PotentialFunc_V_dq)&V_dq;
    self->potential.V_dqdq = (PotentialFunc_V_dqdq)&V_dqdq;
    self->potential.V_dqdqdq = (PotentialFunc_V_dqdqdq)&V_dqdqdq;

    return 0;
}

/* Make custom parameters available to python here. */
static PyMemberDef members_list[] = {
    {"_config", T_OBJECT_EX, offsetof(ConfigSpringPotential, config), 0, trep_internal_doc},
    {"_q0", T_DOUBLE, offsetof(ConfigSpringPotential, q0), 0, trep_internal_doc},
    {"_k", T_DOUBLE, offsetof(ConfigSpringPotential, k), 0, trep_internal_doc},
    {NULL}  /* Sentinel */
};


extern PyTypeObject PotentialType;
PyTypeObject ConfigSpringPotentialType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_trep._ConfigSpringPotential", /*tp_name*/
    sizeof(ConfigSpringPotential),  /*tp_basicsize*/
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


