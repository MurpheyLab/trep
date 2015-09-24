#include <Python.h>
#include "structmember.h"
#define TREP_MODULE
#include "../trep.h"


typedef struct
{
    Potential potential; // Inherits from Potential
    Config *config;
    Spline *spline;
    double m;
    double b;
} NonlinearConfigSpring;

static double V(NonlinearConfigSpring *self)
{
    /* This should be the integral of the spline from -inf to to the
     * current config value.  The actual value of V isn't used for
     * simulation or optimization, so I'm not implementing this for
     * now.
     */
    return 0.0;
}

static double V_dq(NonlinearConfigSpring *self, Config *q1)
{
    double x;

    if(q1 == self->config) {
        x = self->m * self->config->q + self->b;
        return -Spline_y(self->spline, x);
    } else {
        return 0.0;
    }    
}

static double V_dqdq(NonlinearConfigSpring *self, Config *q1, Config *q2)
{
    double x;

    if(q1 == self->config && q2 == self->config) {
        x = self->m * self->config->q + self->b;
        return -Spline_dy(self->spline, x) * self->m;
    } else {
        return 0.0;
    }    
}

static double V_dqdqdq(NonlinearConfigSpring *self, Config *q1, Config *q2, Config *q3)
{
    double x;

    if(q1 == self->config && q2 == self->config && q3 == self->config) {
        x = self->m * self->config->q + self->b;
        return -Spline_ddy(self->spline, x) * -self->m * self->m;
    } else {
        return 0.0;
    }    
}


static void dealloc(NonlinearConfigSpring *self)
{
    Py_CLEAR(self->config);
    Py_CLEAR(self->spline);
    //((PyObject*)self)->ob_type->tp_free((PyObject*)self);
}

static int init(NonlinearConfigSpring *self, PyObject *args, PyObject *kwds)
{
    // Note that we do not call _Potential.__init__ here.  It will
    // be called by Potential.__init__.

    /* Don't touch these though */
    self->potential.V = (PotentialFunc_V)&V;
    self->potential.V_dq = (PotentialFunc_V_dq)&V_dq;
    self->potential.V_dqdq = (PotentialFunc_V_dqdq)&V_dqdq;
    self->potential.V_dqdqdq = (PotentialFunc_V_dqdqdq)&V_dqdqdq;

    return 0;
}

/* Make custom parameters available to python here. */
static PyMemberDef members_list[] = {
    {"_config", T_OBJECT_EX, offsetof(NonlinearConfigSpring, config), 0, trep_internal_doc},
    {"_spline", T_OBJECT_EX, offsetof(NonlinearConfigSpring, spline), 0, trep_internal_doc},
    {"_m", T_DOUBLE, offsetof(NonlinearConfigSpring, m), 0, trep_internal_doc},
    {"_b", T_DOUBLE, offsetof(NonlinearConfigSpring, b), 0, trep_internal_doc},
    {NULL}  /* Sentinel */
};


extern PyTypeObject PotentialType;
PyTypeObject NonlinearConfigSpringType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_trep._NonlinearConfigSpring", /*tp_name*/
    sizeof(NonlinearConfigSpring),  /*tp_basicsize*/
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


