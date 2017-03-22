#include <Python.h>
#include "structmember.h"
#define TREP_MODULE
#include "../trep.h"

typedef struct
{
    Force force; // Inherits from Force
    Frame *piston;
    Config *crank_angle;
    double offset;
    Spline *combustion_model;
    double magnitude;
} PistonExampleForce;


static double f(PistonExampleForce *self, Config *q)
{
    mat4x4 t1;
    vec6 vec;
    double angle = 0.0;
    double comb_force = 0.0;

    /* if(!Frame_USES_CONFIG(self->piston, q)) */
    /*     return 0.0; */

    mul_mm4(t1,
	    *Frame_g_inv(self->piston),
	    *Frame_g_dq(self->piston, q));
    unhat(vec, t1);

    angle = fmod(self->crank_angle->q - self->offset, 2*M_PI);
    if(angle < 0.0) angle += 2*M_PI;
    comb_force = self->magnitude * Spline_y(self->combustion_model, angle);

    return -comb_force * vec[2];
}

static double f_dq(PistonExampleForce *self, Config *q, Config *q1)
{
    mat4x4 t1, t2;
    vec6 vec;
    vec6 vec_dq1;
    double result = 0.0;
    double angle = 0.0;
    double comb_force = 0.0;

    /* if(!Frame_USES_CONFIG(self->piston, q)) */
    /*     return 0.0; */
    /* if(!Frame_USES_CONFIG(self->piston, q1) && */
    /*    (q1 != self->crank_angle)) */
    /*     return 0.0; */
    
    mul_mm4(t1,
	    *Frame_g_inv(self->piston),
	    *Frame_g_dq(self->piston, q));
    unhat(vec, t1);

    mul_mm4(t1,
	    *Frame_g_inv_dq(self->piston, q1),
	    *Frame_g_dq(self->piston, q));
    mul_mm4(t2,
	    *Frame_g_inv(self->piston),
	    *Frame_g_dqdq(self->piston, q, q1));
    add_mm4(t1, t1, t2);
    unhat(vec_dq1, t1);

    angle = fmod(self->crank_angle->q - self->offset, 2*M_PI);
    if(angle < 0.0) angle += 2*M_PI;
    comb_force = self->magnitude * Spline_y(self->combustion_model, angle);
    
    result = comb_force * vec_dq1[2];
    if(q1 == self->crank_angle) 
        result += self->magnitude * Spline_dy(self->combustion_model, angle) * vec[2];
    return -result;
}

static double f_ddq(PistonExampleForce *self, Config *config, Config *dq1) { return 0.0; }
static double f_du(PistonExampleForce *self, Config *q, Input *u1) { return 0.0; }



static void dealloc(PistonExampleForce *self)
{
    Py_CLEAR(self->piston);
    Py_CLEAR(self->crank_angle);
    Py_CLEAR(self->combustion_model);
    /* ((PyObject*)self)->ob_type->tp_free((PyObject*)self); */
	Py_TYPE(((PyObject*)self))->tp_free((PyObject*)self);
}

static int init(PistonExampleForce *self, PyObject *args, PyObject *kwds)
{
    // Note that we do not call Force.__init__ here.  It will
    // be called by Force.__init__.

    /* We don't implement higher derivatives, which means this force
       will be unusable for optimizations.
    */
    self->force.f = (ForceFunc_f)&f;
    self->force.f_dq = (ForceFunc_f_dq)&f_dq;
    self->force.f_ddq = (ForceFunc_f_ddq)&f_ddq;
    self->force.f_du = (ForceFunc_f_du)&f_du;
    /* self->force.f_dqdq = (ForceFunc_f_dqdq)&f_dqdq; */
    /* self->force.f_ddqdq = (ForceFunc_f_ddqdq)&f_ddqdq; */
    /* self->force.f_ddqddq = (ForceFunc_f_ddqddq)&f_ddqddq; */
    /* self->force.f_dudq = (ForceFunc_f_dudq)&f_dudq; */
    /* self->force.f_duddq = (ForceFunc_f_duddq)&f_duddq; */
    /* self->force.f_dudu = (ForceFunc_f_dudu)&f_dudu; */
    return 0;
}

static PyMemberDef members_list[] = {
    {"_piston", T_OBJECT_EX, offsetof(PistonExampleForce, piston), 0, trep_internal_doc}, 
    {"_crank_angle", T_OBJECT_EX, offsetof(PistonExampleForce, crank_angle), 0, trep_internal_doc},
    {"_offset", T_DOUBLE, offsetof(PistonExampleForce, offset), 0, trep_internal_doc},
    {"_combustion_model", T_OBJECT_EX, offsetof(PistonExampleForce, combustion_model), 0, trep_internal_doc},
    {"_magnitude", T_DOUBLE, offsetof(PistonExampleForce, magnitude), 0, trep_internal_doc},
    {NULL}  /* Sentinel */
};

extern PyTypeObject ForceType;
PyTypeObject PistonExampleForceType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    /* 0,                         /\*ob_size*\/ */
    "_trep._PistonExampleForce",  /*tp_name*/
    sizeof(PistonExampleForce),   /*tp_basicsize*/
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


