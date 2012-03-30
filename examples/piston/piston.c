#include <Python.h>
#include "structmember.h"
#include <numpy/arrayobject.h>
#include <trep/trep.h>


typedef struct
{
    Force force; // Inherits from Force
    Frame *piston;
    Config *crank_angle;
    double offset;
    Spline *combustion_model;
    double magnitude;
} PistonForce;


static double f(PistonForce *self, Config *q)
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

static double f_dq(PistonForce *self, Config *q, Config *q1)
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

static double f_ddq(PistonForce *self, Config *config, Config *dq1) { return 0.0; }
static double f_du(PistonForce *self, Config *q, Input *u1) { return 0.0; }



static void dealloc(PistonForce *self)
{
    Py_CLEAR(self->piston);
    Py_CLEAR(self->crank_angle);
    Py_CLEAR(self->combustion_model);
    ((PyObject*)self)->ob_type->tp_free((PyObject*)self);
}

static int init(PistonForce *self, PyObject *args, PyObject *kwds)
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
    {"_piston", T_OBJECT_EX, offsetof(PistonForce, piston), 0, ""}, 
    {"_crank_angle", T_OBJECT_EX, offsetof(PistonForce, crank_angle), 0, ""},
    {"_offset", T_DOUBLE, offsetof(PistonForce, offset), 0, ""},
    {"_combustion_model", T_OBJECT_EX, offsetof(PistonForce, combustion_model), 0, ""},
    {"_magnitude", T_DOUBLE, offsetof(PistonForce, magnitude), 0, ""},
    {NULL}  /* Sentinel */
};

PyTypeObject PistonForceType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_piston._PistonForce",    /*tp_name*/
    sizeof(PistonForce),       /*tp_basicsize*/
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
    0,                         /* tp_base */   
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)init,            /* tp_init */
    0,                         /* tp_alloc */
    0,                       /* tp_new */
};



#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC __attribute__((visibility("default"))) init_piston(void) 
{
    PyObject* m;
    if(import_trep())
	return;
    import_array()

    m = Py_InitModule3("_piston", NULL, "C back-end for piston forces");
    if(m == NULL)
      return;

    PistonForceType.tp_new = PyType_GenericNew;
    PistonForceType.tp_base = ForceType;
    if (PyType_Ready(&PistonForceType) < 0)
        return;

    Py_INCREF(&PistonForceType);
    PyModule_AddObject(m, "_PistonForce", (PyObject *)(&PistonForceType));    
}

