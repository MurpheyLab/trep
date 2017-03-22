#include <Python.h>
#include "structmember.h"
#define TREP_MODULE
#include "../trep.h"


typedef struct
{
    Force force; // Inherits from Force
    Input *wrench_var[6];  // Variable elements of the force
    double wrench_con[6];  // Constant elements of the force
    Frame *frame;
} SpatialWrenchForce;

#define ISNONE(obj) ((PyObject*)(obj) == Py_None)

static double f(SpatialWrenchForce *self, Config *q)
{
    int i;
    mat4x4 t1;
    vec6 vec;
    double result = 0.0;

    if(!Frame_USES_CONFIG(self->frame, q))
	return 0.0;

    mul_mm4(t1,
	    *Frame_g_dq(self->frame, q),
	    *Frame_g_inv(self->frame));
    unhat(vec, t1);

    result = 0.0;
    for(i = 0; i < 6; i++) {
	if(!ISNONE(self->wrench_var[i]))
	    result += vec[i]*self->wrench_var[i]->u;
	else
	    result += vec[i]*self->wrench_con[i];
    }	    
    return result;
}

static double f_dq(SpatialWrenchForce *self, Config *q, Config *q1)
{
    int i;
    mat4x4 t1, t2;
    vec6 vec;
    double result = 0.0;

    if(!Frame_USES_CONFIG(self->frame, q) ||
       !Frame_USES_CONFIG(self->frame, q1))
	return 0.0;

    mul_mm4(t1,
	    *Frame_g_dq(self->frame, q),
	    *Frame_g_inv_dq(self->frame, q1));
    mul_mm4(t2,
            *Frame_g_dqdq(self->frame, q, q1),            
	    *Frame_g_inv(self->frame));
    add_mm4(t1, t1, t2);
    unhat(vec, t1);

    result = 0.0;
    for(i = 0; i < 6; i++) {
	if(!ISNONE(self->wrench_var[i]))
	    result += vec[i]*self->wrench_var[i]->u;
	else
	    result += vec[i]*self->wrench_con[i];
    }	    
    return result;
}

static double f_dqdq(SpatialWrenchForce *self, Config *q, Config *q1, Config *q2)
{
    int i;
    mat4x4 t1, t2;
    vec6 vec;
    double result = 0.0;

    if(!Frame_USES_CONFIG(self->frame, q) ||
       !Frame_USES_CONFIG(self->frame, q1) ||
       !Frame_USES_CONFIG(self->frame, q2))
	return 0.0;

    mul_mm4(t1,
	    *Frame_g_dq(self->frame, q),
	    *Frame_g_inv_dqdq(self->frame, q1, q2));
    mul_mm4(t2,
	    *Frame_g_dqdq(self->frame, q, q2),
	    *Frame_g_inv_dq(self->frame, q1));
    add_mm4(t1, t1, t2);
    mul_mm4(t2,
	    *Frame_g_dqdq(self->frame, q, q1),
	    *Frame_g_inv_dq(self->frame, q2));
    add_mm4(t1, t1, t2);
    mul_mm4(t2,
	    *Frame_g_dqdqdq(self->frame, q, q1, q2),
	    *Frame_g_inv(self->frame));
    add_mm4(t1, t1, t2);
    unhat(vec, t1);

    result = 0.0;
    for(i = 0; i < 6; i++) {
	if(!ISNONE(self->wrench_var[i]))
	    result += vec[i]*self->wrench_var[i]->u;
	else
	    result += vec[i]*self->wrench_con[i];
    }	    
    return result;
}

static double f_ddq(SpatialWrenchForce *self, Config *config, Config *dq1) { return 0.0; }
static double f_ddqdq(SpatialWrenchForce *self, Config *config, Config *dq1, Config *q2) { return 0.0; }
static double f_ddqddq(SpatialWrenchForce *self, Config *config, Config *dq1, Config *dq2) { return 0.0; }

static double f_du(SpatialWrenchForce *self, Config *q, Input *u1)
{
    int i;
    mat4x4 t1;
    vec6 vec;
    double result = 0.0;

    if(!Frame_USES_CONFIG(self->frame, q))
	return 0.0;

    /*
    if(!(self->wrench_var[0] == u1 ||
	 self->wrench_var[1] == u1 ||
	 self->wrench_var[2] == u1 ||
	 self->wrench_var[3] == u1 ||
	 self->wrench_var[4] == u1 ||
	 self->wrench_var[5] == u1))
	return 0.0;
    */

    mul_mm4(t1,
	    *Frame_g_dq(self->frame, q),
	    *Frame_g_inv(self->frame));
    unhat(vec, t1);

    result = 0.0;
    for(i = 0; i < 6; i++) {
	if(self->wrench_var[i] == u1)
	    result += vec[i];
    }	    
    return result;
}

static double f_dudq(SpatialWrenchForce *self, Config *q, Input *u1, Config *q2)
{
    int i;
    mat4x4 t1, t2;
    vec6 vec;
    double result = 0.0;

    if(!Frame_USES_CONFIG(self->frame, q) ||
       !Frame_USES_CONFIG(self->frame, q2))
	return 0.0;
    /*
    if(!(self->wrench_var[0] == u1 ||
	 self->wrench_var[1] == u1 ||
	 self->wrench_var[2] == u1 ||
	 self->wrench_var[3] == u1 ||
	 self->wrench_var[4] == u1 ||
	 self->wrench_var[5] == u1))
	return 0.0;
    */

    mul_mm4(t1,
	    *Frame_g_dq(self->frame, q),
	    *Frame_g_inv_dq(self->frame, q2));
    mul_mm4(t2,
	    *Frame_g_dqdq(self->frame, q, q2),
	    *Frame_g_inv(self->frame));
    add_mm4(t1, t1, t2);
    unhat(vec, t1);

    result = 0.0;
    for(i = 0; i < 6; i++) {
	if(self->wrench_var[i] == u1)
	    result += vec[i];
    }	    
    return result;
}

static double f_duddq(SpatialWrenchForce *self, Config *q, Input *u1, Config *dq2) { return 0.0; }
static double f_dudu(SpatialWrenchForce *self, Config *q, Input *u1, Input *u2) { return 0.0; }

static void dealloc(SpatialWrenchForce *self)
{
    int i;
    Py_CLEAR(self->frame);
    for(i = 0; i < 6; i++) 
        Py_CLEAR(self->wrench_var[i]);
    /* ((PyObject*)self)->ob_type->tp_free((PyObject*)self); */
	Py_TYPE(((PyObject*)self))->tp_free((PyObject*)self);
}

static int init(SpatialWrenchForce *self, PyObject *args, PyObject *kwds)
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
    {"_frame", T_OBJECT_EX, offsetof(SpatialWrenchForce, frame), 0, trep_internal_doc},
    {"_wrench_var0", T_OBJECT_EX, offsetof(SpatialWrenchForce, wrench_var[0]), 0, trep_internal_doc},
    {"_wrench_var1", T_OBJECT_EX, offsetof(SpatialWrenchForce, wrench_var[1]), 0, trep_internal_doc},
    {"_wrench_var2", T_OBJECT_EX, offsetof(SpatialWrenchForce, wrench_var[2]), 0, trep_internal_doc},
    {"_wrench_var3", T_OBJECT_EX, offsetof(SpatialWrenchForce, wrench_var[3]), 0, trep_internal_doc},
    {"_wrench_var4", T_OBJECT_EX, offsetof(SpatialWrenchForce, wrench_var[4]), 0, trep_internal_doc},
    {"_wrench_var5", T_OBJECT_EX, offsetof(SpatialWrenchForce, wrench_var[5]), 0, trep_internal_doc},
    {"_wrench_con0", T_DOUBLE, offsetof(SpatialWrenchForce, wrench_con[0]), 0, trep_internal_doc},
    {"_wrench_con1", T_DOUBLE, offsetof(SpatialWrenchForce, wrench_con[1]), 0, trep_internal_doc},
    {"_wrench_con2", T_DOUBLE, offsetof(SpatialWrenchForce, wrench_con[2]), 0, trep_internal_doc},
    {"_wrench_con3", T_DOUBLE, offsetof(SpatialWrenchForce, wrench_con[3]), 0, trep_internal_doc},
    {"_wrench_con4", T_DOUBLE, offsetof(SpatialWrenchForce, wrench_con[4]), 0, trep_internal_doc},
    {"_wrench_con5", T_DOUBLE, offsetof(SpatialWrenchForce, wrench_con[5]), 0, trep_internal_doc},
    {NULL}  /* Sentinel */
};

extern PyTypeObject ForceType;
PyTypeObject SpatialWrenchForceType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    /* 0,                         /\*ob_size*\/ */
    "_trep._SpatialWrenchForce",  /*tp_name*/
    sizeof(SpatialWrenchForce),   /*tp_basicsize*/
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
