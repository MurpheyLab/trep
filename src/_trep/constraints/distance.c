#include <Python.h>
#include "structmember.h"
#define TREP_MODULE
#include "../trep.h"


typedef struct
{
    Constraint constraint;  // Inherits from Constraint
    Frame *frame1;
    Frame *frame2;
    Config *config;
    double distance;
} DistanceConstraint;

static double h(DistanceConstraint *self)
{
    vec4 v;
    double h = 0.0;

    sub_vec4(v,
	     *Frame_p(self->frame1),
	     *Frame_p(self->frame2));

    h = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    
    if((PyObject*)self->config == Py_None)
	h -= self->distance * self->distance;
    else
	h -= self->config->q * self->config->q;

    return h;
}

static double h_dq(DistanceConstraint *self, Config *q1)
{
    vec4 v, v_d1;
    double h = 0.0;

    if(!Frame_USES_CONFIG(self->frame1, q1) &&
       !Frame_USES_CONFIG(self->frame2, q1) &&
       self->config != q1)
	return 0.0;

    sub_vec4(v,
	     *Frame_p(self->frame1),
	     *Frame_p(self->frame2));
    sub_vec4(v_d1,
	     *Frame_p_dq(self->frame1, q1),
	     *Frame_p_dq(self->frame2, q1));

    h = v[0] * v_d1[0] +
	v[1] * v_d1[1] +
	v[2] * v_d1[2];
    
    if(self->config == q1)
	h -= self->config->q;

    h *= 2.0;

    return h;
}

static double h_dqdq(DistanceConstraint *self, Config *q1, Config *q2)
{
    vec4 v;
    vec4 v_d1;
    vec4 v_d2;
    vec4 v_d12;
    double h;

    if(!Frame_USES_CONFIG(self->frame1, q1) &&
       !Frame_USES_CONFIG(self->frame2, q1) &&
       self->config != q1)
	return 0.0;
    if(!Frame_USES_CONFIG(self->frame1, q2) &&
       !Frame_USES_CONFIG(self->frame2, q2) &&
       self->config != q2)
	return 0.0;

    sub_vec4(v, *Frame_p(self->frame1), *Frame_p(self->frame2));
    sub_vec4(v_d1, *Frame_p_dq(self->frame1, q1), *Frame_p_dq(self->frame2, q1));
    sub_vec4(v_d2, *Frame_p_dq(self->frame1, q2), *Frame_p_dq(self->frame2, q2));
    sub_vec4(v_d12, *Frame_p_dqdq(self->frame1, q1, q2), *Frame_p_dqdq(self->frame2, q1, q2));

    h = v_d1[0]*v_d2[0] + v_d1[1]*v_d2[1] + v_d1[2]*v_d2[2] +
	v[0]*v_d12[0] + v[1]*v_d12[1] + v[2]*v_d12[2];
    
    if(self->config == q1 && self->config == q2)
	h -= 1;

    h *= 2.0;

    return h;
}

static double h_dqdqdq(DistanceConstraint *self, Config *q1, Config *q2, Config *q3)
{
    vec4 v;
    vec4 v_d1;
    vec4 v_d2;
    vec4 v_d3;
    vec4 v_d12;
    vec4 v_d13;
    vec4 v_d23;
    vec4 v_d123;
    double h;

    if(!Frame_USES_CONFIG(self->frame1, q1) && !Frame_USES_CONFIG(self->frame2, q1))
	return 0.0;
    if(!Frame_USES_CONFIG(self->frame1, q2) && !Frame_USES_CONFIG(self->frame2, q2))
	return 0.0;
    if(!Frame_USES_CONFIG(self->frame1, q3) && !Frame_USES_CONFIG(self->frame2, q3))
	return 0.0;

    sub_vec4(v, *Frame_p(self->frame1), *Frame_p(self->frame2));
    sub_vec4(v_d1, *Frame_p_dq(self->frame1, q1), *Frame_p_dq(self->frame2, q1));
    sub_vec4(v_d2, *Frame_p_dq(self->frame1, q2), *Frame_p_dq(self->frame2, q2));
    sub_vec4(v_d3, *Frame_p_dq(self->frame1, q3), *Frame_p_dq(self->frame2, q3));
    sub_vec4(v_d12, *Frame_p_dqdq(self->frame1, q1, q2), *Frame_p_dqdq(self->frame2, q1, q2));
    sub_vec4(v_d13, *Frame_p_dqdq(self->frame1, q1, q3), *Frame_p_dqdq(self->frame2, q1, q3));
    sub_vec4(v_d23, *Frame_p_dqdq(self->frame1, q2, q3), *Frame_p_dqdq(self->frame2, q2, q3));
    sub_vec4(v_d123, *Frame_p_dqdqdq(self->frame1, q1, q2, q3), *Frame_p_dqdqdq(self->frame2, q1, q2, q3));

    h = v_d1[0]*v_d23[0] + v_d1[1]*v_d23[1] + v_d1[2]*v_d23[2] +
	v_d2[0]*v_d13[0] + v_d2[1]*v_d13[1] + v_d2[2]*v_d13[2] +
	v_d3[0]*v_d12[0] + v_d3[1]*v_d12[1] + v_d3[2]*v_d12[2] +
	v[0]*v_d123[0] + v[1]*v_d123[1] + v[2]*v_d123[2];
    
    h *= 2.0;

    return h;
}

static double h_dqdqdqdq(DistanceConstraint *self, Config *q1, Config *q2, Config *q3, Config *q4)
{
    vec4 v;
    vec4 v_d1;
    vec4 v_d2;
    vec4 v_d3;
    vec4 v_d4;
    vec4 v_d12;
    vec4 v_d13;
    vec4 v_d14;
    vec4 v_d23;
    vec4 v_d24;
    vec4 v_d34;
    vec4 v_d123;
    vec4 v_d124;
    vec4 v_d134;
    vec4 v_d234;
    vec4 v_d1234;
    
    double h;

    if(!Frame_USES_CONFIG(self->frame1, q1) && !Frame_USES_CONFIG(self->frame2, q1))
	return 0.0;
    if(!Frame_USES_CONFIG(self->frame1, q2) && !Frame_USES_CONFIG(self->frame2, q2))
	return 0.0;
    if(!Frame_USES_CONFIG(self->frame1, q3) && !Frame_USES_CONFIG(self->frame2, q3))
	return 0.0;
    if(!Frame_USES_CONFIG(self->frame1, q4) && !Frame_USES_CONFIG(self->frame2, q4))
	return 0.0;

    sub_vec4(v, *Frame_p(self->frame1), *Frame_p(self->frame2));
    sub_vec4(v_d1, *Frame_p_dq(self->frame1, q1), *Frame_p_dq(self->frame2, q1));
    sub_vec4(v_d2, *Frame_p_dq(self->frame1, q2), *Frame_p_dq(self->frame2, q2));
    sub_vec4(v_d3, *Frame_p_dq(self->frame1, q3), *Frame_p_dq(self->frame2, q3));
    sub_vec4(v_d4, *Frame_p_dq(self->frame1, q4), *Frame_p_dq(self->frame2, q4));
    sub_vec4(v_d12, *Frame_p_dqdq(self->frame1, q1, q2), *Frame_p_dqdq(self->frame2, q1, q2));
    sub_vec4(v_d13, *Frame_p_dqdq(self->frame1, q1, q3), *Frame_p_dqdq(self->frame2, q1, q3));
    sub_vec4(v_d14, *Frame_p_dqdq(self->frame1, q1, q4), *Frame_p_dqdq(self->frame2, q1, q4));
    sub_vec4(v_d23, *Frame_p_dqdq(self->frame1, q2, q3), *Frame_p_dqdq(self->frame2, q2, q3));
    sub_vec4(v_d24, *Frame_p_dqdq(self->frame1, q2, q4), *Frame_p_dqdq(self->frame2, q2, q4));
    sub_vec4(v_d34, *Frame_p_dqdq(self->frame1, q3, q4), *Frame_p_dqdq(self->frame2, q3, q4));
    sub_vec4(v_d123, *Frame_p_dqdqdq(self->frame1, q1, q2, q3), *Frame_p_dqdqdq(self->frame2, q1, q2, q3));
    sub_vec4(v_d124, *Frame_p_dqdqdq(self->frame1, q1, q2, q4), *Frame_p_dqdqdq(self->frame2, q1, q2, q4));
    sub_vec4(v_d134, *Frame_p_dqdqdq(self->frame1, q1, q3, q4), *Frame_p_dqdqdq(self->frame2, q1, q3, q4));
    sub_vec4(v_d234, *Frame_p_dqdqdq(self->frame1, q2, q3, q4), *Frame_p_dqdqdq(self->frame2, q2, q3, q4));
    sub_vec4(v_d1234, *Frame_p_dqdqdqdq(self->frame1, q1, q2, q3, q4),
             *Frame_p_dqdqdqdq(self->frame2, q1, q2, q3, q4));

    h = v_d1[0]*v_d234[0] + v_d1[1]*v_d234[1] + v_d1[2]*v_d234[2] +
	v_d2[0]*v_d134[0] + v_d2[1]*v_d134[1] + v_d2[2]*v_d134[2] +
	v_d3[0]*v_d124[0] + v_d3[1]*v_d124[1] + v_d3[2]*v_d124[2] +
	v_d4[0]*v_d123[0] + v_d4[1]*v_d123[1] + v_d4[2]*v_d123[2] +
	v_d14[0]*v_d23[0] + v_d14[1]*v_d23[1] + v_d14[2]*v_d23[2] +
	v_d24[0]*v_d13[0] + v_d24[1]*v_d13[1] + v_d24[2]*v_d13[2] +
	v_d34[0]*v_d12[0] + v_d34[1]*v_d12[1] + v_d34[2]*v_d12[2] +
	v[0]*v_d1234[0] + v[1]*v_d1234[1] + v[2]*v_d1234[2];
    
    h *= 2.0;

    return h;
}

static void dealloc(DistanceConstraint *self)
{
    Py_CLEAR(self->frame1);
    Py_CLEAR(self->frame2);
    Py_CLEAR(self->config);
    ((PyObject*)self)->ob_type->tp_free((PyObject*)self);
}

static int init(DistanceConstraint *self, PyObject *args, PyObject *kwds)
{
    // Note that we do not called Constraint.__init__ here.  It will
    // be called by Constraint.__init__.
    self->distance = 0.0;
    
    self->constraint.h = (ConstraintFunc_h)&h;
    self->constraint.h_dq = (ConstraintFunc_h_dq)&h_dq;
    self->constraint.h_dqdq = (ConstraintFunc_h_dqdq)&h_dqdq;
    self->constraint.h_dqdqdq = (ConstraintFunc_h_dqdqdq)&h_dqdqdq;
    self->constraint.h_dqdqdqdq = (ConstraintFunc_h_dqdqdqdq)&h_dqdqdqdq;
    return 0;
}

static PyMemberDef members_list[] = {
    {"_frame1", T_OBJECT_EX, offsetof(DistanceConstraint, frame1), 0, trep_internal_doc},
    {"_frame2", T_OBJECT_EX, offsetof(DistanceConstraint, frame2), 0, trep_internal_doc},
    {"_config", T_OBJECT_EX, offsetof(DistanceConstraint, config), 0, trep_internal_doc},
    {"_distance", T_DOUBLE, offsetof(DistanceConstraint, distance), 0, trep_internal_doc},
    {NULL}  /* Sentinel */
};

extern PyTypeObject ConstraintType;
PyTypeObject DistanceConstraintType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_trep._DistanceConstraint",  /*tp_name*/
    sizeof(DistanceConstraint),   /*tp_basicsize*/
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
    &ConstraintType,          /* tp_base */   
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)init,            /* tp_init */
    0,                         /* tp_alloc */
    0,                       /* tp_new */
};
