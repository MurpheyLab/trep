#include <Python.h>
#include "structmember.h"
#include "trep.h"


typedef struct
{
    Constraint constraint;  // Inherits from Constraint
    Frame *frame1;
    Frame *frame2;
    vec4 axis;
} PointConstraint;

static double h(PointConstraint *self)
{
    vec4 n;
    vec4 dp;

    mul_mv4(n, *Frame_g(self->frame1), self->axis);
    sub_vec4(dp,
	     *Frame_p(self->frame1),
	     *Frame_p(self->frame2));    
    return n[0]*dp[0] + n[1]*dp[1] + n[2]*dp[2];
}

static double h_dq(PointConstraint *self,  Config *q1)
{
    vec4 n;
    vec4 dp;
    double h = 0.0;

    mul_mv4(n, *Frame_g_dq(self->frame1, q1), self->axis);
    sub_vec4(dp, *Frame_p(self->frame1), *Frame_p(self->frame2));
    h = n[0]*dp[0] + n[1]*dp[1] + n[2]*dp[2];

    mul_mv4(n, *Frame_g(self->frame1), self->axis);    
    sub_vec4(dp,
	     *Frame_p_dq(self->frame1, q1),
	     *Frame_p_dq(self->frame2, q1));
    h += n[0]*dp[0] + n[1]*dp[1] + n[2]*dp[2];

    return h;
}

static double h_dqdq(PointConstraint *self, Config *q1, Config *q2)
{
    vec4 n;
    vec4 dp;
    double h = 0.0;

    mul_mv4(n, *Frame_g_dqdq(self->frame1, q1, q2), self->axis);
    sub_vec4(dp, *Frame_p(self->frame1), *Frame_p(self->frame2));
    h = n[0]*dp[0] + n[1]*dp[1] + n[2]*dp[2];

    mul_mv4(n, *Frame_g_dq(self->frame1, q1), self->axis);    
    sub_vec4(dp,
	     *Frame_p_dq(self->frame1, q2),
	     *Frame_p_dq(self->frame2, q2));
    h += n[0]*dp[0] + n[1]*dp[1] + n[2]*dp[2];

    mul_mv4(n, *Frame_g_dq(self->frame1, q2), self->axis);    
    sub_vec4(dp,
	     *Frame_p_dq(self->frame1, q1),
	     *Frame_p_dq(self->frame2, q1));
    h += n[0]*dp[0] + n[1]*dp[1] + n[2]*dp[2];

    mul_mv4(n, *Frame_g(self->frame1), self->axis);    
    sub_vec4(dp,
	     *Frame_p_dqdq(self->frame1, q1, q2),
	     *Frame_p_dqdq(self->frame2, q1, q2));
    h += n[0]*dp[0] + n[1]*dp[1] + n[2]*dp[2];

    return h;
}

static void dealloc(PointConstraint *self)
{
    Py_CLEAR(self->frame1);
    Py_CLEAR(self->frame2);
    ((PyObject*)self)->ob_type->tp_free((PyObject*)self);
}

static int init(PointConstraint *self, PyObject *args, PyObject *kwds)
{
    // Note that we do not called Constraint.__init__ here.  It will
    // be called by Constraint.__init__.
    clear_vec4(self->axis);
    
    self->constraint.h = (ConstraintFunc_h)&h;
    self->constraint.h_dq = (ConstraintFunc_h_dq)&h_dq;
    self->constraint.h_dqdq = (ConstraintFunc_h_dqdq)&h_dqdq;
    //self->constraint.h_dqdqdq = (ConstraintFunc_h_dqdqdq)&h_dqdqdq;
    //self->constraint.h_dqdqdqdq = (ConstraintFunc_h_dqdqdqdq)&h_dqdqdqdq;
    return 0;
}

static PyMemberDef members_list[] = {
    {"_frame1", T_OBJECT_EX, offsetof(PointConstraint, frame1), 0, trep_internal_doc},
    {"_frame2", T_OBJECT_EX, offsetof(PointConstraint, frame2), 0, trep_internal_doc},
    {"_axis0", T_DOUBLE, offsetof(PointConstraint, axis[0]), 0, trep_internal_doc},
    {"_axis1", T_DOUBLE, offsetof(PointConstraint, axis[1]), 0, trep_internal_doc},
    {"_axis2", T_DOUBLE, offsetof(PointConstraint, axis[2]), 0, trep_internal_doc},
    {NULL}  /* Sentinel */
};

extern PyTypeObject ConstraintType;
PyTypeObject PointConstraintType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_trep._PointConstraint",  /*tp_name*/
    sizeof(PointConstraint),   /*tp_basicsize*/
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
