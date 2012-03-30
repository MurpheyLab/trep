#include <Python.h>
#include "structmember.h"
#include "../trep_internal.h"


typedef struct
{
    Constraint constraint;  // Inherits from Constraint
    Frame *plane_frame;
    Frame *point_frame;
    vec4 normal;
} PointOnPlaneConstraint;

static double h(PointOnPlaneConstraint *self)
{
    vec4 n;
    vec4 dp;

    mul_mv4(n, *Frame_g(self->plane_frame), self->normal);
    sub_vec4(dp,
	     *Frame_p(self->plane_frame),
	     *Frame_p(self->point_frame));    
    return n[0]*dp[0] + n[1]*dp[1] + n[2]*dp[2];
}

static double h_dq(PointOnPlaneConstraint *self,  Config *q1)
{
    vec4 n;
    vec4 dp;
    double h = 0.0;

    mul_mv4(n, *Frame_g_dq(self->plane_frame, q1), self->normal);
    sub_vec4(dp, *Frame_p(self->plane_frame), *Frame_p(self->point_frame));
    h = n[0]*dp[0] + n[1]*dp[1] + n[2]*dp[2];

    mul_mv4(n, *Frame_g(self->plane_frame), self->normal);    
    sub_vec4(dp,
	     *Frame_p_dq(self->plane_frame, q1),
	     *Frame_p_dq(self->point_frame, q1));
    h += n[0]*dp[0] + n[1]*dp[1] + n[2]*dp[2];

    return h;
}

static double h_dqdq(PointOnPlaneConstraint *self, Config *q1, Config *q2)
{
    vec4 n;
    vec4 dp;
    double h = 0.0;

    mul_mv4(n, *Frame_g_dqdq(self->plane_frame, q1, q2), self->normal);
    sub_vec4(dp, *Frame_p(self->plane_frame), *Frame_p(self->point_frame));
    h = n[0]*dp[0] + n[1]*dp[1] + n[2]*dp[2];

    mul_mv4(n, *Frame_g_dq(self->plane_frame, q1), self->normal);    
    sub_vec4(dp,
	     *Frame_p_dq(self->plane_frame, q2),
	     *Frame_p_dq(self->point_frame, q2));
    h += n[0]*dp[0] + n[1]*dp[1] + n[2]*dp[2];

    mul_mv4(n, *Frame_g_dq(self->plane_frame, q2), self->normal);    
    sub_vec4(dp,
	     *Frame_p_dq(self->plane_frame, q1),
	     *Frame_p_dq(self->point_frame, q1));
    h += n[0]*dp[0] + n[1]*dp[1] + n[2]*dp[2];

    mul_mv4(n, *Frame_g(self->plane_frame), self->normal);    
    sub_vec4(dp,
	     *Frame_p_dqdq(self->plane_frame, q1, q2),
	     *Frame_p_dqdq(self->point_frame, q1, q2));
    h += n[0]*dp[0] + n[1]*dp[1] + n[2]*dp[2];

    return h;
}

static void dealloc(PointOnPlaneConstraint *self)
{
    Py_CLEAR(self->plane_frame);
    Py_CLEAR(self->point_frame);
    ((PyObject*)self)->ob_type->tp_free((PyObject*)self);
}

static int init(PointOnPlaneConstraint *self, PyObject *args, PyObject *kwds)
{
    // Note that we do not called Constraint.__init__ here.  It will
    // be called by Constraint.__init__.
    clear_vec4(self->normal);
    
    self->constraint.h = (ConstraintFunc_h)&h;
    self->constraint.h_dq = (ConstraintFunc_h_dq)&h_dq;
    self->constraint.h_dqdq = (ConstraintFunc_h_dqdq)&h_dqdq;
    //self->constraint.h_dqdqdq = (ConstraintFunc_h_dqdqdq)&h_dqdqdq;
    //self->constraint.h_dqdqdqdq = (ConstraintFunc_h_dqdqdqdq)&h_dqdqdqdq;
    return 0;
}

static PyMemberDef members_list[] = {
    {"_plane_frame", T_OBJECT_EX, offsetof(PointOnPlaneConstraint, plane_frame), 0, trep_internal_doc},
    {"_point_frame", T_OBJECT_EX, offsetof(PointOnPlaneConstraint, point_frame), 0, trep_internal_doc},
    {"_normal0", T_DOUBLE, offsetof(PointOnPlaneConstraint, normal[0]), 0, trep_internal_doc},
    {"_normal1", T_DOUBLE, offsetof(PointOnPlaneConstraint, normal[1]), 0, trep_internal_doc},
    {"_normal2", T_DOUBLE, offsetof(PointOnPlaneConstraint, normal[2]), 0, trep_internal_doc},
    {NULL}  /* Sentinel */
};

extern PyTypeObject ConstraintType;
PyTypeObject PointOnPlaneConstraintType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_trep._PointOnPlaneConstraint",  /*tp_name*/
    sizeof(PointOnPlaneConstraint),   /*tp_basicsize*/
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
