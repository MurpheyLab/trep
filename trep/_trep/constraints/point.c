#include <Python.h>
#include "structmember.h"
#define TREP_MODULE
#include "../trep.h"


typedef struct
{
    Constraint constraint;  // Inherits from Constraint
    Frame *frame1;
    Frame *frame2;
    int component;
} PointToPointConstraint;


static double h(PointToPointConstraint *self)
{
    vec4 v;

    sub_vec4(v,
	     *Frame_p(self->frame1),
	     *Frame_p(self->frame2));

    return v[self->component];
}

static double h_dq(PointToPointConstraint *self, Config *q1)
{
    vec4 v_d1;

    sub_vec4(v_d1,
	     *Frame_p_dq(self->frame1, q1),
	     *Frame_p_dq(self->frame2, q1));

    return v_d1[self->component];
}

static double h_dqdq(PointToPointConstraint *self, Config *q1, Config *q2)
{
    vec4 v_d12;

    sub_vec4(v_d12, *Frame_p_dqdq(self->frame1, q1, q2), *Frame_p_dqdq(self->frame2, q1, q2));

    return v_d12[self->component];
}

static double h_dqdqdq(PointToPointConstraint *self, Config *q1, Config *q2, Config *q3)
{
    vec4 v_d123;

    sub_vec4(v_d123, *Frame_p_dqdqdq(self->frame1, q1, q2, q3), *Frame_p_dqdqdq(self->frame2, q1, q2, q3));

    return v_d123[self->component];
}

static double h_dqdqdqdq(PointToPointConstraint *self, Config *q1, Config *q2, Config *q3, Config *q4)
{
    vec4 v_d1234;

    sub_vec4(v_d1234, *Frame_p_dqdqdqdq(self->frame1, q1, q2, q3, q4),
             *Frame_p_dqdqdqdq(self->frame2, q1, q2, q3, q4));

    return v_d1234[self->component];
}

static void dealloc(PointToPointConstraint *self)
{
    Py_CLEAR(self->frame1);
    Py_CLEAR(self->frame2);
    /* ((PyObject*)self)->ob_type->tp_free((PyObject*)self); */
	Py_TYPE(((PyObject*)self))->tp_free((PyObject*)self);
}

static int init(PointToPointConstraint *self, PyObject *args, PyObject *kwds)
{
    // Note that we do not called Constraint.__init__ here.  It will
    // be called by Constraint.__init__.

    self->constraint.h = (ConstraintFunc_h)&h;
    self->constraint.h_dq = (ConstraintFunc_h_dq)&h_dq;
    self->constraint.h_dqdq = (ConstraintFunc_h_dqdq)&h_dqdq;
    self->constraint.h_dqdqdq = (ConstraintFunc_h_dqdqdq)&h_dqdqdq;
    self->constraint.h_dqdqdqdq = (ConstraintFunc_h_dqdqdqdq)&h_dqdqdqdq;

    return 0;
}

static PyMemberDef members_list[] = {
    {"_frame1", T_OBJECT_EX, offsetof(PointToPointConstraint, frame1), 0, trep_internal_doc},
    {"_frame2", T_OBJECT_EX, offsetof(PointToPointConstraint, frame2), 0, trep_internal_doc},
    {"_component", T_INT, offsetof(PointToPointConstraint, component), 0, trep_internal_doc},
    {NULL}  /* Sentinel */
};

extern PyTypeObject ConstraintType;
PyTypeObject PointToPointConstraintType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    /* 0,                         /\*ob_size*\/ */
    "_trep._PointToPointConstraint",  /*tp_name*/
    sizeof(PointToPointConstraint),   /*tp_basicsize*/
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
