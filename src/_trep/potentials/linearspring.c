#include <Python.h>
#include "structmember.h"
#include "trep.h"

typedef struct
{
    Potential potential; // Inherits from Potential
    Frame *frame1;
    Frame *frame2;
    double k;
    double x0;
} LinearSpringPotential;


static double V(LinearSpringPotential *self)
{
    double x;
    vec4 v;

    sub_vec4(v,
	     *Frame_p(self->frame1),
	     *Frame_p(self->frame2));
    x = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    return 0.5 * self->k * (x - self->x0) * (x - self->x0);
}

static double V_dq(LinearSpringPotential *self, Config *q1)
{
    double x, dx;
    vec4 v, dv;

    sub_vec4(v,
	     *Frame_p(self->frame1),
	     *Frame_p(self->frame2));
    sub_vec4(dv,
	     *Frame_p_dq(self->frame1, q1),
	     *Frame_p_dq(self->frame2, q1));
    x = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    dx = (1.0/x)*(v[0]*dv[0] + v[1]*dv[1] + v[2]*dv[2]);
    return self->k * (x - self->x0) * dx;
}

static double V_dqdq(LinearSpringPotential *self, Config *q1, Config *q2)
{
    double x, dix, djx, ddx;
    vec4 v, div, djv, ddv;

    sub_vec4(v,
	     *Frame_p(self->frame1),
	     *Frame_p(self->frame2));
    sub_vec4(div,
	     *Frame_p_dq(self->frame1, q1),
	     *Frame_p_dq(self->frame2, q1));
    sub_vec4(djv,
	     *Frame_p_dq(self->frame1, q2),
	     *Frame_p_dq(self->frame2, q2));
    sub_vec4(ddv,
	     *Frame_p_dqdq(self->frame1, q1, q2),
	     *Frame_p_dqdq(self->frame2, q1, q2));

    x = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    dix = (1.0/x)*(v[0]*div[0] + v[1]*div[1] + v[2]*div[2]);
    djx = (1.0/x)*(v[0]*djv[0] + v[1]*djv[1] + v[2]*djv[2]);
    ddx = -djx/(x*x) * (v[0]*div[0] + v[1]*div[1] + v[2]*div[2]) +
	   1.0/x * (djv[0]*div[0] + djv[1]*div[1] + djv[2]*div[2]) +
	   1.0/x * (v[0]*ddv[0] + v[1]*ddv[1] + v[2]*ddv[2]);
    return self->k * dix * djx + self->k * (x - self->x0) * ddx;
}

static void dealloc(LinearSpringPotential *self)
{
    Py_CLEAR(self->frame1);
    Py_CLEAR(self->frame2);
    ((PyObject*)self)->ob_type->tp_free((PyObject*)self);
}

static int init(LinearSpringPotential *self, PyObject *args, PyObject *kwds)
{
    // Note that we do not called Potential.__init__ here.  It will
    // be called by Potential.__init__.
    self->x0 = 0.0;
    self->k = 0.0;

    self->potential.V = (PotentialFunc_V)&V;
    self->potential.V_dq = (PotentialFunc_V_dq)&V_dq;
    self->potential.V_dqdq = (PotentialFunc_V_dqdq)&V_dqdq;    
    return 0;
}


static PyMemberDef members_list[] = {
    {"_frame1", T_OBJECT_EX, offsetof(LinearSpringPotential, frame1), 0, trep_internal_doc},
    {"_frame2", T_OBJECT_EX, offsetof(LinearSpringPotential, frame2), 0, trep_internal_doc},
    {"_x0", T_DOUBLE, offsetof(LinearSpringPotential, x0), 0, trep_internal_doc},
    {"_k", T_DOUBLE, offsetof(LinearSpringPotential, k), 0, trep_internal_doc},
    {NULL}  /* Sentinel */
};


extern PyTypeObject PotentialType;
PyTypeObject LinearSpringPotentialType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_trep._LinearSpringPotential",  /*tp_name*/
    sizeof(LinearSpringPotential),   /*tp_basicsize*/
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


