#include <Python.h>
#include "structmember.h"
#define TREP_MODULE
#include "../trep.h"

typedef struct
{
    Potential potential; // Inherits from Potential
    vec4 gravity;
} GravityPotential;

static double V(GravityPotential *self)
{
    double v = 0.0;
    int i;
    Frame *frame;
    vec4 *p;

    for(i = 0; i < System_MASSES(self->potential.system); i++) {
	frame = System_MASS(self->potential.system, i);
	p = Frame_p(frame);
        v -= frame->mass *  DOT_VEC3(self->gravity, *p);
    }
    return v;
}

static double V_dq(GravityPotential *self, Config *q1)
{
    double v = 0.0;
    int i;
    Frame *frame;
    vec4 *p;

    for(i = 0; i < Config_MASSES(q1); i++) {
	frame = Config_MASS(q1, i);
	p = Frame_p_dq(frame, q1);
	v -= frame->mass * DOT_VEC3(self->gravity, *p);
    }
    return v;
}

static double V_dqdq(GravityPotential *self, Config *q1, Config *q2)
{
    double v = 0.0;
    int i;
    Frame *frame;
    vec4 *p;
    Config *mass_config = NULL;

    // Find the config with the fewest number of masses to iterative over.
    mass_config = q1;
    if(Config_MASSES(q2) < Config_MASSES(mass_config))
	mass_config = q2;

    for(i = 0; i < Config_MASSES(mass_config); i++) {
	frame = Config_MASS(mass_config, i);
	if(!Frame_USES_CONFIG(frame, q1) ||
	   !Frame_USES_CONFIG(frame, q2))
	    continue;

	p = Frame_p_dqdq(frame, q1, q2);
	v -= frame->mass * DOT_VEC3(self->gravity, *p);
    }
    return v;
}

static double V_dqdqdq(GravityPotential *self, Config *q1, Config *q2, Config *q3)
{
    double v = 0.0;
    int i;
    Frame *frame;
    vec4 *p;
    Config *mass_config = NULL;

    // Iterate over the config with the smallest number of masses
    mass_config = q1;
    if(Config_MASSES(q2) < Config_MASSES(mass_config))
	mass_config = q2;
    if(Config_MASSES(q3) < Config_MASSES(mass_config))
	mass_config = q3;

    for(i = 0; i < Config_MASSES(mass_config); i++)
    {
	frame = Config_MASS(mass_config, i);

	if(!Frame_USES_CONFIG(frame, q1) ||
	   !Frame_USES_CONFIG(frame, q2) ||
	   !Frame_USES_CONFIG(frame, q3))
	    continue;
	p = Frame_p_dqdqdq(frame, q1, q2, q3);
	v -= frame->mass * DOT_VEC3(self->gravity, *p);
    }
    return v;
}

static void dealloc(GravityPotential *self)
{
    //((PyObject*)self)->ob_type->tp_free((PyObject*)self);
}

static int init(GravityPotential *self, PyObject *args, PyObject *kwds)
{
    // Note that we do not called Potential.__init__ here.  It will
    // be called by Potential.__init__.
    self->gravity[0] = 0.0;
    self->gravity[1] = 0.0;
    self->gravity[2] = -9.8;
    self->gravity[3] = 0.0;
    
    self->potential.V = (PotentialFunc_V)&V;
    self->potential.V_dq = (PotentialFunc_V_dq)&V_dq;
    self->potential.V_dqdq = (PotentialFunc_V_dqdq)&V_dqdq;
    self->potential.V_dqdqdq = (PotentialFunc_V_dqdqdq)&V_dqdqdq;

    return 0;
}

static PyMemberDef members_list[] = {
    {"_gravity0", T_DOUBLE, offsetof(GravityPotential, gravity[0]), 0, trep_internal_doc},
    {"_gravity1", T_DOUBLE, offsetof(GravityPotential, gravity[1]), 0, trep_internal_doc},
    {"_gravity2", T_DOUBLE, offsetof(GravityPotential, gravity[2]), 0, trep_internal_doc},
    {NULL}  /* Sentinel */
};

extern PyTypeObject PotentialType;
PyTypeObject GravityPotentialType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_trep._GravityPotential", /*tp_name*/
    sizeof(GravityPotential),  /*tp_basicsize*/
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


