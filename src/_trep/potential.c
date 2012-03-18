#include <Python.h>
#include "structmember.h"
#include "trep.h"

static double python_V(Potential *potential)
{
    PyObject *ret;
    double value;

    ret = PyObject_CallMethod((PyObject*)potential, "V", "");
    if(ret == NULL)
	return NAN;
    if(!PyFloat_Check(ret)) {
	PyErr_Format(PyExc_TypeError, "%s.V() returned non-float.",
	    potential->ob_type->tp_name);
	Py_XDECREF(ret);
	return NAN;
    }
    value = PyFloat_AS_DOUBLE(ret);
    Py_DECREF(ret);
    return value;
}

static double python_V_dq(Potential *potential, Config *q1)
{
    PyObject *ret;
    double value;

    ret = PyObject_CallMethod((PyObject*)potential, "V_dq", "O", q1);
    if(ret == NULL)
	return NAN;
    if(!PyFloat_Check(ret)) {
	PyErr_Format(PyExc_TypeError, "%s.V_dq() returned non-float.",
	    potential->ob_type->tp_name);
	Py_XDECREF(ret);
	return NAN;
    }
    value = PyFloat_AS_DOUBLE(ret);
    Py_DECREF(ret);
    return value;
}

static double python_V_dqdq(Potential *potential, Config *q1, Config *q2)
{
    PyObject *ret;
    double value;

    ret = PyObject_CallMethod((PyObject*)potential, "V_dqdq", "OO", q1, q2);
    if(ret == NULL)
	return NAN;
    if(!PyFloat_Check(ret)) {
	PyErr_Format(PyExc_TypeError, "%s.V_dqdq() returned non-float.",
	    potential->ob_type->tp_name);
	Py_XDECREF(ret);
	return NAN;
    }
    value = PyFloat_AS_DOUBLE(ret);
    Py_DECREF(ret);
    return value;
}

static double python_V_dqdqdq(Potential *potential, Config *q1, Config *q2, Config *q3)
{
    PyObject *ret;
    double value;

    ret = PyObject_CallMethod((PyObject*)potential, "V_dqdqdq", "OOO", q1, q2, q3);
    if(ret == NULL)
	return NAN;
    if(!PyFloat_Check(ret)) {
	PyErr_Format(PyExc_TypeError, "%s.V_dqdqdq() returned non-float.",
	    potential->ob_type->tp_name);
	Py_XDECREF(ret);
	return NAN;
    }
    value = PyFloat_AS_DOUBLE(ret);
    Py_DECREF(ret);
    return value;
}

static void dealloc(Potential *self)
{
    Py_CLEAR(self->system);
    self->ob_type->tp_free((PyObject*)self);
}

static int init(Potential *self, PyObject *args, PyObject *kwds)
{
    self->V = python_V;
    self->V_dq = python_V_dq;
    self->V_dqdq = python_V_dqdq;
    self->V_dqdqdq = python_V_dqdqdq;
    return 0;
}

static PyObject* V(Potential *self)
{
    PyObject *ret = NULL;
    
    // Check for cycles (function has not been defined in Python or C layers)
    if(self->V == NULL || self->V == python_V)
	return PyErr_Format(PyExc_NotImplementedError, "V() is undefined for this potential.");
    ret = PyFloat_FromDouble(self->V(self));
    if(PyErr_Occurred()) 
	return NULL;
    return ret;
}

static PyObject* V_dq(Potential *self, PyObject *args)
{
    PyObject *ret = NULL;
    Config *config = NULL;

    // Check for cycles (function has not been defined in Python or C layers)
    if(self->V == NULL || self->V_dq == python_V_dq)
	return PyErr_Format(PyExc_NotImplementedError, "V_dq() is undefined for this potential.");
    if(!PyArg_ParseTuple(args, "O", &config))
        return NULL; 
    ret = PyFloat_FromDouble(self->V_dq(self, config));
    if(PyErr_Occurred())
	return NULL;
    return ret;
}

static PyObject* V_dqdq(Potential *self, PyObject *args)
{
    PyObject *ret = NULL;
    Config *config1 = NULL;
    Config *config2 = NULL;

    // Check for cycles (function has not been defined in Python or C layers)
    if(self->V_dqdq == NULL || self->V_dqdq == python_V_dqdq)
	return PyErr_Format(PyExc_NotImplementedError, "V_dqdq() is undefined for this potential.");
    if(!PyArg_ParseTuple(args, "OO", &config1, &config2))
        return NULL; 
    ret = PyFloat_FromDouble(self->V_dqdq(self, config1, config2));
    if(PyErr_Occurred())
	return NULL;
    return ret;
}

static PyObject* V_dqdqdq(Potential *self, PyObject *args)
{
    PyObject *ret = NULL;
    Config *q1 = NULL;
    Config *q2 = NULL;
    Config *q3 = NULL;

    // Check for cycles (function has not been defined in Python or C layers)
    if(self->V_dqdqdq == python_V_dqdqdq)
	return PyErr_Format(PyExc_NotImplementedError, "V_dqdqdq() is undefined for this potential.");
    if(!PyArg_ParseTuple(args, "OOO", &q1, &q2, &q3))
        return NULL; 
    ret = PyFloat_FromDouble(self->V_dqdqdq(self, q1, q2, q3));
    if(PyErr_Occurred())
	return NULL;
    return ret;
}

static PyMethodDef methods_list[] = {
    {"_V", (PyCFunction)V, METH_NOARGS, trep_internal_doc},
    {"_V_dq", (PyCFunction)V_dq, METH_VARARGS, trep_internal_doc},
    {"_V_dqdq", (PyCFunction)V_dqdq, METH_VARARGS, trep_internal_doc},
    {"_V_dqdqdq", (PyCFunction)V_dqdqdq, METH_VARARGS, trep_internal_doc},
    {NULL}  /* Sentinel */
};


static PyMemberDef members_list[] = {
    {"_system", T_OBJECT_EX, offsetof(Potential, system), 0, trep_internal_doc},
    {NULL}  /* Sentinel */
};


PyTypeObject PotentialType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_trep._Potential",       /*tp_name*/
    sizeof(Potential),        /*tp_basicsize*/
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
    trep_internal_doc,         /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    methods_list,              /* tp_methods */
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
