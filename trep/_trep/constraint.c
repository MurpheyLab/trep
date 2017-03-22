#include <Python.h>
#include "structmember.h"
#define TREP_MODULE
#include "trep.h"

/*
 * These python_h* functions are the default functions pointed to in
 * the Constraint->h*() function pointers.  They call corresponding
 * functions in the Python layer.  This allows new constraints to be
 * written in pure python. If a new constraint defines its own
 * function in C, these are never called.  If the function is not
 * defined in Python, later code will recognize the cycle and raise an
 * exception that the function is not implemented.
 */
static double python_h(Constraint *constraint)
{
    PyObject *ret;
    double value;

    ret = PyObject_CallMethod((PyObject*)constraint, "h", "");
    if(ret == NULL)
	return NAN;
    if(!PyFloat_Check(ret)) {
		PyErr_Format(PyExc_TypeError, "%s.h() returned non-float.", Py_TYPE(constraint)->tp_name);
	Py_XDECREF(ret);
	return NAN;
    }
    value = PyFloat_AS_DOUBLE(ret);
    Py_DECREF(ret);
    return value;
}

static double python_h_dq(Constraint *constraint, Config *q1)
{
    PyObject *ret;
    double value;

    ret = PyObject_CallMethod((PyObject*)constraint, "h_dq", "O", q1);
    if(ret == NULL)
	return NAN;
    if(!PyFloat_Check(ret)) {
		PyErr_Format(PyExc_TypeError, "%s.h_dq() returned non-float.", Py_TYPE(constraint)->tp_name);
	Py_XDECREF(ret);
	return NAN;
    }
    value = PyFloat_AS_DOUBLE(ret);
    Py_DECREF(ret);
    return value;
}

static double python_h_dqdq(Constraint *constraint, Config *q1, Config *q2)
{
    PyObject *ret;
    double value;

    ret = PyObject_CallMethod((PyObject*)constraint, "h_dqdq", "OO", q1, q2);
    if(ret == NULL)
	return NAN;
    if(!PyFloat_Check(ret)) {
		PyErr_Format(PyExc_TypeError, "%s.h_dqdq() returned non-float.", Py_TYPE(constraint)->tp_name);
	Py_XDECREF(ret);
	return NAN;
    }
    value = PyFloat_AS_DOUBLE(ret);
    Py_DECREF(ret);
    return value;
}

static double python_h_dqdqdq(Constraint *constraint, Config *q1, Config *q2, Config *q3)
{
    PyObject *ret;
    double value;

    ret = PyObject_CallMethod((PyObject*)constraint, "h_dqdqdq", "OOO", q1, q2, q3);
    if(ret == NULL)
	return NAN;
    if(!PyFloat_Check(ret)) {
		PyErr_Format(PyExc_TypeError, "%s.h_dqdqdq() returned non-float.", Py_TYPE(constraint)->tp_name);
	Py_XDECREF(ret);
	return NAN;
    }
    value = PyFloat_AS_DOUBLE(ret);
    Py_DECREF(ret);
    return value;
}

static double python_h_dqdqdqdq(Constraint *constraint, Config *q1, Config *q2, Config *q3, Config *q4)
{
    PyObject *ret;
    double value;

    ret = PyObject_CallMethod((PyObject*)constraint, "h_dqdqdq", "OOOO", q1, q2, q3, q4);
    if(ret == NULL)
	return NAN;
    if(!PyFloat_Check(ret)) {
		PyErr_Format(PyExc_TypeError, "%s.h_dqdqdqdq() returned non-float.", Py_TYPE(constraint)->tp_name);
	Py_XDECREF(ret);
	return NAN;
    }
    value = PyFloat_AS_DOUBLE(ret);
    Py_DECREF(ret);
    return value;
}

static void dealloc(Constraint *self)
{
    Py_CLEAR(self->system);
    /* self->ob_type->tp_free((PyObject*)self); */
	Py_TYPE(self)->tp_free((PyObject*)self);
}

static int init(Constraint *self, PyObject *args, PyObject *kwds)
{
    self->tolerance = 1.0e-10;
    self->index = -1;
    self->h = python_h;
    self->h_dq = python_h_dq;
    self->h_dqdq = python_h_dqdq;
    self->h_dqdqdq = python_h_dqdqdq;
    self->h_dqdqdqdq = python_h_dqdqdqdq;
    return 0;
}

static PyGetSetDef getset_list[] = {
    {NULL}  /* Sentinel */
};

static PyObject* h(Constraint *self)
{
    PyObject *ret = NULL;
    
    // Check for cycles (function has not been defined in Python or C layers)
    if(self->h == NULL || self->h == python_h) 
	return PyErr_Format(PyExc_NotImplementedError, "h() is undefined for this constraint.");
    ret = PyFloat_FromDouble(self->h(self));
    if(PyErr_Occurred())
	return NULL;
    return ret;
}

static PyObject* h_dq(Constraint *self, PyObject *args)
{
    PyObject *ret = NULL;
    Config *q1 = NULL;

    // Check for cycles (function has not been defined in Python or C layers)
    if(self->h_dq == NULL || self->h_dq == python_h_dq) 
	return PyErr_Format(PyExc_NotImplementedError, "h_dq() is undefined for this constraint.");
    if(!PyArg_ParseTuple(args, "O", &q1))
        return NULL; 
    ret = PyFloat_FromDouble(self->h_dq(self, q1));
    if(PyErr_Occurred())
	return NULL;
    return ret;
}

static PyObject* h_dqdq(Constraint *self, PyObject *args)
{
    PyObject *ret = NULL;
    Config *q1 = NULL;
    Config *q2 = NULL;

    // Check for cycles (function has not been defined in Python or C layers)
    if(self->h_dqdq == NULL || self->h_dqdq == python_h_dqdq) 
	return PyErr_Format(PyExc_NotImplementedError, "h_dqdq() is undefined for this constraint.");
    if(!PyArg_ParseTuple(args, "OO", &q1, &q2))
        return NULL; 
    ret = PyFloat_FromDouble(self->h_dqdq(self, q1, q2));
    if(PyErr_Occurred())
	return NULL;
    return ret;
}

static PyObject* h_dqdqdq(Constraint *self, PyObject *args)
{
    PyObject *ret = NULL;
    Config *q1 = NULL;
    Config *q2 = NULL;
    Config *q3 = NULL;

    // Check for cycles (function has not been defined in Python or C layers)
    if(self->h_dqdqdq == NULL || self->h_dqdqdq == python_h_dqdqdq) 
	return PyErr_Format(PyExc_NotImplementedError, "h_dqdqdq() is undefined for this constraint.");
    if(!PyArg_ParseTuple(args, "OOO", &q1, &q2, &q3))
        return NULL; 
    ret = PyFloat_FromDouble(self->h_dqdqdq(self, q1, q2, q3));
    if(PyErr_Occurred())
	return NULL;
    return ret;
}

static PyObject* h_dqdqdqdq(Constraint *self, PyObject *args)
{
    PyObject *ret = NULL;
    Config *q1 = NULL;
    Config *q2 = NULL;
    Config *q3 = NULL;
    Config *q4 = NULL;

    // Check for cycles (function has not been defined in Python or C layers)
    if(self->h_dqdqdqdq == NULL || self->h_dqdqdqdq == python_h_dqdqdqdq) 
	return PyErr_Format(PyExc_NotImplementedError, "h_dqdqdqdq() is undefined for this constraint.");
    if(!PyArg_ParseTuple(args, "OOOO", &q1, &q2, &q3, &q4))
        return NULL; 
    ret = PyFloat_FromDouble(self->h_dqdqdqdq(self, q1, q2, q3, q4));
    if(PyErr_Occurred())
	return NULL;
    return ret;
}

static PyMethodDef methods_list[] = {
    {"_h", (PyCFunction)h, METH_NOARGS, trep_internal_doc},
    {"_h_dq", (PyCFunction)h_dq, METH_VARARGS, trep_internal_doc},
    {"_h_dqdq", (PyCFunction)h_dqdq, METH_VARARGS, trep_internal_doc},
    {"_h_dqdqdq", (PyCFunction)h_dqdqdq, METH_VARARGS, trep_internal_doc},
    {"_h_dqdqdqdq", (PyCFunction)h_dqdqdqdq, METH_VARARGS, trep_internal_doc},
    {NULL}  /* Sentinel */
};

static PyMemberDef members_list[] = {
    {"_system", T_OBJECT_EX, offsetof(Constraint, system), 0, trep_internal_doc},
    {"_index", T_INT, offsetof(Constraint, index), 0, trep_internal_doc},
    {"_tolerance", T_DOUBLE, offsetof(Constraint, tolerance), 0, trep_internal_doc},
    {NULL}  /* Sentinel */
};


PyTypeObject ConstraintType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    /* 0,                         /\*ob_size*\/ */
    "_trep._Constraint",       /*tp_name*/
    sizeof(Constraint),        /*tp_basicsize*/
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
    getset_list,               /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)init,            /* tp_init */
    0,                         /* tp_alloc */
    0,                         /* tp_new */
};
