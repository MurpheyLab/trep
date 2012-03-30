#include <Python.h>
#include "structmember.h"
#include "trep_internal.h"

/*
 * These python_f* functions are the default functions pointed to in
 * the Force->f*() function pointers.  They call corresponding
 * functions in the Python layer.  This allows new forces to be
 * written in pure python. If a new force defines its own
 * function in C, these are never called.  If the function is not
 * defined in Python, later code will recognize the cycle and raise an
 * exception that the function is not implemented.
 */

static double python_f(Force *self, Config *q)
{
    PyObject *ret;
    double value;

    ret = PyObject_CallMethod((PyObject*)self, "f", "O", q);
    if(ret == NULL)
	return NAN;
    if(!PyFloat_Check(ret)) {
	PyErr_Format(PyExc_TypeError, "%s.f() returned non-float.", self->ob_type->tp_name);
	Py_XDECREF(ret);
	return NAN;
    }
    value = PyFloat_AS_DOUBLE(ret);
    Py_DECREF(ret);
    return value;
}

static double python_f_dq(Force *self, Config *q, Config *q1)
{
    PyObject *ret;
    double value;

    ret = PyObject_CallMethod((PyObject*)self, "f_dq", "OO", q, q1);
    if(ret == NULL)
	return NAN;
    if(!PyFloat_Check(ret)) {
	PyErr_Format(PyExc_TypeError, "%s.f_dq() returned non-float.", self->ob_type->tp_name);
	Py_XDECREF(ret);
	return NAN;
    }
    value = PyFloat_AS_DOUBLE(ret);
    Py_DECREF(ret);
    return value;
}

static double python_f_ddq(Force *self, Config *q, Config *dq1)
{
    PyObject *ret;
    double value;

    ret = PyObject_CallMethod((PyObject*)self, "f_ddq", "OO", q, dq1);
    if(ret == NULL)
	return NAN;
    if(!PyFloat_Check(ret)) {
	PyErr_Format(PyExc_TypeError, "%s.f_ddq() returned non-float.", self->ob_type->tp_name);
	Py_XDECREF(ret);
	return NAN;
    }
    value = PyFloat_AS_DOUBLE(ret);
    Py_DECREF(ret);
    return value;
}

static double python_f_du(Force *self, Config *q, Input *u1)
{
    PyObject *ret;
    double value;

    ret = PyObject_CallMethod((PyObject*)self, "f_du", "OO", q, u1);
    if(ret == NULL)
	return NAN;
    if(!PyFloat_Check(ret)) {
	PyErr_Format(PyExc_TypeError, "%s.f_du() returned non-float.", self->ob_type->tp_name);
	Py_XDECREF(ret);
	return NAN;
    }
    value = PyFloat_AS_DOUBLE(ret);
    Py_DECREF(ret);
    return value;
}

static double python_f_dqdq(Force *self, Config *q, Config *q1, Config *q2)
{
    PyObject *ret;
    double value;

    ret = PyObject_CallMethod((PyObject*)self, "f_dqdq", "OOO", q, q1, q2);
    if(ret == NULL)
	return NAN;
    if(!PyFloat_Check(ret)) {
	PyErr_Format(PyExc_TypeError, "%s.f_dqdq() returned non-float.", self->ob_type->tp_name);
	Py_XDECREF(ret);
	return NAN;
    }
    value = PyFloat_AS_DOUBLE(ret);
    Py_DECREF(ret);
    return value;
}

static double python_f_ddqdq(Force *self, Config *q, Config *dq1, Config *q2)
{
    PyObject *ret;
    double value;

    ret = PyObject_CallMethod((PyObject*)self, "f_ddqdq", "OOO", q, dq1, q2);
    if(ret == NULL)
	return NAN;
    if(!PyFloat_Check(ret)) {
	PyErr_Format(PyExc_TypeError, "%s.f_ddqdq() returned non-float.", self->ob_type->tp_name);
	Py_XDECREF(ret);
	return NAN;
    }
    value = PyFloat_AS_DOUBLE(ret);
    Py_DECREF(ret);
    return value;
}

static double python_f_ddqddq(Force *self, Config *q, Config *dq1, Config *dq2)
{
    PyObject *ret;
    double value;

    ret = PyObject_CallMethod((PyObject*)self, "f_ddqddq", "OOO", q, dq1, dq2);
    if(ret == NULL)
	return NAN;
    if(!PyFloat_Check(ret)) {
	PyErr_Format(PyExc_TypeError, "%s.f_ddqddq() returned non-float.", self->ob_type->tp_name);
	Py_XDECREF(ret);
	return NAN;
    }
    value = PyFloat_AS_DOUBLE(ret);
    Py_DECREF(ret);
    return value;
}

static double python_f_dudq(Force *self, Config *q, Input *u1, Config *q2)
{
    PyObject *ret;
    double value;

    ret = PyObject_CallMethod((PyObject*)self, "f_dudq", "OOO", q, u1, q2);
    if(ret == NULL)
	return NAN;
    if(!PyFloat_Check(ret)) {
	PyErr_Format(PyExc_TypeError, "%s.f_dudq() returned non-float.", self->ob_type->tp_name);
	Py_XDECREF(ret);
	return NAN;
    }
    value = PyFloat_AS_DOUBLE(ret);
    Py_DECREF(ret);
    return value;
}

static double python_f_duddq(Force *self, Config *q, Input *u1, Config *dq2)
{
    PyObject *ret;
    double value;

    ret = PyObject_CallMethod((PyObject*)self, "f_duddq", "OOO", q, u1, dq2);
    if(ret == NULL)
	return NAN;
    if(!PyFloat_Check(ret)) {
	PyErr_Format(PyExc_TypeError, "%s.f_duddq() returned non-float.", self->ob_type->tp_name);
	Py_XDECREF(ret);
	return NAN;
    }
    value = PyFloat_AS_DOUBLE(ret);
    Py_DECREF(ret);
    return value;
}

static double python_f_dudu(Force *self, Config *q, Input *u1, Input *u2)
{
    PyObject *ret;
    double value;

    ret = PyObject_CallMethod((PyObject*)self, "f_dudu", "OOO", q, u1, u2);
    if(ret == NULL)
	return NAN;
    if(!PyFloat_Check(ret)) {
	PyErr_Format(PyExc_TypeError, "%s.f_dudu() returned non-float.", self->ob_type->tp_name);
	Py_XDECREF(ret);
	return NAN;
    }
    value = PyFloat_AS_DOUBLE(ret);
    Py_DECREF(ret);
    return value;
}

static void dealloc(Force *self)
{
    Py_CLEAR(self->system);
    self->ob_type->tp_free((PyObject*)self);
}

static int init(Force *self, PyObject *args, PyObject *kwds)
{
    self->f = python_f;
    self->f_dq = python_f_dq;
    self->f_ddq = python_f_ddq;
    self->f_du = python_f_du;
    self->f_dqdq = python_f_dqdq;
    self->f_ddqdq = python_f_ddqdq;
    self->f_ddqddq = python_f_ddqddq;
    self->f_dudq = python_f_dudq;
    self->f_duddq = python_f_duddq;
    self->f_dudu = python_f_dudu;
    return 0;
}

static PyObject* f(Force *self, PyObject *args)
{
    Config *q = NULL;
    PyObject *ret = NULL;
    
    // Check for cycles (function has not been defined in Python or C layers)
    if(self->f == NULL || self->f == python_f) 
	return PyErr_Format(PyExc_NotImplementedError, "f() is undefined for this force.");
    if(!PyArg_ParseTuple(args, "O", &q))
        return NULL; 
    ret = PyFloat_FromDouble(self->f(self, q));
    if(PyErr_Occurred())
	return NULL;
    return ret;
}

static PyObject* f_dq(Force *self, PyObject *args)
{
    PyObject *ret = NULL;
    Config *q = NULL;
    Config *q1 = NULL;

    // Check for cycles (function has not been defined in Python or C layers)
    if(self->f_dq == NULL || self->f_dq == python_f_dq) 
	return PyErr_Format(PyExc_NotImplementedError, "f_dq() is undefined for this force.");
    if(!PyArg_ParseTuple(args, "OO", &q, &q1))
        return NULL; 
    ret = PyFloat_FromDouble(self->f_dq(self, q, q1));
    if(PyErr_Occurred())
	return NULL;
    return ret;
}

static PyObject* f_ddq(Force *self, PyObject *args)
{
    PyObject *ret = NULL;
    Config *q = NULL;
    Config *dq1 = NULL;

    // Check for cycles (function has not been defined in Python or C layers)
    if(self->f_ddq == NULL || self->f_ddq == python_f_ddq) 
	return PyErr_Format(PyExc_NotImplementedError, "f_ddq() is undefined for this force.");
    if(!PyArg_ParseTuple(args, "OO", &q, &dq1))
        return NULL; 
    ret = PyFloat_FromDouble(self->f_ddq(self, q, dq1));
    if(PyErr_Occurred())
	return NULL;
    return ret;
}

static PyObject* f_du(Force *self, PyObject *args)
{
    PyObject *ret = NULL;
    Config *q = NULL;
    Input *u1 = NULL;

    // Check for cycles (function has not been defined in Python or C layers)
    if(self->f_du == NULL || self->f_du == python_f_du) 
	return PyErr_Format(PyExc_NotImplementedError, "f_du() is undefined for this force.");
    if(!PyArg_ParseTuple(args, "OO", &q, &u1))
        return NULL; 
    ret = PyFloat_FromDouble(self->f_du(self, q, u1));
    if(PyErr_Occurred())
	return NULL;
    return ret;
}

static PyObject* f_dqdq(Force *self, PyObject *args)
{
    PyObject *ret = NULL;
    Config *q = NULL;
    Config *q1 = NULL;
    Config *q2 = NULL;

    // Check for cycles (function has not been defined in Python or C layers)
    if(self->f_dqdq == NULL || self->f_dqdq == python_f_dqdq) 
	return PyErr_Format(PyExc_NotImplementedError, "f_dqdq() is undefined for this force.");
    if(!PyArg_ParseTuple(args, "OOO", &q, &q1, &q2))
        return NULL; 
    ret = PyFloat_FromDouble(self->f_dqdq(self, q, q1, q2));
    if(PyErr_Occurred())
	return NULL;
    return ret;
}

static PyObject* f_ddqdq(Force *self, PyObject *args)
{
    PyObject *ret = NULL;
    Config *q = NULL;
    Config *dq1 = NULL;
    Config *q2 = NULL;

    // Check for cycles (function has not been defined in Python or C layers)
    if(self->f_ddqdq == NULL || self->f_ddqdq == python_f_ddqdq) 
	return PyErr_Format(PyExc_NotImplementedError, "f_ddqdq() is undefined for this force.");
    if(!PyArg_ParseTuple(args, "OOO", &q, &dq1, &q2))
        return NULL; 
    ret = PyFloat_FromDouble(self->f_ddqdq(self, q, dq1, q2));
    if(PyErr_Occurred())
	return NULL;
    return ret;
}

static PyObject* f_ddqddq(Force *self, PyObject *args)
{
    PyObject *ret = NULL;
    Config *q = NULL;
    Config *dq1 = NULL;
    Config *dq2 = NULL;

    // Check for cycles (function has not been defined in Python or C layers)
    if(self->f_ddqddq == NULL || self->f_ddqddq == python_f_ddqdq) 
	return PyErr_Format(PyExc_NotImplementedError, "f_ddqddq() is undefined for this force.");
    if(!PyArg_ParseTuple(args, "OOO", &q, &dq1, &dq2))
        return NULL; 
    ret = PyFloat_FromDouble(self->f_ddqddq(self, q, dq1, dq2));
    if(PyErr_Occurred())
	return NULL;
    return ret;
}

static PyObject* f_dudq(Force *self, PyObject *args)
{
    PyObject *ret = NULL;
    Config *q = NULL;
    Input *u1 = NULL;
    Config *q2 = NULL;
    
    // Check for cycles (function has not been defined in Python or C layers)
    if(self->f_dudq == NULL || self->f_dudq == python_f_dudq) 
	return PyErr_Format(PyExc_NotImplementedError, "f_dudq() is undefined for this force.");
    if(!PyArg_ParseTuple(args, "OOO", &q, &u1, &q2))
        return NULL; 
    ret = PyFloat_FromDouble(self->f_dudq(self, q, u1, q2));
    if(PyErr_Occurred())
	return NULL;
    return ret;
}

static PyObject* f_duddq(Force *self, PyObject *args)
{
    PyObject *ret = NULL;
    Config *q = NULL;
    Input *u1 = NULL;
    Config *dq2 = NULL;
    
    // Check for cycles (function has not been defined in Python or C layers)
    if(self->f_duddq == NULL || self->f_duddq == python_f_duddq) 
	return PyErr_Format(PyExc_NotImplementedError, "f_duddq() is undefined for this force.");
    if(!PyArg_ParseTuple(args, "OOO", &q, &u1, &dq2))
        return NULL; 
    ret = PyFloat_FromDouble(self->f_duddq(self, q, u1, dq2));
    if(PyErr_Occurred())
	return NULL;
    return ret;
}

static PyObject* f_dudu(Force *self, PyObject *args)
{
    PyObject *ret = NULL;
    Config *q = NULL;
    Input *u1 = NULL;
    Input *u2 = NULL;
    
    // Check for cycles (function has not been defined in Python or C layers)
    if(self->f_dudu == NULL || self->f_dudu == python_f_dudu) 
	return PyErr_Format(PyExc_NotImplementedError, "f_dudu() is undefined for this force.");
    if(!PyArg_ParseTuple(args, "OOO", &q, &u1, &u2))
        return NULL; 
    ret = PyFloat_FromDouble(self->f_dudu(self, q, u1, u2));
    if(PyErr_Occurred())
	return NULL;
    return ret;
}

static PyMethodDef methods_list[] = {
    {"_f", (PyCFunction)f, METH_VARARGS, trep_internal_doc},
    {"_f_dq", (PyCFunction)f_dq, METH_VARARGS, trep_internal_doc},
    {"_f_ddq", (PyCFunction)f_ddq, METH_VARARGS, trep_internal_doc},
    {"_f_du", (PyCFunction)f_du, METH_VARARGS, trep_internal_doc},
    {"_f_dqdq", (PyCFunction)f_dqdq, METH_VARARGS, trep_internal_doc},
    {"_f_ddqdq", (PyCFunction)f_ddqdq, METH_VARARGS, trep_internal_doc},
    {"_f_ddqddq", (PyCFunction)f_ddqddq, METH_VARARGS, trep_internal_doc},
    {"_f_dudq", (PyCFunction)f_dudq, METH_VARARGS, trep_internal_doc},
    {"_f_duddq", (PyCFunction)f_duddq, METH_VARARGS, trep_internal_doc},
    {"_f_dudu", (PyCFunction)f_dudu, METH_VARARGS, trep_internal_doc},
    {NULL}  /* Sentinel */
};

static PyMemberDef members_list[] = {
    {"_system", T_OBJECT_EX, offsetof(Force, system), 0, trep_internal_doc},
    {NULL}  /* Sentinel */
};

PyTypeObject ForceType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_trep._Force",            /*tp_name*/
    sizeof(Force),             /*tp_basicsize*/
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
    0,                         /* tp_new */
};
