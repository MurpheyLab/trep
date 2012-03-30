#include <Python.h>
#include "trep_internal.h"
#include "structmember.h"


#include <stdio.h>
static int get_index(Spline *self, double x)
{
    int i;
    int len = (int)PyArray_SIZE(self->x_points);
    double *x_points = (double*)PyArray_DATA(self->x_points);
    if(x < x_points[0])
        return 0;
    if(x >= x_points[len-1])
        return len - 2;
    i = 0;
    while(x >= x_points[i+1])
        i += 1;
    return i;         
}

double Spline_y(Spline *self, double x)
{
    int i = get_index(self, x);
    double *x_points = (double*)PyArray_DATA(self->x_points);
    double *coeffs   = (double*)PyArray_GETPTR1(self->coeffs, i);
    double dx = x - x_points[i];
    return (coeffs[0]*dx*dx*dx*dx*dx +
            coeffs[1]*dx*dx*dx*dx +
            coeffs[2]*dx*dx*dx +
            coeffs[3]*dx*dx +
            coeffs[4]*dx +
            coeffs[5]);
}

double Spline_dy(Spline *self, double x)
{
    int i = get_index(self, x);
    double *x_points = (double*)PyArray_DATA(self->x_points);
    double *coeffs   = (double*)PyArray_GETPTR1(self->coeffs, i);
    double dx = x - x_points[i];
    return (5*coeffs[0]*dx*dx*dx*dx +
            4*coeffs[1]*dx*dx*dx +
            3*coeffs[2]*dx*dx +
            2*coeffs[3]*dx +
            1*coeffs[4]);
}

double Spline_ddy(Spline *self, double x)
{
    int i = get_index(self, x);
    double *x_points = (double*)PyArray_DATA(self->x_points);
    double *coeffs   = (double*)PyArray_GETPTR1(self->coeffs, i);
    double dx = x - x_points[i];
    return (20*coeffs[0]*dx*dx*dx +
            12*coeffs[1]*dx*dx +
             6*coeffs[2]*dx +
             2*coeffs[3]);
}


static void dealloc(Spline *self)
{
    Py_CLEAR(self->x_points);
    Py_CLEAR(self->y_points);
    Py_CLEAR(self->coeffs);
    self->ob_type->tp_free((PyObject*)self);
}

static int init(Spline *self, PyObject *args, PyObject *kwds)
{
    self->x_points = NULL;
    self->y_points = NULL;
    self->coeffs = NULL;
    return 0;
}

static PyObject* py_calc_y(Spline *self, PyObject *args)
{
    double x;
    if(!PyArg_ParseTuple(args, "d", &x))
        return NULL; 
    return PyFloat_FromDouble(Spline_y(self, x));
}

static PyObject* py_calc_dy(Spline *self, PyObject *args)
{
    double x;
    if(!PyArg_ParseTuple(args, "d", &x))
        return NULL; 
    return PyFloat_FromDouble(Spline_dy(self, x));
}

static PyObject* py_calc_ddy(Spline *self, PyObject *args)
{
    double x;
    if(!PyArg_ParseTuple(args, "d", &x))
        return NULL; 
    return PyFloat_FromDouble(Spline_ddy(self, x));
}

static PyMethodDef methods_list[] = {
    {"_y", (PyCFunction)py_calc_y, METH_VARARGS, trep_internal_doc},
    {"_dy", (PyCFunction)py_calc_dy, METH_VARARGS, trep_internal_doc},
    {"_ddy", (PyCFunction)py_calc_ddy, METH_VARARGS, trep_internal_doc},
    {NULL}  /* Sentinel */
};

static PyGetSetDef getset_list[] = {
    {NULL}  /* Sentinel */
};

static PyMemberDef members_list[] = {
    {"_x_points", T_OBJECT_EX, offsetof(Spline, x_points), 0, trep_internal_doc},
    {"_coefficients", T_OBJECT_EX, offsetof(Spline, coeffs), 0, trep_internal_doc},
    {NULL}  /* Sentinel */
};

PyTypeObject SplineType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_trep._Spline",           /*tp_name*/
    sizeof(Spline),            /*tp_basicsize*/
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

