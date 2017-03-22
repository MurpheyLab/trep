#include <Python.h>
#include <structmember.h>
#define TREP_MODULE
#include "trep.h"

double TapeMeasure_length(TapeMeasure *self)
{
    double xk = 0.0;  // Length of one segment
    double x = 0.0;   // Total length
    vec4 v;           // Vector between two points
    Frame *frame1 = NULL;
    Frame *frame2 = NULL;
    int k;

    for(k = 0; k < PyTuple_GET_SIZE(self->frames)-1; k++) {
        frame1 = (Frame*)PyTuple_GET_ITEM(self->frames, k);
        frame2 = (Frame*)PyTuple_GET_ITEM(self->frames, k+1);

        sub_vec4(v, *Frame_p(frame1), *Frame_p(frame2));
        xk = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        
        x += xk;
    }
    return x;
}

double TapeMeasure_length_dq(TapeMeasure *self, Config *q1)
{
    int *segments = (int*)IDX1(self->seg_table, q1->index);
    double xk = 0.0;  // Length of one segment
    double xk_dq1 = 0.0; // Derivative of one segment
    double x_dq1 = 0.0; // Derivative of total length 
    vec4 v;           // Vector between two points
    vec4 v_dq1;       // Derivative of vector between two points.
    Frame *frame1 = NULL;
    Frame *frame2 = NULL;
    int k;

    for(k = 0; segments[k] != -1; k++) {
        frame1 = (Frame*)PyTuple_GET_ITEM(self->frames, segments[k]);
        frame2 = (Frame*)PyTuple_GET_ITEM(self->frames, segments[k]+1);

        sub_vec4(v, *Frame_p(frame1), *Frame_p(frame2));
        sub_vec4(v_dq1, *Frame_p_dq(frame1, q1), *Frame_p_dq(frame2, q1));
        xk = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        xk_dq1 = 1.0/xk*(v[0]*v_dq1[0] + v[1]*v_dq1[1] + v[2]*v_dq1[2]);

        x_dq1 += xk_dq1;
    }
    return x_dq1;
}

double TapeMeasure_length_dqdq(TapeMeasure *self, Config *q1, Config *q2)
{
    double xk = 0.0;  // Length of one segment
    double xk_dq1 = 0.0; // 1st Derivative of one segment
    double xk_dq2 = 0.0; // 1st Derivative of one segment
    double xk_dq1dq2 = 0.0; // 2nd Derivative of one segment
    double x_dq1dq2 = 0.0; // 2nd Derivative of total length
    vec4 v;           // Vector between two points
    vec4 v_dq1;       // 1st Derivative of vector between two points.
    vec4 v_dq2;       // 1st Derivative of vector between two points.
    vec4 v_dq1dq2;       // 2nd Derivative of vector between two points.
    Frame *frame1 = NULL;
    Frame *frame2 = NULL;
    int k;

    int *segments = (int*)IDX1(self->seg_table, q1->index);
    
    for(k = 0; segments[k] != -1; k++) {
        frame1 = (Frame*)PyTuple_GET_ITEM(self->frames, segments[k]);
        frame2 = (Frame*)PyTuple_GET_ITEM(self->frames, segments[k]+1);

        /* Only need to work on segments where only one frame depends on q2 */
        if(!Frame_USES_CONFIG(frame1, q2) && !Frame_USES_CONFIG(frame2, q2))
            continue;
        if(Frame_USES_CONFIG(frame1, q2) && Frame_USES_CONFIG(frame2, q2))
            continue;

        sub_vec4(v, *Frame_p(frame1), *Frame_p(frame2));
        sub_vec4(v_dq1, *Frame_p_dq(frame1, q1), *Frame_p_dq(frame2, q1));
        sub_vec4(v_dq2, *Frame_p_dq(frame1, q2), *Frame_p_dq(frame2, q2));
        sub_vec4(v_dq1dq2, *Frame_p_dqdq(frame1, q1, q2), *Frame_p_dqdq(frame2, q1, q2));
        
        xk = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        xk_dq1 = 1.0/xk*(v[0]*v_dq1[0] + v[1]*v_dq1[1] + v[2]*v_dq1[2]);
        xk_dq2 = 1.0/xk*(v[0]*v_dq2[0] + v[1]*v_dq2[1] + v[2]*v_dq2[2]);
        xk_dq1dq2 = xk_dq1 * xk_dq2 
            - (v_dq1[0]*v_dq2[0] + v_dq1[1]*v_dq2[1] + v_dq1[2]*v_dq2[2])
            - (v[0]*v_dq1dq2[0] + v[1]*v_dq1dq2[1] + v[2]*v_dq1dq2[2]);
        xk_dq1dq2 = -1.0/xk * xk_dq1dq2;

        x_dq1dq2 += xk_dq1dq2;
    }
    return x_dq1dq2;
}

double TapeMeasure_length_dqdqdq(TapeMeasure *self, Config *q1, Config *q2, Config *q3)
{
    double xk = 0.0;  // Length of one segment
    double xk_dq1 = 0.0; // 1st Derivative of one segment
    double xk_dq2 = 0.0; // 1st Derivative of one segment
    double xk_dq3 = 0.0; // 1st Derivative of one segment
    double xk_dq1dq2 = 0.0; // 2nd Derivative of one segment
    double xk_dq1dq3 = 0.0; // 2nd Derivative of one segment
    double xk_dq2dq3 = 0.0; // 2nd Derivative of one segment
    double xk_dq1dq2dq3 = 0.0; // 3rd Derivative of one segment
    double x_dq1dq2dq3 = 0.0; // 3rd Derivative of total length
    vec4 v;           // Vector between two points
    vec4 v_dq1;       // 1st Derivative of vector between two points.
    vec4 v_dq2;       // 1st Derivative of vector between two points.
    vec4 v_dq3;       // 1st Derivative of vector between two points.
    vec4 v_dq1dq2;       // 2nd Derivative of vector between two points.
    vec4 v_dq1dq3;       // 2nd Derivative of vector between two points.
    vec4 v_dq2dq3;       // 2nd Derivative of vector between two points.
    vec4 v_dq1dq2dq3;       // 3rd Derivative of vector between two points.
    Frame *frame1 = NULL;
    Frame *frame2 = NULL;
    int k;

    for(k = 0; k < PyTuple_GET_SIZE(self->frames)-1; k++) {
        frame1 = (Frame*)PyTuple_GET_ITEM(self->frames, k);
        frame2 = (Frame*)PyTuple_GET_ITEM(self->frames, k+1);

        /* if(!Frame_USES_CONFIG(frame1, q1) && !Frame_USES_CONFIG(frame2, q1)) */
        /*     continue; */
        /* if(!Frame_USES_CONFIG(frame1, q2) && !Frame_USES_CONFIG(frame2, q2)) */
        /*     continue; */
        /* if(!Frame_USES_CONFIG(frame1, q3) && !Frame_USES_CONFIG(frame2, q3)) */
        /*     continue; */

        sub_vec4(v, *Frame_p(frame1), *Frame_p(frame2));
        sub_vec4(v_dq1, *Frame_p_dq(frame1, q1), *Frame_p_dq(frame2, q1));
        sub_vec4(v_dq2, *Frame_p_dq(frame1, q2), *Frame_p_dq(frame2, q2));
        sub_vec4(v_dq3, *Frame_p_dq(frame1, q3), *Frame_p_dq(frame2, q3));
        sub_vec4(v_dq1dq2, *Frame_p_dqdq(frame1, q1, q2), *Frame_p_dqdq(frame2, q1, q2));
        sub_vec4(v_dq1dq3, *Frame_p_dqdq(frame1, q1, q3), *Frame_p_dqdq(frame2, q1, q3));
        sub_vec4(v_dq2dq3, *Frame_p_dqdq(frame1, q2, q3), *Frame_p_dqdq(frame2, q2, q3));
        sub_vec4(v_dq1dq2dq3, *Frame_p_dqdqdq(frame1, q1, q2, q3), *Frame_p_dqdqdq(frame2, q1, q2, q3));
        
        xk = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        xk_dq1 = 1.0/xk*(v[0]*v_dq1[0] + v[1]*v_dq1[1] + v[2]*v_dq1[2]);
        xk_dq2 = 1.0/xk*(v[0]*v_dq2[0] + v[1]*v_dq2[1] + v[2]*v_dq2[2]);
        xk_dq3 = 1.0/xk*(v[0]*v_dq3[0] + v[1]*v_dq3[1] + v[2]*v_dq3[2]);
        xk_dq1dq2 = xk_dq1 * xk_dq2 +
            - (v_dq1[0]*v_dq2[0] + v_dq1[1]*v_dq2[1] + v_dq1[2]*v_dq2[2]) 
            - (v[0]*v_dq1dq2[0] + v[1]*v_dq1dq2[1] + v[2]*v_dq1dq2[2]);
        xk_dq1dq2 = -1.0/xk * xk_dq1dq2;
        xk_dq1dq3 = xk_dq1 * xk_dq3 +
            - (v_dq1[0]*v_dq3[0] + v_dq1[1]*v_dq3[1] + v_dq1[2]*v_dq3[2]) 
            - (v[0]*v_dq1dq3[0] + v[1]*v_dq1dq3[1] + v[2]*v_dq1dq3[2]);
        xk_dq1dq3 = -1.0/xk * xk_dq1dq3;
        xk_dq2dq3 = xk_dq2 * xk_dq3 
            - (v_dq2[0]*v_dq3[0] + v_dq2[1]*v_dq3[1] + v_dq2[2]*v_dq3[2]) 
            - (v[0]*v_dq2dq3[0] + v[1]*v_dq2dq3[1] + v[2]*v_dq2dq3[2]);
        xk_dq2dq3 = -1.0/xk * xk_dq2dq3;
        
        xk_dq1dq2dq3 = xk_dq1 * xk_dq2dq3 + xk_dq2 * xk_dq1dq3 + xk_dq3 * xk_dq1dq2
            - (v_dq1[0]*v_dq2dq3[0] + v_dq1[1]*v_dq2dq3[1] + v_dq1[2]*v_dq2dq3[2])
            - (v_dq2[0]*v_dq1dq3[0] + v_dq2[1]*v_dq1dq3[1] + v_dq2[2]*v_dq1dq3[2])
            - (v_dq3[0]*v_dq1dq2[0] + v_dq3[1]*v_dq1dq2[1] + v_dq3[2]*v_dq1dq2[2])
            - (v[0]*v_dq1dq2dq3[0] + v[1]*v_dq1dq2dq3[1] + v[2]*v_dq1dq2dq3[2]);
        xk_dq1dq2dq3 = -1.0/xk * xk_dq1dq2dq3;

        x_dq1dq2dq3 += xk_dq1dq2dq3;
    }
    return x_dq1dq2dq3;
}

double TapeMeasure_velocity(TapeMeasure *self)
{
    Config *q = NULL;
    double dxdq = 0;
    double v = 0.0;    
    int k;

    for(k = 0; k < System_CONFIGS(self->system); k++) {
        q = System_CONFIG(self->system, k);
        dxdq = TapeMeasure_length_dq(self, q);
        v += dxdq * q->dq;
    }

    return v;
}

double TapeMeasure_velocity_dq(TapeMeasure *self, Config *q1)
{
    Config *q = NULL;
    double dxdqdq1 = 0;
    double v_dq1 = 0.0;    
    int k;

    for(k = 0; k < System_CONFIGS(self->system); k++) {
        q = System_CONFIG(self->system, k);
        dxdqdq1 = TapeMeasure_length_dqdq(self, q, q1);
        v_dq1 += dxdqdq1 * q->dq;
    }

    return v_dq1;
}

double TapeMeasure_velocity_dqdq(TapeMeasure *self, Config *q1, Config *q2)
{
    Config *q = NULL;
    double dxdqdq1dq2 = 0;
    double v_dq1dq2 = 0.0;    
    int k;

    for(k = 0; k < System_CONFIGS(self->system); k++) {
        q = System_CONFIG(self->system, k);
        dxdqdq1dq2 = TapeMeasure_length_dqdqdq(self, q, q1, q2);
        v_dq1dq2 += dxdqdq1dq2 * q->dq;
    }

    return v_dq1dq2;
}

double TapeMeasure_velocity_ddq(TapeMeasure *self, Config *dq1)
{
    return TapeMeasure_length_dq(self, dq1);
}

double TapeMeasure_velocity_ddqdq(TapeMeasure *self, Config *dq1, Config *q2)
{
    return TapeMeasure_length_dqdq(self, dq1, q2);
}

static void dealloc(TapeMeasure *self)
{
    Py_CLEAR(self->system);
    Py_CLEAR(self->frames);
    Py_CLEAR(self->seg_table);
    /* self->ob_type->tp_free((PyObject*)self); */
	Py_TYPE(self)->tp_free((PyObject*)self);
}

static int init(TapeMeasure *self, PyObject *args, PyObject *kwds)
{
    return 0;
}

static PyObject* Py_length(TapeMeasure *self, PyObject *args)
{
    double x;

    x = TapeMeasure_length(self);
    return Py_BuildValue("d", x);
}

static PyObject* Py_length_dq(TapeMeasure *self, PyObject *args)
{
    double x_dq1;
    Config *q1 = NULL;

    if(!PyArg_ParseTuple(args, "O", &q1))
        return NULL; 
    
    x_dq1 = TapeMeasure_length_dq(self, q1);
    return Py_BuildValue("d", x_dq1);
}

static PyObject* Py_length_dqdq(TapeMeasure *self, PyObject *args)
{
    double x_dq1dq2;
    Config *q1 = NULL;
    Config *q2 = NULL;

    if(!PyArg_ParseTuple(args, "OO", &q1, &q2))
        return NULL; 
    
    x_dq1dq2 = TapeMeasure_length_dqdq(self, q1, q2);
    return Py_BuildValue("d", x_dq1dq2);
}

static PyObject* Py_length_dqdqdq(TapeMeasure *self, PyObject *args)
{
    double x_dq1dq2dq3;
    Config *q1 = NULL;
    Config *q2 = NULL;
    Config *q3 = NULL;

    if(!PyArg_ParseTuple(args, "OOO", &q1, &q2, &q3))
        return NULL; 
    
    x_dq1dq2dq3 = TapeMeasure_length_dqdqdq(self, q1, q2, q3);
    return Py_BuildValue("d", x_dq1dq2dq3);
}

static PyObject* Py_velocity(TapeMeasure *self, PyObject *args)
{
    double x;

    x = TapeMeasure_velocity(self);
    return Py_BuildValue("d", x);
}

static PyObject* Py_velocity_dq(TapeMeasure *self, PyObject *args)
{
    double x_dq1;
    Config *q1 = NULL;

    if(!PyArg_ParseTuple(args, "O", &q1))
        return NULL; 
    
    x_dq1 = TapeMeasure_velocity_dq(self, q1);
    return Py_BuildValue("d", x_dq1);
}

static PyObject* Py_velocity_dqdq(TapeMeasure *self, PyObject *args)
{
    double x_dq1dq2;
    Config *q1 = NULL;
    Config *q2 = NULL;

    if(!PyArg_ParseTuple(args, "OO", &q1, &q2))
        return NULL; 
    
    x_dq1dq2 = TapeMeasure_velocity_dqdq(self, q1, q2);
    return Py_BuildValue("d", x_dq1dq2);
}

static PyObject* Py_velocity_ddq(TapeMeasure *self, PyObject *args)
{
    double x_ddq1;
    Config *dq1 = NULL;

    if(!PyArg_ParseTuple(args, "O", &dq1))
        return NULL; 
    
    x_ddq1 = TapeMeasure_velocity_ddq(self, dq1);
    return Py_BuildValue("d", x_ddq1);
}

static PyObject* Py_velocity_ddqdq(TapeMeasure *self, PyObject *args)
{
    double x_ddq1dq2;
    Config *dq1 = NULL;
    Config *q2 = NULL;

    if(!PyArg_ParseTuple(args, "OO", &dq1, &q2))
        return NULL; 
    
    x_ddq1dq2 = TapeMeasure_velocity_ddqdq(self, dq1, q2);
    return Py_BuildValue("d", x_ddq1dq2);
}

static PyMethodDef methods_list[] = {
    {"_length", (PyCFunction)Py_length, METH_VARARGS, trep_internal_doc},
    {"_length_dq", (PyCFunction)Py_length_dq, METH_VARARGS, trep_internal_doc},
    {"_length_dqdq", (PyCFunction)Py_length_dqdq, METH_VARARGS, trep_internal_doc},
    {"_length_dqdqdq", (PyCFunction)Py_length_dqdqdq, METH_VARARGS, trep_internal_doc},
    {"_velocity", (PyCFunction)Py_velocity, METH_VARARGS, trep_internal_doc},
    {"_velocity_dq", (PyCFunction)Py_velocity_dq, METH_VARARGS, trep_internal_doc},
    {"_velocity_dqdq", (PyCFunction)Py_velocity_dqdq, METH_VARARGS, trep_internal_doc},
    {"_velocity_ddq", (PyCFunction)Py_velocity_ddq, METH_VARARGS, trep_internal_doc},
    {"_velocity_ddqdq", (PyCFunction)Py_velocity_ddqdq, METH_VARARGS, trep_internal_doc},
    {NULL}  /* Sentinel */
};

static PyMemberDef members_list[] = {
    {"_system", T_OBJECT_EX, offsetof(TapeMeasure, system), 0, trep_internal_doc},
    {"_frames", T_OBJECT_EX, offsetof(TapeMeasure, frames), 0, trep_internal_doc},
    {"_seg_table", T_OBJECT_EX, offsetof(TapeMeasure, seg_table), 0, trep_internal_doc},
    {NULL}  /* Sentinel */
};

PyTypeObject TapeMeasureType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    /* 0,                         /\*ob_size*\/ */
    "_trep._TapeMeasure",    /*tp_name*/
    sizeof(TapeMeasure),     /*tp_basicsize*/
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


