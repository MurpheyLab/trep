#include <Python.h>
#define IMPORT_ARRAY // need to import numpy array interface.
#include "trep_internal.h"

PyObject *ConvergenceError; // Exception when MVI step fails to converge.
char trep_internal_doc[] = "Internal use only - see trep developer documentation for more information.";

// The C types are definted in their respective files.
extern PyTypeObject FrameTransformType;
extern PyTypeObject ConfigType;
extern PyTypeObject FrameType;
extern PyTypeObject ForceType;
extern PyTypeObject InputType;
extern PyTypeObject ConstraintType;
extern PyTypeObject PotentialType;
extern PyTypeObject MidpointVIType;
extern PyTypeObject SystemType;
extern PyTypeObject SplineType;
extern PyTypeObject FrameSequenceType;

// Constraint Types defined in src/_trep/constraints/*
extern PyTypeObject DistanceConstraintType;
extern PyTypeObject PointOnPlaneConstraintType;

// Potential Energy Types defined in src/_trep/potentials/*
extern PyTypeObject GravityPotentialType;
extern PyTypeObject LinearSpringPotentialType;
extern PyTypeObject ConfigSpringPotentialType;
extern PyTypeObject NonlinearConfigSpringType;

// Force Types defined in src/_trep/forces/*
extern PyTypeObject DampingForceType;
extern PyTypeObject JointForceType;
extern PyTypeObject BodyWrenchForceType;
extern PyTypeObject HybridWrenchForceType;
extern PyTypeObject SpatialWrenchForceType;

extern PyTypeObject PistonExampleForceType;

void initialize_transform_types(PyObject *module);

static PyMethodDef CTrepMethods[] = {
    {NULL, NULL, 0, NULL}
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC __attribute__((visibility("default"))) init_trep(void) 
{
    int i;
    PyObject* m;

    struct {
        char* name;
        PyTypeObject* type;
        int not_generic_new;
    } custom_types[] = {
        {"_System", &SystemType, 0},
        {"_Frame", &FrameType, 0},
        {"FrameTransform", &FrameTransformType, 1},
        {"_Config", &ConfigType, 0},
        {"_Input", &InputType, 0},
        {"_Potential", &PotentialType, 0},
        {"_Constraint", &ConstraintType, 0},
        {"_Force", &ForceType, 0},
        {"_MidpointVI", &MidpointVIType, 0},
        {"_Spline", &SplineType, 0},
        {"_DistanceConstraint", &DistanceConstraintType, 0},
        {"_PointOnPlaneConstraint", &PointOnPlaneConstraintType, 0},
        {"_GravityPotential", &GravityPotentialType, 0},
        {"_LinearSpringPotential", &LinearSpringPotentialType, 0},
        {"_DampingForce", &DampingForceType, 0},
        {"_JointForce", &JointForceType, 0},
        {"_BodyWrenchForce", &BodyWrenchForceType, 0},
        {"_HybridWrenchForce", &HybridWrenchForceType, 0},
        {"_SpatialWrenchForce", &SpatialWrenchForceType, 0},
        {"_ConfigSpringPotential", &ConfigSpringPotentialType, 0},
        {"_FrameSequence", &FrameSequenceType, 0},
        {"_NonlinearConfigSpring", &NonlinearConfigSpringType, 0},
        {"_PistonExampleForce", &PistonExampleForceType, 0},
        {NULL}
    };

    for(i = 0; custom_types[i].name != NULL; i++) {
        if(!custom_types[i].not_generic_new)
            custom_types[i].type->tp_new = PyType_GenericNew;
        if (PyType_Ready(custom_types[i].type) < 0)
            return;
    }    
           
    // Initialize module
    m = Py_InitModule3("_trep", CTrepMethods,
                       "Example module that creates an extension type.");

    if (m == NULL)
      return;

    // Import Numpy array interface.
    import_array()

    for(i = 0; custom_types[i].name != NULL; i++) {
        Py_INCREF(custom_types[i].type);
        PyModule_AddObject(m, custom_types[i].name, (PyObject *)(custom_types[i].type));
    }    

    // Add Frame transform types
    initialize_transform_types(m);
    // Add integer constants for frame caching bitfields
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_NONE", SYSTEM_CACHE_NONE);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_LG", SYSTEM_CACHE_LG);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_G", SYSTEM_CACHE_G);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_G_DQ", SYSTEM_CACHE_G_DQ);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_G_DQDQ", SYSTEM_CACHE_G_DQDQ);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_G_DQDQDQ", SYSTEM_CACHE_G_DQDQDQ);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_G_DQDQDQDQ", SYSTEM_CACHE_G_DQDQDQDQ);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_G_INV", SYSTEM_CACHE_G_INV);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_G_INV_DQ", SYSTEM_CACHE_G_INV_DQ);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_G_INV_DQDQ", SYSTEM_CACHE_G_INV_DQDQ);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_VB", SYSTEM_CACHE_VB);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_VB_DQ", SYSTEM_CACHE_VB_DQ);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_VB_DQDQ", SYSTEM_CACHE_VB_DQDQ);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_VB_DQDQDQ", SYSTEM_CACHE_VB_DQDQDQ);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_VB_DDQ", SYSTEM_CACHE_VB_DDQ);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_VB_DDQDQ", SYSTEM_CACHE_VB_DDQDQ);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_VB_DDQDQDQ", SYSTEM_CACHE_VB_DDQDQDQ);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_VB_DDQDQDQDQ", SYSTEM_CACHE_VB_DDQDQDQDQ);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_DYNAMICS", SYSTEM_CACHE_DYNAMICS);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_DYNAMICS_DERIV1", SYSTEM_CACHE_DYNAMICS_DERIV1);
    PyModule_AddIntConstant(m, "SYSTEM_CACHE_DYNAMICS_DERIV2", SYSTEM_CACHE_DYNAMICS_DERIV2);

    // Create exceptions
    ConvergenceError = PyErr_NewException("_trep.ConvergenceError", PyExc_StandardError, NULL);
    PyModule_AddObject(m, "ConvergenceError", ConvergenceError);

}

