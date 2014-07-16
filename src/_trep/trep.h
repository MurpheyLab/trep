#ifndef _TREP_INTERNAL_H_
#define _TREP_INTERNAL_H_


#ifdef TREP_MODULE
/* If we're being included by another extension model, let them handle importing numpy. */

/* Import Numpy C API as described at
 * http://docs.scipy.org/doc/numpy/reference/c-api.array.html#miscellaneous
 * Tested against Numpy API 1.7
 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL trep_ARRAY_API
#ifndef IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#endif
#include <numpy/arrayobject.h>

#endif /* TREP_MODULE */



/*******************************************************************************
 *******************************************************************************
 *******************************************************************************
 * These are public definitions that are exported in the C API. Any
 * changes need to be updated in trep.h
 */


#ifdef TREP_MODULE
/* When we build the module, these functions should not be declared
 * static since they are shared among files.
 */
#define PYEXPORT
#else
/* But they are static when this is included by other extension modules.
 */
#define PYEXPORT static
#endif



#ifdef USE_CALLGRIND
#include <valgrind/callgrind.h>
#else
/* Remove START/STOP instrumentation macros when callgrind isn't
 * included. */
#define CALLGRIND_START_INSTRUMENTATION
#define CALLGRIND_STOP_INSTRUMENTATION
#endif


/*******************************************************************************
 * Types
 */

typedef struct System_s System;
typedef struct Frame_s Frame;
typedef struct Config_s Config;
typedef struct Force_s Force;
typedef struct Input_s Input;
typedef struct Constraint_s Constraint;
typedef struct Potential_s Potential;
typedef struct MidpointVI_s MidpointVI;
typedef struct Spline_s Spline;
typedef struct TapeMeasure_s TapeMeasure;

// Common fixed-dimensional vectors & matrices
typedef double vec4[4];
typedef double vec6[6];
typedef double mat4x4[4][4];


/*******************************************************************************
 * FrameTransformType acts as an enumeration to define the frame
 * transform types.  We could just use integer constants, but by
 * defining a new type, we can have python print out string
 * representations like "TX" instead of "2" so it's a bit easier to
 * work with on the python command line.
 */
typedef struct {
    PyObject_HEAD
    PyObject *name;
} FrameTransform;

#ifdef TREP_MODULE
/* These are defined in C_API for external extensions.  They cannot
 * share a definition because extensions are defined as static.
 */
extern FrameTransform* TREP_WORLD;
extern FrameTransform* TREP_TX;
extern FrameTransform* TREP_TY;
extern FrameTransform* TREP_TZ;
extern FrameTransform* TREP_RX;
extern FrameTransform* TREP_RY;
extern FrameTransform* TREP_RZ;
extern FrameTransform* TREP_CONST_SE3;
#endif

/*******************************************************************************
 * Config Objects
 */

struct Config_s {
    PyObject_HEAD

    /* 'system' is the system that configuration variable belongs
     * in. */
    System *system;

    /* 'kinematic' is a boolean value that indicates if this is a
     * kinematic or dynamic configuration variable.  Dynamic variables
     * are "normal" configuration variables.  They are affected by the
     * dynamics of the system.  Kinematic variables are considered
     * "ideally controlled" inputs.  The user directly specifies their
     * next value (discrete dynamics) or acceleration (continuous
     * dynamics) instead of being calculated by the dynamics. */
    char kinematic;

    /* 'q' and 'dq' are the configuration variables current value and
     * current time derivative (ie, velocity). */
    double q;
    double dq;
    double ddq;

    /* 'masses' is a tuple of all the frames that depend on this
     * config that have non-zero inertia properties.  It is managed by
     * the system's synchronize() function and is used to improve
     * performance in the Lagrangian calculation.  When someone asks
     * for a derivative of the Lagrangian with respect to a config, we
     * only have to iterative over the frames that depend on the
     * config. */
    PyTupleObject *masses;  

    /* 'config_gen' applies only to configuration variables that drive
     * frame transformations.  It is the number of variable
     * transformations between this config's frame and the base of the
     * tree.  This is best illustrated with a diagram that I have yet
     * to draw.  This value is extremely important to the caching
     * code.  It is managed by the synchronize() functions.
     *
     * This value is -1 for configuration variables that do not drive
     * frame transformations (e.g. kinematic variables used in String
     * constraints). */
    int config_gen;         

    /* index is the index of the configuration variable in its
     * System's 'configs' tuple.  It is managed by the system's
     * synchronize() function. */
    int index;
    /* k_index is the index of the configuration variable in its
     *  System's 'kinematic_configs' tuple.  It is managed by the
     *  system's synchronize() function.  For dynamic configuration
     *  variables, it is -1. */
    int k_index;
};

// Return the i-th frame with in the config's masses tuple.
#define Config_MASS(self, i) ((Frame*)PyTuple_GET_ITEM(self->masses, i))
// Return the number of masses that depend on this config.
#define Config_MASSES(self) PyTuple_GET_SIZE(self->masses)

/*******************************************************************************
 * Frame Objects
 */

struct Frame_s {
    PyObject_HEAD

    /* 'system' is the system that the frame belongs in. */
    System *system;

    /* 'transform' is the transformation that defines the frame from
     * its parent frame. */
    FrameTransform *transform;

    /* 'value' is the transformation parameter for constant frames.
     * It is ignored for variable frames.*/
    double value;

    /* 'config' is the configuration variable that the frame depends
     * on.  It is NULL for constant frames. */
    Config *config;

    /*  'parent' is the frame's parent frame.  It is non-NULL except
     *  for the world_frame. */
    Frame *parent;

    /* 'child' is a tuple of the frames children.  It is always
     * non-NULL, but may be empty. */
    PyTupleObject *children; 

    /* These are the mass parameters for the frame.  They default to
     * zero.  They should NOT be set directly.  They need to be set
     * with set_mass() so that masses() can be updated to include
     * every frame with non-zero interia properties. */
    double mass, Ixx, Iyy, Izz;

    /* The remaining members of Frame are used to implement caching of
     * frame-related values (position, velocities, and their
     * derivatives).  They are managed by the Frame's synchronize()
     * function. 
     */
    
    /* 'cache_index' is a tuple of every configuration variable that
     * this frame depends on.  The configs are ordered from the base
     * of the tree to the frame.  Because of this, if a frame depends
     * on a config (using psuedocode): cache_index[config->config_gen]
     * == config_gen.  This provides an extremely efficient way to
     * check for dependencies and look up values from the cache.
     *
     * The caches for derivatives are arrays that are the same length
     * as cache_index and are ordered in the same way.  In other
     * words, if cache_index = [configA, configG, configE], then g_dq
     * = [dg/d(configA), dg/d(configG), dg/d(configE)].  Again, this
     * allows for extremely fast look ups in the cache.  When given a
     * config, we check if the frame depends on it, and if it does,
     * extract the value from the cache using config->config_gen to
     * index the array. */

    /* New: Profiling found Frame_get_cache_index to be a major
     * bottleneck.  To reduce the number of checks done there,
     * cache_index has been redone.  Every cache_index is now a tuple
     * of length N+1.  The index is build as before, and then all the
     * extra items are set to PyNone.  cache_size is the number of the
     * non-None configs in the index.  Now you can always do
     * cache_index[config->config_gen] without having to check for
     * config_gen being -1 or to make sure cache_index is long
     * enough. This seems to give a speed up of around 10% for second
     * derivatives.
     */
    int cache_size;
    PyTupleObject *cache_index;

    /* The remaining members store the actual cache data.  Caches are
     * maintained separately for each value, but for all frames.  In
     * other words, when someone calls Frame_get_g(), the function
     * will build 'g' values for all frames, but will not build 'g_dq'
     * values.  The 'cache' member of System indicates which values
     * have been cached. */
    mat4x4 lg, lg_inv;
    mat4x4 lg_dq, lg_inv_dq;
    mat4x4 lg_dqdq, lg_inv_dqdq;
    mat4x4 lg_dqdqdq, lg_inv_dqdqdq;
    mat4x4 lg_dqdqdqdq, lg_inv_dqdqdqdq;
    mat4x4 twist_hat;
    mat4x4 g, g_inv;
    PyArrayObject *g_dq;
    PyArrayObject *g_dqdq;
    PyArrayObject *g_dqdqdq;
    PyArrayObject *g_dqdqdqdq;
    PyArrayObject *g_inv_dq;
    PyArrayObject *g_inv_dqdq;
    vec4 p;
    PyArrayObject *p_dq;
    PyArrayObject *p_dqdq;
    PyArrayObject *p_dqdqdq;
    PyArrayObject *p_dqdqdqdq;
    mat4x4 vb;
    PyArrayObject *vb_dq;
    PyArrayObject *vb_dqdq;
    PyArrayObject *vb_dqdqdq;
    PyArrayObject *vb_ddq;
    PyArrayObject *vb_ddqdq;
    PyArrayObject *vb_ddqdqdq;
    PyArrayObject *vb_ddqdqdqdq;

    // New optimized caching stuff - see trep optimization doc for description.
    double cos_param, sin_param;
    void (*multiply_gk)(Frame *self, mat4x4 dest, mat4x4 X, int n);
    void (*add_sandwich_gk)(Frame *self, mat4x4 dest, mat4x4 X, int n1, int n2);
};

/* Return the i-th child of the frame. */
#define Frame_CHILD(self, i) ((Frame*)PyTuple_GET_ITEM(self->children, i))
/* Return the number of children of the frame. */
#define Frame_CHILD_SIZE(self) PyTuple_GET_SIZE(self->children)
/* Return the i-th config in the cache index. */
#define Frame_CACHE(self, i) ((Config*)PyTuple_GET_ITEM(self->cache_index, i))
/* Return the size of the cache index. */
//#define Frame_CACHE_SIZE(self) PyTuple_GET_SIZE(self->cache_size)
#define Frame_CACHE_SIZE(self) (self->cache_size)
/* Return true if frame depends on config, false if not. */
#define Frame_USES_CONFIG(self, q) (Frame_CACHE(self, q->config_gen) == q)


/*******************************************************************************
 * Input Objects
 */

struct Input_s {
    PyObject_HEAD;
    System *system;  // System that the input belongs in.
    Force *force;  // Force that uses this input.
    double u;  // Current value of the force input
    int index;  // Index in system's Input tuple.
};


/*******************************************************************************
 * Force Objects
 */

typedef double (*ForceFunc_f)(Force *self, Config *q);
typedef double (*ForceFunc_f_dq)(Force *self, Config *q, Config *q1);
typedef double (*ForceFunc_f_ddq)(Force *self, Config *q, Config *dq1);
typedef double (*ForceFunc_f_du)(Force *self, Config *q, Input *u1);
typedef double (*ForceFunc_f_dqdq)(Force *self, Config *q, Config *q1, Config *q2);
typedef double (*ForceFunc_f_ddqdq)(Force *self, Config *q, Config *dq1, Config *q2);
typedef double (*ForceFunc_f_ddqddq)(Force *self, Config *q, Config *dq1, Config *dq2);
typedef double (*ForceFunc_f_dudq)(Force *self, Config *q, Input *u1, Config *q2);
typedef double (*ForceFunc_f_duddq)(Force *self, Config *q, Input *u1, Config *dq2);
typedef double (*ForceFunc_f_dudu)(Force *self, Config *q, Input *u1, Input *u2);

struct Force_s {
    PyObject_HEAD;
    
    System *system;  // System that the force belongs in.

    ForceFunc_f f;
    ForceFunc_f_dq f_dq;
    ForceFunc_f_ddq f_ddq;
    ForceFunc_f_du f_du;
    ForceFunc_f_dqdq f_dqdq;
    ForceFunc_f_ddqdq f_ddqdq;
    ForceFunc_f_ddqddq f_ddqddq;
    ForceFunc_f_dudq f_dudq;
    ForceFunc_f_duddq f_duddq;
    ForceFunc_f_dudu f_dudu;
};

/*******************************************************************************
 * Constraint Objects
 */

typedef double (*ConstraintFunc_h)(Constraint *constraint);
typedef double (*ConstraintFunc_h_dq)(Constraint *constraint, Config *q1);
typedef double (*ConstraintFunc_h_dqdq)(Constraint *constraint, Config *q1, Config *q2);
typedef double (*ConstraintFunc_h_dqdqdq)(Constraint *constraint, Config *q1, Config *q2, Config *q3);
typedef double (*ConstraintFunc_h_dqdqdqdq)(Constraint *constraint, Config *q1, Config *q2, Config *q3, Config *q4);
struct Constraint_s {
    PyObject_HEAD

    System *system;
    double tolerance;
    int index;

    ConstraintFunc_h h;
    ConstraintFunc_h_dq h_dq;
    ConstraintFunc_h_dqdq h_dqdq;
    ConstraintFunc_h_dqdqdq h_dqdqdq;
    ConstraintFunc_h_dqdqdqdq h_dqdqdqdq;
};

/*******************************************************************************
 * Potential Objects
 */


typedef double (*PotentialFunc_V)(Potential *self);
typedef double (*PotentialFunc_V_dq)(Potential *self, Config *q1);
typedef double (*PotentialFunc_V_dqdq)(Potential *self, Config *q1, Config *q2);
typedef double (*PotentialFunc_V_dqdqdq)(Potential *self, Config *q1, Config *q2, Config *q3);

struct Potential_s {
    PyObject_HEAD

    System *system;

    PotentialFunc_V V;
    PotentialFunc_V_dq V_dq;
    PotentialFunc_V_dqdq V_dqdq;
    PotentialFunc_V_dqdqdq V_dqdqdq;
};



/*******************************************************************************
 * MidpointVI Objects
 */

#define MIDPOINTVI_CACHE_SOLUTION        0x01
#define MIDPOINTVI_CACHE_SOLUTION_DERIV1 0x02
#define MIDPOINTVI_CACHE_SOLUTION_DERIV2 0x04 

struct MidpointVI_s {
    PyObject_HEAD

    System *system;
    double tolerance;
    unsigned long cache;

    /* These indicate the sizes of the allocated variables below. They
     * may not necessarily equal the sizes of the system, they just
     * need to be larger.  (ie, you can build a variational
     * integrator, delete a bunch of stuff, and then the variables
     * will be larger than needed.  This is so that in the future if
     * sizes are frequently changing in simulation, like from
     * collisions and contact modelling, we don't have to constantly
     * allocate new variables.
     */
    double t1, t2;
    PyArrayObject *q1, *q2; // [nd+nk]
    PyArrayObject *p1, *p2; // [nd]
    PyArrayObject *u1;      // [nu]
    PyArrayObject *lambda1;

    PyArrayObject *Dh1T; // [nd+nk x nc]
    PyArrayObject *Dh2; // [nc x nd+nk]

    // Used for the root-finding to calculate the next configuration
    PyArrayObject *f;
    PyArrayObject *Df;
    PyArrayObject *Df_index;

    /* First order derivative variables */
    PyArrayObject *DDh1T; // [nd+nk x nd+nk x nc]

    PyArrayObject *M2_lu; // [nd x nd]
    PyArrayObject *M2_lu_index; // [nd]
    PyArrayObject *proj_lu; // [nc x nc]
    PyArrayObject *proj_lu_index; // [nc]
  
    /*  indexed as [derivative variable index]["output" value index] */
    PyArrayObject *q2_dq1; // [nd+nk x nd]
    PyArrayObject *q2_dp1; // [nd x nd]
    PyArrayObject *q2_du1; // [nu x nd]
    PyArrayObject *q2_dk2; // [nk x nd]
    PyArrayObject *p2_dq1; // [nd+nk x nd]
    PyArrayObject *p2_dp1; // [nd x nd]
    PyArrayObject *p2_du1; // [nu x nd]
    PyArrayObject *p2_dk2; // [nk x nd]
    PyArrayObject *l1_dq1; // [nd+nk x nc]
    PyArrayObject *l1_dp1; // [nd x nc]
    PyArrayObject *l1_du1; // [nu x nc]
    PyArrayObject *l1_dk2; // [nk x nc]
    
    /* Second order derivatives are index as:
     * p1_dAdB ->  [A variable index][B variable index][output index]
     */
    PyArrayObject *q2_dq1dq1; // [nd+nk x nd+nk x nd]
    PyArrayObject *q2_dq1dp1; // [nd+nk x nd x nd]
    PyArrayObject *q2_dq1du1; // [nd+nk x nu x nd]
    PyArrayObject *q2_dq1dk2; // [nd+nk x nk x nd]
    PyArrayObject *q2_dp1dp1; // [nd x nd x nd]
    PyArrayObject *q2_dp1du1; // [nd x nu x nd]
    PyArrayObject *q2_dp1dk2; // [nd x nk x nd]
    PyArrayObject *q2_du1du1; // [nu x nu x nd]
    PyArrayObject *q2_du1dk2; // [nu x nk x nd]
    PyArrayObject *q2_dk2dk2; // [nk x nk x nd]

    PyArrayObject *p2_dq1dq1; // [nd+nk x nd+nk x nd]
    PyArrayObject *p2_dq1dp1; // [nd+nk x nd x nd]
    PyArrayObject *p2_dq1du1; // [nd+nk x nu x nd]
    PyArrayObject *p2_dq1dk2; // [nd+nk x nk x nd]
    PyArrayObject *p2_dp1dp1; // [nd x nd x nd]
    PyArrayObject *p2_dp1du1; // [nd x nu x nd]
    PyArrayObject *p2_dp1dk2; // [nd x nk x nd]
    PyArrayObject *p2_du1du1; // [nu x nu x nd]
    PyArrayObject *p2_du1dk2; // [nu x nk x nd]
    PyArrayObject *p2_dk2dk2; // [nk x nk x nd]

    PyArrayObject *l1_dq1dq1; // [nd+nk x nd+nk x nc]
    PyArrayObject *l1_dq1dp1; // [nd+nk x nd x nc]
    PyArrayObject *l1_dq1du1; // [nd+nk x nu x nc]
    PyArrayObject *l1_dq1dk2; // [nd+nk x nk x nc]
    PyArrayObject *l1_dp1dp1; // [nd x nd x nc]
    PyArrayObject *l1_dp1du1; // [nd x nu x nc]
    PyArrayObject *l1_dp1dk2; // [nd x nk x nc]
    PyArrayObject *l1_du1du1; // [nu x nu x nc]
    PyArrayObject *l1_du1dk2; // [nu x nk x nc]
    PyArrayObject *l1_dk2dk2; // [nk x nk x nc]

    PyArrayObject *DDDh1T; // [nd+nk x nd+nk x nd+nk x nc]    
    PyArrayObject *DDh2; // [nc x nd+nk x nd+nk]
    
    PyArrayObject *temp_ndnc; // [nd x nc]

    PyArrayObject *D1D1L2_D1fm2;
    PyArrayObject *D2D1L2_D2fm2;
    PyArrayObject *D1D2L2;
    PyArrayObject *D2D2L2;
    PyArrayObject *D3fm2;
    PyArrayObject *D1D1D1L2_D1D1fm2;
    PyArrayObject *D1D2D1L2_D1D2fm2;
    PyArrayObject *_D2D2D1L2_D2D2fm2;
    PyArrayObject *D1D1D2L2;
    PyArrayObject *D1D2D2L2;
    PyArrayObject *_D2D2D2L2;
    PyArrayObject *D1D3fm2;
    PyArrayObject *D2D3fm2;
    PyArrayObject *D3D3fm2;

    /* These are used in the second derivative code to convert the
     * bilinear operators into linear operators that get reused.
     * Previously this was done in mvi->temp_* but they are being
     * separated out now so that calculations can run in parallel.
     */
    PyArrayObject *dq2_dq1_op, *dl1_dq1_op, *dp2_dq1_op;
    PyArrayObject *dq2_dp1_op, *dl1_dp1_op, *dp2_dp1_op;
    PyArrayObject *dq2_du1_op, *dl1_du1_op, *dp2_du1_op;
    PyArrayObject *dq2_dk2_op, *dl1_dk2_op, *dp2_dk2_op;
    

    struct mvi_threading_s *threading;
};


/*******************************************************************************
 * System Objects
 */

// Flags for System->cache
#define SYSTEM_CACHE_NONE           0x00000000
#define SYSTEM_CACHE_LG             0x00000001
#define SYSTEM_CACHE_G              0x00000002
#define SYSTEM_CACHE_G_DQ           0x00000004
#define SYSTEM_CACHE_G_DQDQ         0x00000008
#define SYSTEM_CACHE_G_DQDQDQ       0x00000010
#define SYSTEM_CACHE_G_DQDQDQDQ     0x00000020
#define SYSTEM_CACHE_G_INV          0x00000040
#define SYSTEM_CACHE_G_INV_DQ       0x00000080
#define SYSTEM_CACHE_G_INV_DQDQ     0x00000100
#define SYSTEM_CACHE_VB             0x00000200
#define SYSTEM_CACHE_VB_DQ          0x00000400
#define SYSTEM_CACHE_VB_DQDQ        0x00000800
#define SYSTEM_CACHE_VB_DQDQDQ      0x00001000
#define SYSTEM_CACHE_VB_DDQ         0x00002000
#define SYSTEM_CACHE_VB_DDQDQ       0x00004000
#define SYSTEM_CACHE_VB_DDQDQDQ     0x00008000
#define SYSTEM_CACHE_VB_DDQDQDQDQ   0x00010000
#define SYSTEM_CACHE_DYNAMICS       0x00020000
#define SYSTEM_CACHE_DYNAMICS_DERIV1 0x00040000
#define SYSTEM_CACHE_DYNAMICS_DERIV2 0x00080000

/* The System class describes a complete mechanical system.  It
 * organizes and owns all the frames, constraints, potentials, and
 * forces in the system.  
 */ 
struct System_s {
    PyObject_HEAD

    /* current time of the system */
    double time; 
    
    /* world_frame is a pointer to the stationary world frame of the
     * system.  It is (should be) always non-NULL except for right
     * before the system is being deleted.
     */
    Frame *world_frame;

    /* frames is a tuple of all the frames in the system.  There is no
     * particular ordering.  frames is always non-NULL, but the tuple
     * may be empty.  
     */ 
    PyTupleObject *frames; // SYNC

    /* Bitflags to indicate which values have been cached.  This
     * should be cleared whenever something about the system changes
     * that would affect frame values.  (adding/removing frames,
     * changing config values, changing constant transform values,
     * etc).
     */
    unsigned long cache;

    /* This is incrmented every time the system's state changes, so
     * objects can use this to implement their own caching.
     */
    int state_counter;
    
    /* configs is a tuple of all configuration variables in the
     * system.  The Python layer is supposed to make sure that it is
     * always ordered as (dyn_configs + kin_configs).  It managed by
     * the synchronize() function.
     */
    PyTupleObject *configs; // SYNC
    PyTupleObject *dyn_configs; // Tuple of all dynamic configuration variables 
    PyTupleObject *kin_configs; // Tuple of all kinematic configuration variables
    PyTupleObject *potentials; // Tuple of all potential energies
    PyTupleObject *constraints;  // Tuple of all holonomic constraints/
    PyTupleObject *forces; // Tuple of all forces.
    PyTupleObject *inputs; // Tuple of all force inputs. 
    
    /* masses is a tuple of all the frames in the system that have
     * non-zero inertia values.  It is managed by the synchronize()
     * function.
     */
    PyTupleObject *masses;  

    /* Cached/reused values for dynamics calculations */

    // Dynamics values
    PyArrayObject *f;
    PyArrayObject *lambda;
    PyArrayObject *D;

    PyArrayObject *M_lu;
    PyArrayObject *M_lu_index;
    PyArrayObject *Ad;
    PyArrayObject *AdT;
    PyArrayObject *Ak;
    PyArrayObject *Adt;
    PyArrayObject *A_proj_lu;
    PyArrayObject *A_proj_lu_index;
    // First derivative values
    PyArrayObject *Ad_dq;
    PyArrayObject *Ak_dq;
    PyArrayObject *Adt_dq;

    PyArrayObject *D_dq, *D_ddq, *D_du, *D_dk;
    PyArrayObject *f_dq, *f_ddq, *f_du, *f_dk;
    PyArrayObject *lambda_dq, *lambda_ddq, *lambda_du, *lambda_dk;
    
    // Second derivative values
    PyArrayObject *Adt_dqdq;
    PyArrayObject *Ad_dqdq;
    PyArrayObject *Ak_dqdq;

    PyArrayObject *D_dqdq, *D_ddqdq, *D_ddqddq, *D_dkdq, *D_dudq, *D_duddq, *D_dudu;
    PyArrayObject *f_dqdq, *f_ddqdq, *f_ddqddq, *f_dkdq, *f_dudq, *f_duddq, *f_dudu;
    PyArrayObject *lambda_dqdq, *lambda_ddqdq, *lambda_ddqddq, *lambda_dkdq;
    PyArrayObject *lambda_dudq, *lambda_duddq, *lambda_dudu;

    // temps used in the middle of calculations
    PyArrayObject *temp_nd;
    PyArrayObject *temp_ndnc;

    // profiling identified a lot of time send in System_ddqddqdq*, so these were added
    // to cache and reuse the results.
    PyArrayObject *M_dq;
    PyArrayObject *M_dqdq; 
};

// Return the i-th frame in the system.
#define System_FRAME(self, i) ((Frame*)PyTuple_GET_ITEM(self->frames, i))
// Return the number of frames in the system.
#define System_FRAMES(self) PyTuple_GET_SIZE(self->frames)
// Return the i-th config in the system. 
#define System_CONFIG(self, i) ((Config*)PyTuple_GET_ITEM(self->configs, i))
// Return the number of configuration variables in the system. 
#define System_CONFIGS(self) PyTuple_GET_SIZE(self->configs)
// Return the i-th dynamic configuration variable in the system.  
#define System_DYN_CONFIG(self, i) ((Config*)PyTuple_GET_ITEM(self->dyn_configs, i))
// Return the number of dynamic configuration variables in the system.  
#define System_DYN_CONFIGS(self) PyTuple_GET_SIZE(self->dyn_configs)
// Return the i-th kinematic configuration variable in the system.  
#define System_KIN_CONFIG(self, i) ((Config*)PyTuple_GET_ITEM(self->kin_configs, i))
// Return the number of kinematic configuration variables in the system. 
#define System_KIN_CONFIGS(self) PyTuple_GET_SIZE(self->kin_configs)
// Return the i-th potential energy in the system. 
#define System_POTENTIAL(self, i) ((Potential*)PyTuple_GET_ITEM(self->potentials, i))
// Return the number of potential energies in the system. 
#define System_POTENTIALS(self) PyTuple_GET_SIZE(self->potentials)
// Return the i-th force in the system. 
#define System_FORCE(self, i) ((Force*)PyTuple_GET_ITEM(self->forces, i))
// Return the number of forces in the system.
#define System_FORCES(self) PyTuple_GET_SIZE(self->forces)
// Return the i-th input in the system. 
#define System_INPUT(self, i) ((Input*)PyTuple_GET_ITEM(self->inputs, i))
// Return the number of inputs in the system.
#define System_INPUTS(self) PyTuple_GET_SIZE(self->inputs)
// Return the i-th holonomic constraint in the system. 
#define System_CONSTRAINT(self, i) ((Constraint*)PyTuple_GET_ITEM(self->constraints, i))
// Return the number of holonomic constraints in the system.
#define System_CONSTRAINTS(self) PyTuple_GET_SIZE(self->constraints)
// Return the i-th frame with non-zero mass in the system. 
#define System_MASS(self, i) ((Frame*)PyTuple_GET_ITEM(self->masses, i))
// Return the number of frames with non-zero mass in the system. 
#define System_MASSES(self) PyTuple_GET_SIZE(self->masses)


/*******************************************************************************
 * Spline Objects
 */

struct Spline_s {
    PyObject_HEAD

    PyArrayObject *x_points;
    PyArrayObject *y_points;
    PyArrayObject *coeffs;  
};



/*******************************************************************************
 * TapeMeasure Objects
 */
struct TapeMeasure_s {
    PyObject_HEAD

    System *system;
    PyTupleObject *frames;
    PyArrayObject *seg_table;
};

#define TapeMeasure_USES_CONFIG(self, q) ( ((int*)IDX1(self->seg_table, q->index))[0] != -1)


/* Functions to safely retrieve configuration/velocity information
 * from a Frame.
 */
PYEXPORT mat4x4* Frame_lg(Frame *frame);
PYEXPORT mat4x4* Frame_lg_inv(Frame *frame);
PYEXPORT mat4x4* Frame_lg_dq(Frame *frame);
PYEXPORT mat4x4* Frame_lg_inv_dq(Frame *frame);
PYEXPORT mat4x4* Frame_lg_dqdq(Frame *frame);
PYEXPORT mat4x4* Frame_lg_inv_dqdq(Frame *frame);
PYEXPORT mat4x4* Frame_lg_dqdqdq(Frame *frame);
PYEXPORT mat4x4* Frame_lg_inv_dqdqdq(Frame *frame);
PYEXPORT mat4x4* Frame_lg_dqdqdqdq(Frame *frame);
PYEXPORT mat4x4* Frame_lg_inv_dqdqdqdq(Frame *frame);
PYEXPORT mat4x4* Frame_twist_hat(Frame *frame);
PYEXPORT mat4x4* Frame_g(Frame *frame);
PYEXPORT mat4x4* Frame_g_dq(Frame *frame, Config *q1);
PYEXPORT mat4x4* Frame_g_dqdq(Frame *frame, Config *q1, Config *q2);
PYEXPORT mat4x4* Frame_g_dqdqdq(Frame *frame, Config *q1, Config *q2, Config *q3);
PYEXPORT mat4x4* Frame_g_dqdqdqdq(Frame *frame, Config *q1, Config *q2, Config *q3, Config *q4);
PYEXPORT mat4x4* Frame_g_inv(Frame *frame);
PYEXPORT mat4x4* Frame_g_inv_dq(Frame *frame, Config *q1);
PYEXPORT mat4x4* Frame_g_inv_dqdq(Frame *frame, Config *q1, Config *q2);
PYEXPORT vec4* Frame_p(Frame *frame);
PYEXPORT vec4* Frame_p_dq(Frame *frame, Config *q1);
PYEXPORT vec4* Frame_p_dqdq(Frame *frame, Config *q1, Config *q2);
PYEXPORT vec4* Frame_p_dqdqdq(Frame *frame, Config *q1, Config *q2, Config *q3);
PYEXPORT vec4* Frame_p_dqdqdqdq(Frame *frame, Config *q1, Config *q2, Config *q3, Config *q4);
PYEXPORT mat4x4* Frame_vb(Frame *frame);
PYEXPORT mat4x4* Frame_vb_dq(Frame *frame, Config *q1);
PYEXPORT mat4x4* Frame_vb_dqdq(Frame *frame, Config *q1, Config *q2);
PYEXPORT mat4x4* Frame_vb_dqdqdq(Frame *frame, Config *q1, Config *q2, Config *q3);
PYEXPORT mat4x4* Frame_vb_ddq(Frame *frame, Config *dq1);
PYEXPORT mat4x4* Frame_vb_ddqdq(Frame *frame, Config *dq1, Config *q2);
PYEXPORT mat4x4* Frame_vb_ddqdqdq(Frame *frame, Config *dq1, Config *q2, Config *q3);
PYEXPORT mat4x4* Frame_vb_ddqdqdqdq(Frame *frame, Config *dq1, Config *q2, Config *q3, Config *q4);


PYEXPORT void copy_vec4(vec4 dest, vec4 src);
PYEXPORT void set_vec4(vec4 dest, double x, double y, double z, double w);
PYEXPORT void clear_vec4(vec4 dest);
PYEXPORT double dot_vec4(vec4 op1, vec4 op2);
PYEXPORT void sub_vec4(vec4 dest, vec4 op1, vec4 op2);
PYEXPORT void mul_mm4(mat4x4 dest, mat4x4 op1, mat4x4 op2);
PYEXPORT void mul_mv4(vec4 dest, mat4x4 m, vec4 v);
PYEXPORT void mul_dm4(mat4x4 dest, double op1, mat4x4 op2);
PYEXPORT void add_mm4(mat4x4 dest, mat4x4 op1, mat4x4 op2);
PYEXPORT void sub_mm4(mat4x4 dest, mat4x4 op1, mat4x4 op2);
PYEXPORT void copy_mat4x4(mat4x4 dest, mat4x4 src);
PYEXPORT void eye_mat4x4(mat4x4 mat);
PYEXPORT void clear_mat4x4(mat4x4 mat);
PYEXPORT void invert_se3(mat4x4 dest, mat4x4 src);
PYEXPORT void unhat(vec6 dest, mat4x4 src);

PYEXPORT PyObject* array_from_mat4x4(mat4x4 mat);
PYEXPORT PyObject* array_from_vec4(vec4 vec);



PYEXPORT double Spline_y(Spline *self, double x);
PYEXPORT double Spline_dy(Spline *self, double x);
PYEXPORT double Spline_ddy(Spline *self, double x);


PYEXPORT double TapeMeasure_length(TapeMeasure *self);
PYEXPORT double TapeMeasure_length_dq(TapeMeasure *self, Config *q1);
PYEXPORT double TapeMeasure_length_dqdq(TapeMeasure *self, Config *q1, Config *q2);
PYEXPORT double TapeMeasure_length_dqdqdq(TapeMeasure *self, Config *q1, Config *q2, Config *q3);


/*******************************************************************************
 * Misc.
 */

#define DOT_VEC3(a, b) ((a)[0]*(b)[0] + (a)[1]*(b)[1]+(a)[2]*(b)[2])


#define ATTRIBUTE_UNUSED __attribute__ ((unused))

/**********************************************************************
 * Functions/Macros fro accessing the data in Numpy arrays.
 *********************************************************************/ 

/* There are two mechanisms available in trep to access Numpy array
 * data.  The normal index routines (IDX[1234]) use Numpy's
 * PyArray_GETPTR* macros.  These can be used directly on a
 * PyArrayObject without any preparation.
 *
 * The fast index routines (F_IDX[1234]) require a declaration at
 * (DECLARE_F_IDX[1234]) beginning of a function before they are used.
 * The declaration takes the address of the array to be accessed and a
 * local alias that is passed to the F_IDX[1234].  The declaration
 * makes a local variable pointing to the array's data and a local
 * cache of the array's stride data.  These are used by F_IDX to
 * access elements of the array.  The compiler is able to optimize
 * these accesses by knowing that the stride data is not changing.  In
 * second derivatives, this can improve the speed by 2x.
 *
 * If TREP_SAFE_INDEXING is declared, all indexing methods go through
 * functions that perform bounds checking and sanity checks.
 */


#ifdef TREP_SAFE_INDEXING

ATTRIBUTE_UNUSED 
static void* IDX1(PyArrayObject *array, int i1) {
    assert(array != NULL);
    assert(PyArray_NDIM(array) >= 1);
    assert(PyArray_DIMS(array)[0] > i1);
    return PyArray_GETPTR1(array, i1);
}

ATTRIBUTE_UNUSED 
static void* IDX2(PyArrayObject *array, int i1, int i2) {
    assert(array != NULL);
    assert(PyArray_NDIM(array) >= 2);
    assert(PyArray_DIMS(array)[0] > i1);
    assert(PyArray_DIMS(array)[1] > i2);
    return PyArray_GETPTR2(array, i1, i2);
}

ATTRIBUTE_UNUSED 
static void* IDX3(PyArrayObject *array, int i1, int i2, int i3)  {
    assert(array != NULL);
    assert(PyArray_NDIM(array) >= 3);
    assert(PyArray_DIMS(array)[0] > i1);
    assert(PyArray_DIMS(array)[1] > i2);
    assert(PyArray_DIMS(array)[2] > i3);
    return PyArray_GETPTR3(array, i1, i2, i3);
}

ATTRIBUTE_UNUSED
static void* IDX4(PyArrayObject *array, int i1, int i2, int i3, int i4)  {
    assert(array != NULL);
    assert(PyArray_NDIM(array) >= 4);
    assert(PyArray_DIMS(array)[0] > i1);
    assert(PyArray_DIMS(array)[1] > i2);
    assert(PyArray_DIMS(array)[2] > i3);
    assert(PyArray_DIMS(array)[3] > i4);
    return PyArray_GETPTR4(array, i1, i2, i3, i4);
}

#define DECLARE_F_IDX1(VARIABLE, LOCAL_NAME) PyArrayObject* LOCAL_NAME = VARIABLE;                         
#define DECLARE_F_IDX2(VARIABLE, LOCAL_NAME) PyArrayObject* LOCAL_NAME = VARIABLE;                         
#define DECLARE_F_IDX3(VARIABLE, LOCAL_NAME) PyArrayObject* LOCAL_NAME = VARIABLE;                         
#define DECLARE_F_IDX4(VARIABLE, LOCAL_NAME) PyArrayObject* LOCAL_NAME = VARIABLE;                         

#define F_IDX1(LOCAL_NAME, i1)             IDX1(LOCAL_NAME, i1)
#define F_IDX2(LOCAL_NAME, i1, i2)         IDX2(LOCAL_NAME, i1, i2)
#define F_IDX3(LOCAL_NAME, i1, i2, i3)     IDX3(LOCAL_NAME, i1, i2, i3)
#define F_IDX4(LOCAL_NAME, i1, i2, i3, i4) IDX4(LOCAL_NAME, i1, i2, i3, i4)

#else  /* TREP_SAFE_INDEXING */
/* Use fast indexing */

#define IDX1(array, i1)             PyArray_GETPTR1(array, i1)
#define IDX2(array, i1, i2)         PyArray_GETPTR2(array, i1, i2)
#define IDX3(array, i1, i2, i3)     PyArray_GETPTR3(array, i1, i2, i3)
#define IDX4(array, i1, i2, i3, i4) PyArray_GETPTR4(array, i1, i2, i3, i4)

#define DECLARE_F_IDX1(VARIABLE, LOCAL_NAME)                     \
    char* LOCAL_NAME = PyArray_DATA(VARIABLE);                    \
    npy_intp LOCAL_NAME##_strides[1];                             \
    LOCAL_NAME##_strides[0] = PyArray_STRIDES(VARIABLE)[0];       

#define DECLARE_F_IDX2(VARIABLE, LOCAL_NAME)                     \
    char* LOCAL_NAME = PyArray_DATA(VARIABLE);                    \
    npy_intp LOCAL_NAME##_strides[2];                             \
    LOCAL_NAME##_strides[0] = PyArray_STRIDES(VARIABLE)[0];       \
    LOCAL_NAME##_strides[1] = PyArray_STRIDES(VARIABLE)[1];       

#define DECLARE_F_IDX3(VARIABLE, LOCAL_NAME)                     \
    char* LOCAL_NAME = PyArray_DATA(VARIABLE);                    \
    npy_intp LOCAL_NAME##_strides[3];                             \
    LOCAL_NAME##_strides[0] = PyArray_STRIDES(VARIABLE)[0];       \
    LOCAL_NAME##_strides[1] = PyArray_STRIDES(VARIABLE)[1];       \
    LOCAL_NAME##_strides[2] = PyArray_STRIDES(VARIABLE)[2];     

#define DECLARE_F_IDX4(VARIABLE, LOCAL_NAME)                     \
    char* LOCAL_NAME = PyArray_DATA(VARIABLE);                    \
    npy_intp LOCAL_NAME##_strides[4];                             \
    LOCAL_NAME##_strides[0] = PyArray_STRIDES(VARIABLE)[0];       \
    LOCAL_NAME##_strides[1] = PyArray_STRIDES(VARIABLE)[1];       \
    LOCAL_NAME##_strides[2] = PyArray_STRIDES(VARIABLE)[2];       \
    LOCAL_NAME##_strides[3] = PyArray_STRIDES(VARIABLE)[3];     

#define F_IDX1(LOCAL_NAME, i1) (LOCAL_NAME +                    \
                                (i1)*LOCAL_NAME##_strides[0])

#define F_IDX2(LOCAL_NAME, i1, i2) (LOCAL_NAME +                        \
                                    (i1)*LOCAL_NAME##_strides[0] +      \
                                    (i2)*LOCAL_NAME##_strides[1])

#define F_IDX3(LOCAL_NAME, i1, i2, i3) (LOCAL_NAME +                    \
                                        (i1)*LOCAL_NAME##_strides[0] +  \
                                        (i2)*LOCAL_NAME##_strides[1] +  \
                                        (i3)*LOCAL_NAME##_strides[2])

#define F_IDX4(LOCAL_NAME, i1, i2, i3, i4) (LOCAL_NAME +                \
                                            (i1)*LOCAL_NAME##_strides[0] + \
                                            (i2)*LOCAL_NAME##_strides[1] + \
                                            (i3)*LOCAL_NAME##_strides[2] + \
                                            (i4)*LOCAL_NAME##_strides[3])

#endif /* !TREP_SAFE_INDEXING */


#define IDX1_DBL(array, i1) (*(double*)IDX1(array, i1))
#define IDX2_DBL(array, i1, i2) (*(double*)IDX2(array, i1, i2))
#define IDX3_DBL(array, i1, i2, i3) (*(double*)IDX3(array, i1, i2, i3))
#define IDX4_DBL(array, i1, i2, i3, i4) (*(double*)IDX4(array, i1, i2, i3, i4))

#define F_IDX1_DBL(LOCAL_NAME, i1)             (*(double*)F_IDX1(LOCAL_NAME, i1))
#define F_IDX2_DBL(LOCAL_NAME, i1, i2)         (*(double*)F_IDX2(LOCAL_NAME, i1, i2))
#define F_IDX3_DBL(LOCAL_NAME, i1, i2, i3)     (*(double*)F_IDX3(LOCAL_NAME, i1, i2, i3))
#define F_IDX4_DBL(LOCAL_NAME, i1, i2, i3, i4) (*(double*)F_IDX4(LOCAL_NAME, i1, i2, i3, i4))


/*******************************************************************************
 *******************************************************************************
 *******************************************************************************
 * These are private definitions that are not exported in the C API.
 */

#ifdef TREP_MODULE

// Default doc string to indicate internal use.  Defined in _trep.c
extern char trep_internal_doc[];

extern PyObject *ConvergenceError;


/* These never need to be called outside of frame-c.c because they are
 * handled automatically by the tree caching.  However, when
 * profiling, it can be useful to force a cache to build when you know
 * it will be built in a later function so that you can figure out how
 * much time is being spent building the cache vs. in the function
 * that causes the cache to be built.
 */
void build_lg_cache(System *system);
void build_g_cache(System *system);
void build_g_dq_cache(System *system);
void build_g_dqdq_cache(System *system);
void build_g_dqdqdq_cache(System *system);
void build_g_dqdqdqdq_cache(System *system);
void build_g_inv_cache(System *system);
void build_g_inv_dq_cache(System *system);
void build_g_inv_dqdq_cache(System *system);
void build_vb_cache(System *system);
void build_vb_dq_cache(System *system);
void build_vb_dqdq_cache(System *system);
void build_vb_dqdqdq_cache(System *system);
void build_vb_ddq_cache(System *system);
void build_vb_ddqdq_cache(System *system);
void build_vb_ddqdqdq_cache(System *system);
void build_vb_ddqdqdqdq_cache(System *system);


/* Updates caching/performance values in the system. */
void System_state_changed(System *system);
double System_total_energy(System *system);
double System_L(System *system);
double System_L_dq(System *system, Config *q1);
double System_L_dqdq(System *system, Config *q1, Config *q2);
double System_L_dqdqdq(System *system, Config *q1, Config *q2, Config *q3);
double System_L_ddq(System *system, Config *dq1);
double System_L_ddqdq(System *system, Config *dq1, Config *q2);
double System_L_ddqdqdq(System *system, Config *dq1, Config *q2, Config *q3);
double System_L_ddqdqdqdq(System *system, Config *dq1, Config *q2, Config *q3, Config *q4);
double System_L_ddqddq(System *system, Config *dq1, Config *dq2);
double System_L_ddqddqdq(System *system, Config *dq1, Config *dq2, Config *q3);
double System_L_ddqddqdqdq(System *system, Config *dq1, Config *dq2, Config *q3, Config *q4);
/* Calculate the current external forcing on 'q' */
double System_F(System *system, Config *q);
double System_F_dq(System *system, Config *q, Config *q1);
double System_F_ddq(System *system, Config *q, Config *dq1);
double System_F_du(System *system, Config *q, Input *u1);
double System_F_dqdq(System *system, Config *q, Config *q1, Config *q2);
double System_F_ddqdq(System *system, Config *q, Config *dq1, Config *q2);
double System_F_ddqddq(System *system, Config *q, Config *dq1, Config *dq2);
double System_F_dudq(System *system, Config *q, Input *u1, Config *q2);
double System_F_duddq(System *system, Config *q, Input *u1, Config *dq2);
double System_F_dudu(System *system, Config *q, Input *u1, Input *u2);

/* Linear Algebra Stuff - Matrix & Vector Operations */
void copy_np_matrix(PyArrayObject *dest, PyArrayObject *src, int rows, int cols);
void transpose_np_matrix(PyArrayObject *dest, PyArrayObject *src);
void copy_vector(double  *dest, double  *src, int length);
void transpose_matrix(double **dest, double **src, int rows, int cols);
double norm_vector(double *vec, int length);
int LU_decomp(PyArrayObject *A, int n, PyArrayObject *index, double tolerance);
void LU_solve_vec(PyArrayObject *A, int n, PyArrayObject *index, double *b);
void LU_solve_mat(PyArrayObject *A, int n, PyArrayObject *index, PyArrayObject *b, int m);
/* Used to return zero values */
extern mat4x4  zero_mat4x4;
extern vec4  zero_vec4;

// Transitional functions
void mul_matvec_c_np_c(double *dest, int length, PyArrayObject *op1, double *op2, int inner);
void mul_matmat_np_np_np(PyArrayObject *dest, int rows, int cols, PyArrayObject *op1,
                         PyArrayObject *op2, int inner);


#endif /* TREP_MODULE defined */


/* If we're not building trep, include the C-API */
#ifndef TREP_MODULE 
#include "c_api.h"
#endif 


#endif /* _TREP_INTERNAL_H_*/


