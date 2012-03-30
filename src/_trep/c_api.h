
#define TREP_C_API_VERSION 0


/* Please increment TREP_C_API_VERSION whenever anything in this file
 * is changed.
 */
enum {
    capi_ForceType,
    capi_ConstraintType,
    capi_PotentialType,
    
    capi_TREP_WORLD,
    capi_TREP_TX,
    capi_TREP_TY,
    capi_TREP_TZ,
    capi_TREP_RX,
    capi_TREP_RY,
    capi_TREP_RZ,
    capi_TREP_CONST_SE3,
    
    capi_Frame_lg,
    capi_Frame_lg_inv,
    capi_Frame_lg_dq,
    capi_Frame_lg_inv_dq,
    capi_Frame_lg_dqdq,
    capi_Frame_lg_inv_dqdq,
    capi_Frame_lg_dqdqdq,
    capi_Frame_lg_inv_dqdqdq,
    capi_Frame_lg_dqdqdqdq,
    capi_Frame_lg_inv_dqdqdqdq,   
    capi_Frame_twist_hat,
    
    capi_Frame_g,
    capi_Frame_g_dq,
    capi_Frame_g_dqdq,
    capi_Frame_g_dqdqdq,
    capi_Frame_g_dqdqdqdq,
    
    capi_Frame_g_inv,
    capi_Frame_g_inv_dq,
    capi_Frame_g_inv_dqdq,
    
    capi_Frame_p,
    capi_Frame_p_dq,
    capi_Frame_p_dqdq,
    capi_Frame_p_dqdqdq,
    capi_Frame_p_dqdqdqdq,
    
    capi_Frame_vb,
    capi_Frame_vb_dq,
    capi_Frame_vb_dqdq,
    capi_Frame_vb_dqdqdq,
    capi_Frame_vb_ddq,
    capi_Frame_vb_ddqdq,
    capi_Frame_vb_ddqdqdq,
    capi_Frame_vb_ddqdqdqdq,
    
    capi_copy_vec4,
    capi_set_vec4,
    capi_clear_vec4,
    capi_dot_vec4,
    capi_sub_vec4,
    capi_mul_mm4,
    capi_mul_mv4,
    capi_mul_dm4,
    capi_add_mm4,
    capi_sub_mm4,
    capi_copy_mat4x4,
    capi_eye_mat4x4,
    capi_clear_mat4x4,
    capi_invert_se3,
    capi_unhat,
    
    capi_array_from_mat4x4,
    capi_array_from_vec4,
    
    capi_Spline_y,
    capi_Spline_dy,
    capi_Spline_ddy,
    
    capi_FrameSequence_length,
    capi_FrameSequence_length_dq,
    capi_FrameSequence_length_dqdq,
    capi_FrameSequence_length_dqdqdq,
    
    capi_size
    
} capi_index;


typedef struct {
    int size;
    int version;
    void **API;
} trep_API_def_t;


#ifndef TREP_MODULE

static void** trep_API;

static PyTypeObject *ForceType;
static PyTypeObject *ConstraintType;
static PyTypeObject *PotentialType;

static FrameTransform *TREP_WORLD;
static FrameTransform *TREP_TX;
static FrameTransform *TREP_TY;
static FrameTransform *TREP_TZ;
static FrameTransform *TREP_RX;
static FrameTransform *TREP_RY;
static FrameTransform *TREP_RZ;
static FrameTransform *TREP_CONST_SE3;


static int import_trep(void)
{
    trep_API_def_t *trep_api;
    
    trep_api = (trep_API_def_t*)PyCapsule_Import("trep._C_API", 0);
    if(trep_api == NULL)
        return -1;

    if(trep_api->size != capi_size) {
        PyErr_Format(PyExc_ImportError, "trep API has unexpected size %d (expected %d)",
                     trep_api->size, capi_size);
        return -1;
        
    }

    if(trep_api->version != TREP_C_API_VERSION) {
        PyErr_Format(PyExc_ImportError, "trep API has unexpected version %d (expected %d)",
                     trep_api->version, TREP_C_API_VERSION);
        return -1;
    }

    trep_API = trep_api->API;

    ForceType = (PyTypeObject*)trep_API[capi_ForceType];
    ConstraintType = (PyTypeObject*)trep_API[capi_ConstraintType];
    PotentialType = (PyTypeObject*)trep_API[capi_PotentialType];
    
    TREP_WORLD = (FrameTransform*)trep_API[capi_TREP_WORLD];
    TREP_TX = (FrameTransform*)trep_API[capi_TREP_TX];
    TREP_TY = (FrameTransform*)trep_API[capi_TREP_TY];
    TREP_TZ = (FrameTransform*)trep_API[capi_TREP_TZ];
    TREP_RX = (FrameTransform*)trep_API[capi_TREP_RX];
    TREP_RY = (FrameTransform*)trep_API[capi_TREP_RY];
    TREP_RZ = (FrameTransform*)trep_API[capi_TREP_RZ];
    TREP_CONST_SE3 = (FrameTransform*)trep_API[capi_TREP_CONST_SE3];

    return 0;
}
    

ATTRIBUTE_UNUSED 
static mat4x4* Frame_lg(Frame *frame)
{
    return (*(mat4x4* (*)(Frame*)) trep_API[capi_Frame_lg])(frame);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_lg_inv(Frame *frame)
{
    return (*(mat4x4* (*)(Frame*)) trep_API[capi_Frame_lg_inv])(frame);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_lg_dq(Frame *frame)
{
    return (*(mat4x4* (*)(Frame*)) trep_API[capi_Frame_lg_dq])(frame);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_lg_inv_dq(Frame *frame)
{
    return (*(mat4x4* (*)(Frame*)) trep_API[capi_Frame_lg_inv_dq])(frame);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_lg_dqdq(Frame *frame)
{
    return (*(mat4x4* (*)(Frame*)) trep_API[capi_Frame_lg_dqdq])(frame);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_lg_inv_dqdq(Frame *frame)
{
    return (*(mat4x4* (*)(Frame*)) trep_API[capi_Frame_lg_inv_dqdq])(frame);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_lg_dqdqdq(Frame *frame)
{
    return (*(mat4x4* (*)(Frame*)) trep_API[capi_Frame_lg_dqdqdq])(frame);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_lg_inv_dqdqdq(Frame *frame)
{
    return (*(mat4x4* (*)(Frame*)) trep_API[capi_Frame_lg_inv_dqdqdq])(frame);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_lg_dqdqdqdq(Frame *frame)
{
    return (*(mat4x4* (*)(Frame*)) trep_API[capi_Frame_lg_dqdqdqdq])(frame);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_lg_inv_dqdqdqdq(Frame *frame)
{
    return (*(mat4x4* (*)(Frame*)) trep_API[capi_Frame_lg_inv_dqdqdqdq])(frame);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_twist_hat(Frame *frame)
{
    return (*(mat4x4* (*)(Frame*)) trep_API[capi_Frame_twist_hat])(frame);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_g(Frame *frame)
{
    return (*(mat4x4* (*)(Frame*)) trep_API[capi_Frame_g])(frame);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_g_dq(Frame *frame, Config *q1)
{
    return (*(mat4x4* (*)(Frame*, Config*)) trep_API[capi_Frame_g_dq])(frame, q1);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_g_dqdq(Frame *frame, Config *q1, Config *q2)
{
    return (*(mat4x4* (*)(Frame*, Config*, Config*)) trep_API[capi_Frame_g_dqdq])(frame, q1, q2);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_g_dqdqdq(Frame *frame, Config *q1, Config *q2, Config *q3)
{
    return (*(mat4x4* (*)(Frame*, Config*, Config*, Config*)) trep_API[capi_Frame_g_dqdqdq])(frame, q1, q2, q3);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_g_dqdqdqdq(Frame *frame, Config *q1, Config *q2, Config *q3, Config *q4)
{
    return (*(mat4x4* (*)(Frame*, Config*, Config*, Config*, Config*)
                ) trep_API[capi_Frame_g_dqdqdqdq])(frame, q1, q2, q3, q4);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_g_inv(Frame *frame)
{
    return (*(mat4x4* (*)(Frame*)) trep_API[capi_Frame_g_inv])(frame);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_g_inv_dq(Frame *frame, Config *q1)
{
    return (*(mat4x4* (*)(Frame*, Config*)) trep_API[capi_Frame_g_inv_dq])(frame, q1);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_g_inv_dqdq(Frame *frame, Config *q1, Config *q2)
{
    return (*(mat4x4* (*)(Frame*, Config*, Config*)) trep_API[capi_Frame_g_inv_dqdq])(frame, q1, q2);
}

ATTRIBUTE_UNUSED 
static vec4* Frame_p(Frame *frame)
{
    return (*(vec4* (*)(Frame*)) trep_API[capi_Frame_p])(frame);
}

ATTRIBUTE_UNUSED 
static vec4* Frame_p_dq(Frame *frame, Config *q1)
{
    return (*(vec4* (*)(Frame*, Config*)) trep_API[capi_Frame_p_dq])(frame, q1);
}

ATTRIBUTE_UNUSED 
static vec4* Frame_p_dqdq(Frame *frame, Config *q1, Config *q2)
{
    return (*(vec4* (*)(Frame*, Config*, Config*)) trep_API[capi_Frame_p_dqdq])(frame, q1, q2);
}

ATTRIBUTE_UNUSED 
static vec4* Frame_p_dqdqdq(Frame *frame, Config *q1, Config *q2, Config *q3)
{
    return (*(vec4* (*)(Frame*, Config*, Config*, Config*)) trep_API[capi_Frame_p_dqdqdq])(frame, q1, q2, q3);
}

ATTRIBUTE_UNUSED 
static vec4* Frame_p_dqdqdqdq(Frame *frame, Config *q1, Config *q2, Config *q3, Config *q4)
{
    return (*(vec4* (*)(Frame*, Config*, Config*, Config*, Config*)
                  ) trep_API[capi_Frame_p_dqdqdqdq])(frame, q1, q2, q3, q4);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_vb(Frame *frame)
{
    return (*(mat4x4* (*)(Frame*)) trep_API[capi_Frame_vb])(frame);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_vb_dq(Frame *frame, Config *q1)
{
    return (*(mat4x4* (*)(Frame*, Config*)) trep_API[capi_Frame_vb_dq])(frame, q1);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_vb_dqdq(Frame *frame, Config *q1, Config *q2)
{
    return (*(mat4x4* (*)(Frame*, Config*, Config*)) trep_API[capi_Frame_vb_dqdq])(frame, q1, q2);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_vb_dqdqdq(Frame *frame, Config *q1, Config *q2, Config *q3)
{
    return (*(mat4x4* (*)(Frame*, Config*, Config*, Config*)
                ) trep_API[capi_Frame_vb_dqdqdq])(frame, q1, q2, q3);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_vb_ddq(Frame *frame, Config *dq1)
{
    return (*(mat4x4* (*)(Frame*, Config*)) trep_API[capi_Frame_vb_ddq])(frame, dq1);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_vb_ddqdq(Frame *frame, Config *dq1, Config *q2)
{
    return (*(mat4x4* (*)(Frame*, Config*, Config*)) trep_API[capi_Frame_vb_ddqdq])(frame, dq1, q2);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_vb_ddqdqdq(Frame *frame, Config *dq1, Config *q2, Config *q3)
{
    return (*(mat4x4* (*)(Frame*, Config*, Config*, Config*)
                ) trep_API[capi_Frame_vb_ddqdqdq])(frame, dq1, q2, q3);
}

ATTRIBUTE_UNUSED 
static mat4x4* Frame_vb_ddqdqdqdq(Frame *frame, Config *dq1, Config *q2, Config *q3, Config *q4)
{
    return (*(mat4x4* (*)(Frame*, Config*, Config*, Config*, Config*)
                ) trep_API[capi_Frame_vb_ddqdqdqdq])(frame, dq1, q2, q3, q4);
}


ATTRIBUTE_UNUSED 
static void copy_vec4(vec4 dest, vec4 src)
{
    (*(void (*)(vec4, vec4)) trep_API[capi_copy_vec4])(dest, src);
}

ATTRIBUTE_UNUSED 
static void set_vec4(vec4 dest, double x, double y, double z, double w)
{
    (*(void (*)(vec4, double, double, double, double)) trep_API[capi_set_vec4])(dest, x, y, z, w);
}

ATTRIBUTE_UNUSED 
static void clear_vec4(vec4 dest)
{
    (*(void (*)(vec4)) trep_API[capi_clear_vec4])(dest);
}

ATTRIBUTE_UNUSED 
static double dot_vec4(vec4 op1, vec4 op2)
{
    return (*(double (*)(vec4, vec4)) trep_API[capi_dot_vec4])(op1, op2);
}

ATTRIBUTE_UNUSED 
static void sub_vec4(vec4 dest, vec4 op1, vec4 op2)
{
    (*(void (*)(vec4, vec4, vec4)) trep_API[capi_sub_vec4])(dest, op1, op2);
}

ATTRIBUTE_UNUSED 
static void mul_mm4(mat4x4 dest, mat4x4 op1, mat4x4 op2)
{
    (*(void (*)(mat4x4, mat4x4, mat4x4)) trep_API[capi_mul_mm4])(dest, op1, op2);
}

ATTRIBUTE_UNUSED 
static void mul_mv4(vec4 dest, mat4x4 m, vec4 v)
{
    (*(void (*)(vec4, mat4x4, vec4)) trep_API[capi_mul_mv4])(dest, m, v);
}

ATTRIBUTE_UNUSED 
static void mul_dm4(mat4x4 dest, double op1, mat4x4 op2)
{
    (*(void (*)(mat4x4, double, mat4x4)) trep_API[capi_mul_dm4])(dest, op1, op2);
}

ATTRIBUTE_UNUSED 
static void add_mm4(mat4x4 dest, mat4x4 op1, mat4x4 op2)
{
    (*(void (*)(mat4x4, mat4x4, mat4x4)) trep_API[capi_add_mm4])(dest, op1, op2);
}

ATTRIBUTE_UNUSED 
static void sub_mm4(mat4x4 dest, mat4x4 op1, mat4x4 op2)
{
    (*(void (*)(mat4x4, mat4x4, mat4x4)) trep_API[capi_sub_mm4])(dest, op1, op2);
}

ATTRIBUTE_UNUSED 
static void copy_mat4x4(mat4x4 dest, mat4x4 src)
{
    (*(void (*)(mat4x4, mat4x4)) trep_API[capi_copy_mat4x4])(dest, src);
}

ATTRIBUTE_UNUSED 
static void eye_mat4x4(mat4x4 mat)
{
    (*(void (*)(mat4x4)) trep_API[capi_eye_mat4x4])(mat);
}

ATTRIBUTE_UNUSED 
static void clear_mat4x4(mat4x4 mat)
{
    (*(void (*)(mat4x4)) trep_API[capi_clear_mat4x4])(mat);
}

ATTRIBUTE_UNUSED 
static void invert_se3(mat4x4 dest, mat4x4 src)
{
    (*(void (*)(mat4x4, mat4x4)) trep_API[capi_invert_se3])(dest, src);
}

ATTRIBUTE_UNUSED 
static void unhat(vec6 dest, mat4x4 src)
{
    (*(void (*)(vec6, mat4x4)) trep_API[capi_unhat])(dest, src);
}

ATTRIBUTE_UNUSED 
static PyObject* array_from_mat4x4(mat4x4 mat)
{
    return (*(PyObject* (*)(mat4x4)) trep_API[capi_array_from_mat4x4])(mat);
}


ATTRIBUTE_UNUSED 
static PyObject* array_from_vec4(vec4 vec)
{
    return (*(PyObject* (*)(vec4)) trep_API[capi_array_from_vec4])(vec);
}


ATTRIBUTE_UNUSED 
double Spline_y(Spline *self, double x)
{
    return (*(double (*)(Spline*, double)) trep_API[capi_Spline_y])(self, x);
}

ATTRIBUTE_UNUSED 
double Spline_dy(Spline *self, double x)
{
    return (*(double (*)(Spline*, double)) trep_API[capi_Spline_dy])(self, x);
}

ATTRIBUTE_UNUSED 
double Spline_ddy(Spline *self, double x)
{
    return (*(double (*)(Spline*, double)) trep_API[capi_Spline_ddy])(self, x);
}

ATTRIBUTE_UNUSED 
double FrameSequence_length(FrameSequence *self)
{
    return (*(double (*)(FrameSequence*)) trep_API[capi_FrameSequence_length])(self);
}

ATTRIBUTE_UNUSED 
double FrameSequence_length_dq(FrameSequence *self, Config *q1)
{
    return (*(double (*)(FrameSequence*, Config*)) trep_API[capi_FrameSequence_length_dq])(self, q1);
}

ATTRIBUTE_UNUSED 
double FrameSequence_length_dqdq(FrameSequence *self, Config *q1, Config *q2)
{
    return (*(double (*)(FrameSequence*, Config*, Config*)
                ) trep_API[capi_FrameSequence_length_dqdq])(self, q1, q2);
}

ATTRIBUTE_UNUSED 
double FrameSequence_length_dqdqdq(FrameSequence *self, Config *q1, Config *q2, Config *q3)
{
    return (*(double (*)(FrameSequence*, Config*, Config*, Config*)
                ) trep_API[capi_FrameSequence_length_dqdqdq])(self, q1, q2, q3);
}


#endif  /* TREP_MODULE not defined. */


#ifdef TREP_MODULE
/* this indicates that we're being included in _trep.c */

static void *trep_API[capi_size];
static trep_API_def_t trep_API_def;

static PyObject* export_trep(void)
{
    int i;
    for(i = 0; i < capi_size; i++)
        trep_API[i] = NULL;

    trep_API[capi_ForceType] = &ForceType;
    trep_API[capi_PotentialType] = &PotentialType;
    trep_API[capi_ConstraintType] = &ConstraintType;
    
    trep_API[capi_TREP_WORLD] = (void*)TREP_WORLD;
    trep_API[capi_TREP_TX] = (void*)TREP_TX;
    trep_API[capi_TREP_TY] = (void*)TREP_TY;
    trep_API[capi_TREP_TZ] = (void*)TREP_TZ;
    trep_API[capi_TREP_RX] = (void*)TREP_RX;
    trep_API[capi_TREP_RY] = (void*)TREP_RY;
    trep_API[capi_TREP_RZ] = (void*)TREP_RZ;
    trep_API[capi_TREP_CONST_SE3] = (void*)TREP_CONST_SE3;
    
    trep_API[capi_Frame_lg] = (void*)Frame_lg;
    trep_API[capi_Frame_lg_inv] = (void*)Frame_lg_inv;
    trep_API[capi_Frame_lg_dq] = (void*)Frame_lg_dq;
    trep_API[capi_Frame_lg_inv_dq] = (void*)Frame_lg_inv_dq;
    trep_API[capi_Frame_lg_dqdq] = (void*)Frame_lg_dqdq;
    trep_API[capi_Frame_lg_inv_dqdq] = (void*)Frame_lg_inv_dqdq;
    trep_API[capi_Frame_lg_dqdqdq] = (void*)Frame_lg_dqdqdq;
    trep_API[capi_Frame_lg_inv_dqdqdq] = (void*)Frame_lg_inv_dqdqdq;
    trep_API[capi_Frame_lg_dqdqdqdq] = (void*)Frame_lg_dqdqdqdq;
    trep_API[capi_Frame_lg_inv_dqdqdqdq   ] = (void*)Frame_lg_inv_dqdqdqdq   ;
    trep_API[capi_Frame_twist_hat] = (void*)Frame_twist_hat;
    
    trep_API[capi_Frame_g] = (void*)Frame_g;
    trep_API[capi_Frame_g_dq] = (void*)Frame_g_dq;
    trep_API[capi_Frame_g_dqdq] = (void*)Frame_g_dqdq;
    trep_API[capi_Frame_g_dqdqdq] = (void*)Frame_g_dqdqdq;
    trep_API[capi_Frame_g_dqdqdqdq] = (void*)Frame_g_dqdqdqdq;
    
    trep_API[capi_Frame_g_inv] = (void*)Frame_g_inv;
    trep_API[capi_Frame_g_inv_dq] = (void*)Frame_g_inv_dq;
    trep_API[capi_Frame_g_inv_dqdq] = (void*)Frame_g_inv_dqdq;
    
    trep_API[capi_Frame_p] = (void*)Frame_p;
    trep_API[capi_Frame_p_dq] = (void*)Frame_p_dq;
    trep_API[capi_Frame_p_dqdq] = (void*)Frame_p_dqdq;
    trep_API[capi_Frame_p_dqdqdq] = (void*)Frame_p_dqdqdq;
    trep_API[capi_Frame_p_dqdqdqdq] = (void*)Frame_p_dqdqdqdq;
    
    trep_API[capi_Frame_vb] = (void*)Frame_vb;
    trep_API[capi_Frame_vb_dq] = (void*)Frame_vb_dq;
    trep_API[capi_Frame_vb_dqdq] = (void*)Frame_vb_dqdq;
    trep_API[capi_Frame_vb_dqdqdq] = (void*)Frame_vb_dqdqdq;
    trep_API[capi_Frame_vb_ddq] = (void*)Frame_vb_ddq;
    trep_API[capi_Frame_vb_ddqdq] = (void*)Frame_vb_ddqdq;
    trep_API[capi_Frame_vb_ddqdqdq] = (void*)Frame_vb_ddqdqdq;
    trep_API[capi_Frame_vb_ddqdqdqdq] = (void*)Frame_vb_ddqdqdqdq;
    
    trep_API[capi_copy_vec4] = (void*)copy_vec4;
    trep_API[capi_set_vec4] = (void*)set_vec4;
    trep_API[capi_clear_vec4] = (void*)clear_vec4;
    trep_API[capi_dot_vec4] = (void*)dot_vec4;
    trep_API[capi_sub_vec4] = (void*)sub_vec4;
    trep_API[capi_mul_mm4] = (void*)mul_mm4;
    trep_API[capi_mul_mv4] = (void*)mul_mv4;
    trep_API[capi_mul_dm4] = (void*)mul_dm4;
    trep_API[capi_add_mm4] = (void*)add_mm4;
    trep_API[capi_sub_mm4] = (void*)sub_mm4;
    trep_API[capi_copy_mat4x4] = (void*)copy_mat4x4;
    trep_API[capi_eye_mat4x4] = (void*)eye_mat4x4;
    trep_API[capi_clear_mat4x4] = (void*)clear_mat4x4;
    trep_API[capi_invert_se3] = (void*)invert_se3;
    trep_API[capi_unhat] = (void*)unhat;
    
    trep_API[capi_array_from_mat4x4] = (void*)array_from_mat4x4;
    trep_API[capi_array_from_vec4] = (void*)array_from_vec4;
    
    trep_API[capi_Spline_y] = (void*)Spline_y;
    trep_API[capi_Spline_dy] = (void*)Spline_dy;
    trep_API[capi_Spline_ddy] = (void*)Spline_ddy;
    
    trep_API[capi_FrameSequence_length] = (void*)FrameSequence_length;
    trep_API[capi_FrameSequence_length_dq] = (void*)FrameSequence_length_dq;
    trep_API[capi_FrameSequence_length_dqdq] = (void*)FrameSequence_length_dqdq;
    trep_API[capi_FrameSequence_length_dqdqdq] = (void*)FrameSequence_length_dqdqdq;

    trep_API_def.version = TREP_C_API_VERSION;
    trep_API_def.size = capi_size;
    trep_API_def.API = trep_API;
    
    /* Create a Capsule containing the API pointer array's address */
    return PyCapsule_New((void *)&trep_API_def, "trep._C_API", NULL);
}


#endif /* TREP_MODULE defined. */

