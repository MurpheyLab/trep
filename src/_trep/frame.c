#include <Python.h>
#include "structmember.h"
#define TREP_MODULE
#include "trep.h"
   
/* Frame_get_cache_index(frame, config) -> int
 *
 * Return the index of 'config' for the frame's cache arrays.  Returns
 * -1 if the frame is not dependent on config.
 */
static inline int Frame_get_cache_index(Frame *frame, Config *config)
{
    if(Frame_CACHE(frame, config->config_gen) == config)
	return config->config_gen;
    else
	return -1;
}

/* Functions to access a Frame's cached values.  These do not check
 * the validity of the cache contents and require the configs to be
 * ordered correctly.  
 */
static mat4x4* g_cached(Frame *frame);
static mat4x4* g_dq_cached(Frame *frame, Config *q1);
static mat4x4* g_dqdq_cached(Frame *frame, Config *q1, Config *q2);
static mat4x4* g_dqdqdq_cached(Frame *frame, Config *q1, Config *q2, Config *q3);
static mat4x4* g_dqdqdqdq_cached(Frame *frame, Config *q1, Config *q2, Config *q3, Config *q4);
static mat4x4* g_inv_cached(Frame *frame);
static mat4x4* g_inv_dq_cached(Frame *frame, Config *q1);
static mat4x4* g_inv_dqdq_cached(Frame *frame, Config *q1, Config *q2);
static vec4* p_dq_cached(Frame *frame, Config *q1);
static vec4* p_dqdq_cached(Frame *frame, Config *q1, Config *q2);
static vec4* p_dqdqdq_cached(Frame *frame, Config *q1, Config *q2, Config *q3);
static vec4* p_dqdqdqdq_cached(Frame *frame, Config *q1, Config *q2, Config *q3, Config *q4);
static mat4x4* vb_cached(Frame *frame);
static mat4x4* vb_dq_cached(Frame *frame, Config *q1);
static mat4x4* vb_dqdq_cached(Frame *frame, Config *q1, Config *q2);
static mat4x4* vb_dqdqdq_cached(Frame *frame, Config *q1, Config *q2, Config *q3);
static mat4x4* vb_ddq_cached(Frame *frame, Config *dq1);
static mat4x4* vb_ddqdq_cached(Frame *frame, Config *dq1, Config *q2);
static mat4x4* vb_ddqdqdq_cached(Frame *frame, Config *dq1, Config *q2, Config *q3);
static mat4x4* vb_ddqdqdqdq_cached(Frame *frame, Config *q1, Config *q2, Config *q3, Config *q4);

#define ISNONE(obj) ( (PyObject*)(obj) == Py_None )

/* Macros to access frame data in the numpy arrays. */
#define G_DQ(i1)                   ((vec4*)IDX1(frame->g_dq, i1))
#define G_DQDQ(i1, i2)             ((vec4*)IDX2(frame->g_dqdq, i1, i2))
#define G_DQDQDQ(i1, i2, i3)       ((vec4*)IDX3(frame->g_dqdqdq, i1, i2, i3))
#define G_DQDQDQDQ(i1, i2, i3, i4) ((vec4*)IDX4(frame->g_dqdqdqdq, i1, i2, i3, i4))
#define G_INV_DQ(i1)               ((vec4*)IDX1(frame->g_inv_dq, i1))
#define G_INV_DQDQ(i1, i2)         ((vec4*)IDX2(frame->g_inv_dqdq, i1, i2))

#define P_DQ(i1)                   ((double*)IDX1(frame->p_dq, i1))
#define P_DQDQ(i1, i2)             ((double*)IDX2(frame->p_dqdq, i1, i2))
#define P_DQDQDQ(i1, i2, i3)       ((double*)IDX3(frame->p_dqdqdq, i1, i2, i3))
#define P_DQDQDQDQ(i1, i2, i3, i4) ((double*)IDX4(frame->p_dqdqdqdq, i1, i2, i3, i4))

#define VB_DQ(i1)                    ((vec4*)IDX1(frame->vb_dq, i1))
#define VB_DQDQ(i1, i2)              ((vec4*)IDX2(frame->vb_dqdq, i1, i2))
#define VB_DQDQDQ(i1, i2, i3)        ((vec4*)IDX3(frame->vb_dqdqdq, i1, i2, i3))
#define VB_DDQ(i1)                   ((vec4*)IDX1(frame->vb_ddq, i1))
#define VB_DDQDQ(i1, i2)             ((vec4*)IDX2(frame->vb_ddqdq, i1, i2))
#define VB_DDQDQDQ(i1, i2, i3)       ((vec4*)IDX3(frame->vb_ddqdqdq, i1, i2, i3))
#define VB_DDQDQDQDQ(i1, i2, i3, i4) ((vec4*)IDX4(frame->vb_ddqdqdqdq, i1, i2, i3, i4))


/////////////////////////////////////
// multiply and sandwich functions //
/////////////////////////////////////

void tx_multiply_gk(Frame *self, mat4x4 dest, mat4x4 X, int n)
{
    assert(n >= 0);
    double x = 0.0; 

    switch(n) {
    case 0:
	if(ISNONE(self->config))
	    x = self->value;
	else
	    x = self->config->q;
	dest[0][0] = X[0][0];
	dest[0][1] = X[0][1];
	dest[0][2] = X[0][2];
	dest[0][3] = X[0][3] + X[0][0]*x;
	dest[1][0] = X[1][0];
	dest[1][1] = X[1][1];
	dest[1][2] = X[1][2];
	dest[1][3] = X[1][3] + X[1][0]*x;
	dest[2][0] = X[2][0];
	dest[2][1] = X[2][1];
	dest[2][2] = X[2][2];
	dest[2][3] = X[2][3] + X[2][0]*x;
	dest[3][0] = 0.0;
	dest[3][1] = 0.0;
	dest[3][2] = 0.0;
	dest[3][3] = X[3][3];
	break;
    case 1:
	dest[0][0] = 0.0;
	dest[0][1] = 0.0;
	dest[0][2] = 0.0;
	dest[0][3] = X[0][0];
	dest[1][0] = 0.0;
	dest[1][1] = 0.0;
	dest[1][2] = 0.0;
	dest[1][3] = X[1][0];
	dest[2][0] = 0.0;
	dest[2][1] = 0.0;
	dest[2][2] = 0.0;
	dest[2][3] = X[2][0];
	dest[3][0] = 0.0;
	dest[3][1] = 0.0;
	dest[3][2] = 0.0;
	dest[3][3] = 0.0;
	break;
    default:
	clear_mat4x4(dest);
	break;
    }
}    

void ty_multiply_gk(Frame *self, mat4x4 dest, mat4x4 X, int n)
{
    assert(n >= 0);
    double x = 0.0; 

    switch(n) {
    case 0:
	if(ISNONE(self->config))
	    x = self->value;
	else
	    x = self->config->q;
	dest[0][0] = X[0][0];
	dest[0][1] = X[0][1];
	dest[0][2] = X[0][2];
	dest[0][3] = X[0][3] + X[0][1]*x;
	dest[1][0] = X[1][0];
	dest[1][1] = X[1][1];
	dest[1][2] = X[1][2];
	dest[1][3] = X[1][3] + X[1][1]*x;
	dest[2][0] = X[2][0];
	dest[2][1] = X[2][1];
	dest[2][2] = X[2][2];
	dest[2][3] = X[2][3] + X[2][1]*x;
	dest[3][0] = 0.0;
	dest[3][1] = 0.0;
	dest[3][2] = 0.0;
	dest[3][3] = X[3][3];
	break;
    case 1:
	dest[0][0] = 0.0;
	dest[0][1] = 0.0;
	dest[0][2] = 0.0;
	dest[0][3] = X[0][1];
	dest[1][0] = 0.0;
	dest[1][1] = 0.0;
	dest[1][2] = 0.0;
	dest[1][3] = X[1][1];
	dest[2][0] = 0.0;
	dest[2][1] = 0.0;
	dest[2][2] = 0.0;
	dest[2][3] = X[2][1];
	dest[3][0] = 0.0;
	dest[3][1] = 0.0;
	dest[3][2] = 0.0;
	dest[3][3] = 0.0;
	break;
    default:
	clear_mat4x4(dest);
	break;
    }
}    


void tz_multiply_gk(Frame *self, mat4x4 dest, mat4x4 X, int n)
{
    assert(n >= 0);
    double x = 0.0; 

    switch(n) {
    case 0:
	if(ISNONE(self->config))
	    x = self->value;
	else
	    x = self->config->q;
	dest[0][0] = X[0][0];
	dest[0][1] = X[0][1];
	dest[0][2] = X[0][2];
	dest[0][3] = X[0][3] + X[0][2]*x;
	dest[1][0] = X[1][0];
	dest[1][1] = X[1][1];
	dest[1][2] = X[1][2];
	dest[1][3] = X[1][3] + X[1][2]*x;
	dest[2][0] = X[2][0];
	dest[2][1] = X[2][1];
	dest[2][2] = X[2][2];
	dest[2][3] = X[2][3] + X[2][2]*x;
	dest[3][0] = 0.0;
	dest[3][1] = 0.0;
	dest[3][2] = 0.0;
	dest[3][3] = X[3][3];
	break;
    case 1:
	dest[0][0] = 0.0;
	dest[0][1] = 0.0;
	dest[0][2] = 0.0;
	dest[0][3] = X[0][2];
	dest[1][0] = 0.0;
	dest[1][1] = 0.0;
	dest[1][2] = 0.0;
	dest[1][3] = X[1][2];
	dest[2][0] = 0.0;
	dest[2][1] = 0.0;
	dest[2][2] = 0.0;
	dest[2][3] = X[2][2];
	dest[3][0] = 0.0;
	dest[3][1] = 0.0;
	dest[3][2] = 0.0;
	dest[3][3] = 0.0;
	break;
    default:
	clear_mat4x4(dest);
	break;
    }
}    

void rx_multiply_gk(Frame *self, mat4x4 dest, mat4x4 X, int n)
{
    double x = 0.0;
    double x1, x2;

    if(ISNONE(self->config))
	x = self->value;
    else
	x = self->config->q;
    if(n == 0) {
	x1 = cos(x);
	x2 = -sin(x);
	dest[0][0] = X[0][0];
	dest[0][1] = X[0][1]*x1 - X[0][2]*x2;
	dest[0][2] = X[0][1]*x2 + X[0][2]*x1;
	dest[0][3] = X[0][3];
	dest[1][0] = X[1][0];
	dest[1][1] = X[1][1]*x1 - X[1][2]*x2;
	dest[1][2] = X[1][1]*x2 + X[1][2]*x1;
	dest[1][3] = X[1][3];
	dest[2][0] = X[2][0];
	dest[2][1] = X[2][1]*x1 - X[2][2]*x2;
	dest[2][2] = X[2][1]*x2 + X[2][2]*x1;
	dest[2][3] = X[2][3];
	dest[3][0] = 0.0;
	dest[3][1] = 0.0;
	dest[3][2] = 0.0;
	dest[3][3] = X[3][3];
	return;
    }

    if(n == 1) {
	x1 = -sin(x);
	x2 = -cos(x);
    }
    else if(n == 2) {
	x1 = -cos(x);
	x2 = sin(x);
    }
    else if(n == 3) {
	x1 = sin(x);
	x2 = cos(x);
    }
    else if(n == 4) {
	x1 = cos(x);
	x2 = -sin(x);
    }
    else {
	// This is untested.
	rx_multiply_gk(self, dest, X, 1+(n-1)%4);
	return;
    }
    
    dest[0][0] = 0.0;
    dest[0][1] = X[0][1]*x1 - X[0][2]*x2;
    dest[0][2] = X[0][1]*x2 + X[0][2]*x1;
    dest[0][3] = 0.0;
    dest[1][0] = 0.0;
    dest[1][1] = X[1][1]*x1 - X[1][2]*x2;
    dest[1][2] = X[1][1]*x2 + X[1][2]*x1;
    dest[1][3] = 0.0;
    dest[2][0] = 0.0;
    dest[2][1] = X[2][1]*x1 - X[2][2]*x2;
    dest[2][2] = X[2][1]*x2 + X[2][2]*x1;
    dest[2][3] = 0.0;
    dest[3][0] = 0.0;
    dest[3][1] = 0.0;
    dest[3][2] = 0.0;
    dest[3][3] = 0.0;
}

void ry_multiply_gk(Frame *self, mat4x4 dest, mat4x4 X, int n)
{
    double x, x1, x2;

    if(ISNONE(self->config))
	x = self->value;
    else
	x = self->config->q;
    if(n == 0) {
	x1 = cos(x);
	x2 = sin(x);
	dest[0][0] = X[0][0]*x1 - X[0][2]*x2;
	dest[0][1] = X[0][1];
	dest[0][2] = X[0][0]*x2 + X[0][2]*x1;
	dest[0][3] = X[0][3];
	dest[1][0] = X[1][0]*x1 - X[1][2]*x2;
	dest[1][1] = X[1][1];
	dest[1][2] = X[1][0]*x2 + X[1][2]*x1;
	dest[1][3] = X[1][3];
	dest[2][0] = X[2][0]*x1 - X[2][2]*x2;
	dest[2][1] = X[2][1];
	dest[2][2] = X[2][0]*x2 + X[2][2]*x1; 
	dest[2][3] = X[2][3];
	dest[3][0] = 0.0;
	dest[3][1] = 0.0;
	dest[3][2] = 0.0;
	dest[3][3] = X[3][3];
	return;
    }
    if(n == 1) {
	x1 = -sin(x);
	x2 = cos(x);
    }
    else if(n == 2) {
	x1 = -cos(x);
	x2 = -sin(x);
    }
    else if(n == 3) {
	x1 = sin(x);
	x2 = -cos(x);
    }
    else if(n == 4) {
	x1 = cos(x);
	x2 = sin(x);
    }
    else {
	// This is untested.
	rx_multiply_gk(self, dest, X, 1+(n-1)%4);
	return;
    }
    dest[0][0] = X[0][0]*x1 - X[0][2]*x2;
    dest[0][1] = 0.0;
    dest[0][2] = X[0][0]*x2 + X[0][2]*x1;
    dest[0][3] = 0.0;
    dest[1][0] = X[1][0]*x1 - X[1][2]*x2;
    dest[1][1] = 0.0;
    dest[1][2] = X[1][0]*x2 + X[1][2]*x1;
    dest[1][3] = 0.0;
    dest[2][0] = X[2][0]*x1 - X[2][2]*x2;
    dest[2][1] = 0.0;
    dest[2][2] = X[2][0]*x2 + X[2][2]*x1; 
    dest[2][3] = 0.0;
    dest[3][0] = 0.0;
    dest[3][1] = 0.0;
    dest[3][2] = 0.0;
    dest[3][3] = 0.0;
}

void rz_multiply_gk(Frame *self, mat4x4 dest, mat4x4 X, int n)
{
    double x, x1, x2;
    if(ISNONE(self->config))
	x = self->value;
    else
	x = self->config->q;
    if(n == 0) {
	x1 = cos(x);
	x2 = -sin(x);
	dest[0][0] = X[0][0]*x1 - X[0][1]*x2;
	dest[0][1] = X[0][0]*x2 + X[0][1]*x1;
	dest[0][2] = X[0][2];
	dest[0][3] = X[0][3];
	dest[1][0] = X[1][0]*x1 - X[1][1]*x2;
	dest[1][1] = X[1][0]*x2 + X[1][1]*x1;
	dest[1][2] = X[1][2];
	dest[1][3] = X[1][3];
	dest[2][0] = X[2][0]*x1 - X[2][1]*x2;
	dest[2][1] = X[2][0]*x2 + X[2][1]*x1;
	dest[2][2] = X[2][2];
	dest[2][3] = X[2][3];
	dest[3][0] = 0.0;
	dest[3][1] = 0.0;
	dest[3][2] = 0.0;
	dest[3][3] = X[3][3];
	return;
    }
    if(n == 1) {
	x1 = -sin(x);
	x2 = -cos(x);
    }
    else if(n == 2) {
	x1 = -cos(x);
	x2 = sin(x);
    }
    else if(n == 3) {
	x1 = sin(x);
	x2 = cos(x);
    }
    else if(n == 4) {
	x1 = cos(x);
	x2 = -sin(x);
    }
    else {
	// This is untested.
	rx_multiply_gk(self, dest, X, 1+(n-1)%4);
	return;
    }
    dest[0][0] = X[0][0]*x1 - X[0][1]*x2;
    dest[0][1] = X[0][0]*x2 + X[0][1]*x1;
    dest[0][2] = 0.0;
    dest[0][3] = 0.0;
    dest[1][0] = X[1][0]*x1 - X[1][1]*x2;
    dest[1][1] = X[1][0]*x2 + X[1][1]*x1;
    dest[1][2] = 0.0;
    dest[1][3] = 0.0;
    dest[2][0] = X[2][0]*x1 - X[2][1]*x2;
    dest[2][1] = X[2][0]*x2 + X[2][1]*x1;
    dest[2][2] = 0.0;
    dest[2][3] = 0.0;
    dest[3][0] = 0.0;
    dest[3][1] = 0.0;
    dest[3][2] = 0.0;
    dest[3][3] = 0.0;
}

void const_se3_multiply_gk(Frame *self, mat4x4 dest, mat4x4 X, int n)
{
    switch(n) {
    case 0:
	mul_mm4(dest, X, self->lg);
	break;
    default:
	clear_mat4x4(dest);
	break;
    }
}

void tx_add_sandwich_gk(Frame *self, mat4x4 dest, mat4x4 X, int n1, int n2)
{
    assert(n1 <= n2);
    double x;

    if(n1 == 0 && n2 == 0) {
	if(ISNONE(self->config))
	    x = self->value;
	else
	    x = self->config->q;

	dest[0][1] += X[0][1];
	dest[0][2] += X[0][2];
	dest[0][3] += X[0][3];
	dest[1][0] -= X[0][1];
	dest[1][2] += X[1][2];
	dest[1][3] += X[1][3] - X[0][1]*x;
	dest[2][0] -= X[0][2];
	dest[2][1] -= X[1][2];
	dest[2][3] += X[2][3] - X[0][2]*x;
	return;
    }
    if(n1 == 0 && n2 == 1) {
	// THIS CASE IS NOT CURRENTLY COVERED BY THE FRAMES TEST!
	dest[1][3] += X[1][0];
	dest[2][3] += X[2][0];
	return;
    }
    // All other cases are zero.
}
    
void ty_add_sandwich_gk(Frame *self, mat4x4 dest, mat4x4 X, int n1, int n2)
{
    assert(n1 <= n2);
    double y;

    if(n1 == 0 && n2 == 0) {
	if(ISNONE(self->config))
	    y = self->value;
	else
	    y = self->config->q;

	dest[0][1] += X[0][1];
	dest[0][2] += X[0][2];
	dest[0][3] += X[0][3] + X[0][1]*y;
	dest[1][0] -= X[0][1];
	dest[1][2] += X[1][2];
	dest[1][3] += X[1][3];
	dest[2][0] -= X[0][2];
	dest[2][1] -= X[1][2];
	dest[2][3] += X[2][3] - X[1][2]*y;
	return;
    }
    if(n1 == 0 && n2 == 1) {
	// THIS CASE IS NOT CURRENTLY COVERED BY THE FRAMES TEST!
	dest[0][3] += X[0][1];
	dest[2][3] += X[2][1];
	return;
    }
    // All other cases are zero.
}

void tz_add_sandwich_gk(Frame *self, mat4x4 dest, mat4x4 X, int n1, int n2) 
{
    assert(n1 <= n2);
    double z;

    if(n1 == 0 && n2 == 0) {
	if(ISNONE(self->config))
	    z = self->value;
	else
	    z = self->config->q;

	dest[0][1] += X[0][1];
	dest[0][2] += X[0][2];
	dest[0][3] += X[0][3] + X[0][2]*z;
	dest[1][0] -= X[0][1];
	dest[1][2] += X[1][2];
	dest[1][3] += X[1][3] + X[1][2]*z;
	dest[2][0] -= X[0][2];
	dest[2][1] -= X[1][2];
	dest[2][3] += X[2][3];
	return;
    }
    if(n1 == 0 && n2 == 1) {
	// THIS CASE IS NOT CURRENTLY COVERED BY THE FRAMES TEST!
	dest[0][3] += X[0][2];
	dest[1][3] += X[1][2];
	return;
    }
    // All other cases are zero.
}

void rx_add_sandwich_gk(Frame *self, mat4x4 dest, mat4x4 X, int n1, int n2) 
{
    double x, cx, sx;
    double val;

    assert(n1 <= n2);

    if(n1 == 0 && n2 == 0) {
	if(ISNONE(self->config))
	    x = self->value;
	else
	    x = self->config->q;
	sx = sin(x);
	cx = cos(x);

	// Using val to take advantage of symmetry
	val = X[0][2]*sx + X[0][1]*cx;
	dest[0][1] += val;
	dest[1][0] -= val;
	val = X[0][2]*cx - X[0][1]*sx;
	dest[0][2] += val;
	dest[2][0] -= val;
	val = X[1][2];
	dest[1][2] += val;
	dest[2][1] -= val;

	dest[0][3] += X[0][3];
	dest[1][3] += X[1][3]*cx + X[2][3]*sx;
	dest[2][3] += -X[1][3]*sx + X[2][3]*cx;
	return;
    }

    if(n1 == 0 && n2 == 1) {
	if(ISNONE(self->config))
	    x = self->value;
	else
	    x = self->config->q;
	sx = sin(x);
	cx = cos(x);

	// Using val to take advantage of symmetry
	val = X[0][2]*cx + X[1][0]*sx;
	dest[0][1] += val;
	dest[1][0] -= val;
	val = X[1][0]*cx - X[0][2]*sx;
	dest[0][2] += val;
	dest[2][0] -= val;

	dest[1][3] += X[2][3]*cx - X[1][3]*sx;
	dest[2][3] += -X[1][3]*cx - X[2][3]*sx;
	return;
    }
    if(n1 == 0 && n2 == 2) {
	if(ISNONE(self->config))
	    x = self->value;
	else
	    x = self->config->q;
	sx = sin(x);
	cx = cos(x);

	// Using val to take advantage of symmetry
	val = X[1][0]*cx + X[2][0]*sx;
	dest[0][1] += val;
	dest[1][0] -= val;
	val = X[2][0]*cx - X[1][0]*sx;
	dest[0][2] += val;
	dest[2][0] -= val;
	val = 2*X[2][1];
	dest[1][2] += val;
	dest[2][1] -= val;

	dest[1][3] += -X[1][3]*cx - X[2][3]*sx;
	dest[2][3] += -X[2][3]*cx + X[1][3]*sx;
	return;
    }
    if(n1 == 0 && n2 == 3) {
	if(ISNONE(self->config))
	    x = self->value;
	else
	    x = self->config->q;
	sx = sin(x);
	cx = cos(x);

	// Using val to take advantage of symmetry
	val = X[2][0]*cx+X[0][1]*sx;
	dest[0][1] += val;
	dest[1][0] -= val;
	val = X[0][1]*cx-X[2][0]*sx;
	dest[0][2] += val;
	dest[2][0] -= val;

	dest[1][3] += -X[2][3]*cx + X[1][3]*sx;
	dest[2][3] += X[1][3]*cx+X[2][3]*sx;
	return;
    }
    if(n1 == 1 && n2 == 1) {
	if(ISNONE(self->config))
	    x = self->value;
	else
	    x = self->config->q;
	sx = sin(x);
	cx = cos(x);

	// Using val to take advantage of symmetry
	val = X[1][2];
	dest[1][2] += val;
	dest[2][1] -= val;
	return;
    }
    if(n1 == 1 && n2 == 2) {
	// All zeros
	return;
    }
    // OTHER CASES ARE NOT HANDLED.
    assert(0);
}

void ry_add_sandwich_gk(Frame *self, mat4x4 dest, mat4x4 X, int n1, int n2) 
{
    assert(n1 <= n2);
    double y, cy, sy;
    double val;

    if(ISNONE(self->config))
	y = self->value;
    else
	y = self->config->q;
    sy = sin(y);
    cy = cos(y);
    
    if(n1 == 0 && n2 == 0) {
	// Using val to take advantage of symmetry
	val = X[0][1]*cy + X[1][2]*sy;
	dest[0][1] += val;
	dest[1][0] -= val;
	val = X[0][2];
	dest[0][2] += val;
	dest[2][0] -= val;
	val = X[1][2]*cy - X[0][1]*sy;
	dest[1][2] += val;
	dest[2][1] -= val;
	
	dest[0][3] += X[0][3]*cy - X[2][3]*sy;
	dest[1][3] += X[1][3];
	dest[2][3] += X[2][3]*cy + X[0][3]*sy;
	return;
    }

    if(n1 == 0 && n2 == 1) {
	// Using val to take advantage of symmetry
	val = X[1][2]*cy + X[1][0]*sy;
	dest[0][1] += val;
	dest[1][0] -= val;
	val = X[1][0]*cy - X[1][2]*sy;
	dest[1][2] += val;
	dest[2][1] -= val; 

	dest[0][3] += -X[2][3]*cy - X[0][3]*sy;
	dest[2][3] += X[0][3]*cy - X[2][3]*sy;
	return;
    }

    if(n1 == 0 && n2 == 2) {
	// Using val to take advantage of symmetry
	val = X[1][0]*cy + X[2][1]*sy;
	dest[0][1] += val;
	dest[1][0] -= val;
	val = 2*X[2][0];
	dest[0][2] += val;
	dest[2][0] -= val;
	val = X[2][1]*cy - X[1][0]*sy;
	dest[1][2] += val;
	dest[2][1] -= val;

	dest[0][3] += -X[0][3]*cy + X[2][3]*sy;
	dest[2][3] += -X[2][3]*cy - X[0][3]*sy;
	return;
    }

    if(n1 == 0 && n2 == 3) {
	// Using val to take advantage of symmetry
	val = X[2][1]*cy + X[0][1]*sy;
	dest[0][1] += val;
	dest[1][0] -= val;
	val = X[0][1]*cy - X[2][1]*sy;
	dest[1][2] += val;
	dest[2][1] -= val;

	dest[0][3] += X[2][3]*cy + X[0][3]*sy;
	dest[2][3] += -X[0][3]*cy + X[2][3]*sy;
	return;
    }

    if(n1 == 1 && n2 == 1) {
	// Using val to take advantage of symmetry
	val = X[0][2];
	dest[0][2] += val;
	dest[2][0] -= val;
	return;
    }

    if(n1 == 1 && n2 == 2) {
	// All zeros.
	return;
    }

    // OTHER CASES ARE NOT HANDLED.
    assert(0);
}

void rz_add_sandwich_gk(Frame *self, mat4x4 dest, mat4x4 X, int n1, int n2) 
{
    double z, cz, sz;
    double val;
    assert(n1 <= n2);

    if(self->config == NULL)
	z = self->value;
    else
	z = self->config->q;
    sz = sin(z);
    cz = cos(z);
    
    if(n1 == 0 && n2 == 0) {
	// Using val to take advantage of symmetry
	val = X[0][1];
	dest[0][1] += val;
	dest[1][0] -= val;
	val = X[0][2]*cz + X[1][2]*sz;
	dest[0][2] += val;
	dest[2][0] -= val;
	val = X[1][2]*cz - X[0][2]*sz;
	dest[1][2] += val;
	dest[2][1] -= val;

	dest[0][3] += X[0][3]*cz + X[1][3]*sz;
	dest[1][3] += X[1][3]*cz - X[0][3]*sz;
	dest[2][3] += X[2][3];
	return;
    }

    if(n1 == 0 && n2 == 1) {
	// Using val to take advantage of symmetry
	val = X[1][2]*cz + X[2][0]*sz;
	dest[0][2] += val;
	dest[2][0] -= val;
	val = X[2][0]*cz - X[1][2]*sz;
	dest[1][2] += val;
	dest[2][1] -= val;

	dest[0][3] += X[1][3]*cz - X[0][3]*sz;
	dest[1][3] += -X[0][3]*cz - X[1][3]*sz;
	return;
    }

    if(n1 == 0 && n2 == 2) {
	// Using val to take advantage of symmetry
	val = 2*X[1][0];
	dest[0][1] += val;
	dest[1][0] -= val;
	val = X[2][0]*cz + X[2][1]*sz;
	dest[0][2] += val;
	dest[2][0] -= val;
	val = X[2][1]*cz - X[2][0]*sz;
	dest[1][2] += val;
	dest[2][1] -= val;

	dest[0][3] += -X[0][3]*cz - X[1][3]*sz;
	dest[1][3] += -X[1][3]*cz + X[0][3]*sz;
	return;
    }

    if(n1 == 0 && n2 == 3) {
	// Using val to take advantage of symmetry
	val = X[2][1]*cz + X[0][2]*sz;
	dest[0][2] += val;
	dest[2][0] -= val;
	val = X[0][2]*cz - X[2][1]*sz;
	dest[1][2] += val;
	dest[2][1] -= val;

	dest[0][3] += -X[1][3]*cz + X[0][3]*sz;
	dest[1][3] += X[0][3]*cz + X[1][3]*sz;
	return;
    }

    if(n1 == 1 && n2 == 1) {
	// Using val to take advantage of symmetry
	val = X[0][1];
	dest[0][1] += val;
	dest[1][0] -= val;
	return;
    }

    if(n1 == 1 && n2 == 2) {
	// All zeros.
	return;
    }
       
    // OTHER CASES ARE NOT HANDLED.
    //assert(0);
}

void const_se3_add_sandwich_gk(Frame *self, mat4x4 dest, mat4x4 X, int n1, int n2) 
{
    assert(n1 <= n2);
    mat4x4 t1, t2;

    // This could still be optimized a little bit.
    if(n1 == 0 && n2 == 0) {
	mul_mm4(t1, X, self->lg);
	mul_mm4(t2, self->lg_inv, t1);
	add_mm4(dest, dest, t2);
	return;
    }
    // All other cases are zero.
}



//////////////////////////////
// Cache building functions //
//////////////////////////////

static void build_lg_cache_int(Frame *frame)
{
    int i;
    double x = 0.0;

    if(ISNONE(frame->config))
	x = frame->value;
    else
	x = frame->config->q;

    if(frame->transform == TREP_WORLD) {
    }
    else if(frame->transform == TREP_TX) {	
	frame->lg[0][3] = x;
	// TX(x)^-1 = TX(-x)
	frame->lg_inv[0][3] = -x;

	frame->lg_dq[0][3] = 1;
	// TX(x)^-1' = TX(-x)' = TX'(-x)(-1) = -TX'(-x)
	frame->lg_inv_dq[0][3] =-1;

	// TX''(x) = TX(x)^-1'' = 0
	// TX'''(x) = TX(x)^-1''' = 0
	// TX''''(x) = TX(x)^-1'''' = 0

	frame->twist_hat[0][3] = 1;
    }
    else if(frame->transform == TREP_TY) {
	frame->lg[1][3] = x;
	// TY(x)^-1 = TY(-x)
	frame->lg_inv[1][3] = -x;

	frame->lg_dq[1][3] = 1;	
	// TY(x)^-1' = TY(-x)' = TY'(-x)(-1) = -TY'(-x)
	frame->lg_inv_dq[1][3] =-1;

	// TY''(x) = TY(x)^-1'' = 0
	// TY'''(x) = TY(x)^-1''' = 0
	// TY''''(x) = TY(x)^-1'''' = 0

	frame->twist_hat[1][3] = 1;
    }
    else if(frame->transform == TREP_TZ) {
	frame->lg[2][3] = x;
	// TZ(x)^-1 = TZ(-x)
	frame->lg_inv[2][3] = -x;

	frame->lg_dq[2][3] = 1;
	// TZ(x)^-1' = TZ(-x)' = TZ'(-x)(-1) = -TZ'(-x)
	frame->lg_inv_dq[2][3] =-1;

	// TZ''(x) = TZ(x)^-1'' = 0
	// TZ'''(x) = TZ(x)^-1''' = 0
	// TZ'''(x) = TZ(x)^-1'''' = 0

	frame->twist_hat[2][3] = 1;
    }
    else if(frame->transform == TREP_RX) {
	frame->lg[1][1] = cos(x);
	frame->lg[1][2] =-sin(x);
	frame->lg[2][1] = sin(x);
	frame->lg[2][2] = cos(x);
	// RX(x)^-1 = RX(-x)
	frame->lg_inv[1][1] = cos(-x);
	frame->lg_inv[1][2] =-sin(-x);
	frame->lg_inv[2][1] = sin(-x);
	frame->lg_inv[2][2] = cos(-x);

	frame->lg_dq[1][1] =-sin(x);
	frame->lg_dq[1][2] =-cos(x);
	frame->lg_dq[2][1] = cos(x);
	frame->lg_dq[2][2] =-sin(x);
	// RX(x)^-1' =  -RX'(-x)
	frame->lg_inv_dq[1][1] = sin(-x);
	frame->lg_inv_dq[1][2] = cos(-x);
	frame->lg_inv_dq[2][1] =-cos(-x);
	frame->lg_inv_dq[2][2] = sin(-x);

	frame->lg_dqdq[1][1] =-cos(x);
	frame->lg_dqdq[1][2] = sin(x);
	frame->lg_dqdq[2][1] =-sin(x);
	frame->lg_dqdq[2][2] =-cos(x);
	// RX(x)^-1'' = RX''(-x)
	frame->lg_inv_dqdq[1][1] =-cos(-x);
	frame->lg_inv_dqdq[1][2] = sin(-x);
	frame->lg_inv_dqdq[2][1] =-sin(-x);
	frame->lg_inv_dqdq[2][2] =-cos(-x);

	frame->lg_dqdqdq[1][1] = sin(x);
	frame->lg_dqdqdq[1][2] = cos(x);
	frame->lg_dqdqdq[2][1] =-cos(x);
	frame->lg_dqdqdq[2][2] = sin(x);
	// RX(x)^-1''' = -RX'''(-x)
	frame->lg_inv_dqdqdq[1][1] =-sin(-x);
	frame->lg_inv_dqdqdq[1][2] =-cos(-x);
	frame->lg_inv_dqdqdq[2][1] = cos(-x);
	frame->lg_inv_dqdqdq[2][2] =-sin(-x);

	frame->lg_dqdqdqdq[1][1] = cos(x);
	frame->lg_dqdqdqdq[1][2] =-sin(x);
	frame->lg_dqdqdqdq[2][1] = sin(x);
	frame->lg_dqdqdqdq[2][2] = cos(x);
	// RX(x)^-1''' = RX''''(-x)
	frame->lg_inv_dqdqdqdq[1][1] = cos(-x);
	frame->lg_inv_dqdqdqdq[1][2] =-sin(-x);
	frame->lg_inv_dqdqdqdq[2][1] = sin(-x);
	frame->lg_inv_dqdqdqdq[2][2] = cos(-x);

	frame->twist_hat[1][2] =-1;
	frame->twist_hat[2][1] = 1;
    }
    else if(frame->transform == TREP_RY) {
	frame->lg[0][0] = cos(x);
	frame->lg[0][2] = sin(x);
	frame->lg[2][0] =-sin(x);
	frame->lg[2][2] = cos(x);
	// RY(x)^-1 = RY(-x)
	frame->lg_inv[0][0] = cos(-x);
	frame->lg_inv[0][2] = sin(-x);
	frame->lg_inv[2][0] =-sin(-x);
	frame->lg_inv[2][2] = cos(-x);

	frame->lg_dq[0][0] =-sin(x);
	frame->lg_dq[0][2] = cos(x);
	frame->lg_dq[2][0] =-cos(x);
	frame->lg_dq[2][2] =-sin(x);
	// RY(x)^-1' = -RY'(-x)
	frame->lg_inv_dq[0][0] = sin(-x);
	frame->lg_inv_dq[0][2] =-cos(-x);
	frame->lg_inv_dq[2][0] = cos(-x);
	frame->lg_inv_dq[2][2] = sin(-x);

	frame->lg_dqdq[0][0] =-cos(x);
	frame->lg_dqdq[0][2] =-sin(x);
	frame->lg_dqdq[2][0] = sin(x);
	frame->lg_dqdq[2][2] =-cos(x);
	// RY(x)^-1'' = RY''(-x)
	frame->lg_inv_dqdq[0][0] =-cos(-x);
	frame->lg_inv_dqdq[0][2] =-sin(-x);
	frame->lg_inv_dqdq[2][0] = sin(-x);
	frame->lg_inv_dqdq[2][2] =-cos(-x);
	
	frame->lg_dqdqdq[0][0] = sin(x);
	frame->lg_dqdqdq[0][2] =-cos(x);
	frame->lg_dqdqdq[2][0] = cos(x);
	frame->lg_dqdqdq[2][2] = sin(x);
	// RY(x)^-1''' = -RY'''(-x)
	frame->lg_inv_dqdqdq[0][0] =-sin(-x);
	frame->lg_inv_dqdqdq[0][2] = cos(-x);
	frame->lg_inv_dqdqdq[2][0] =-cos(-x);
	frame->lg_inv_dqdqdq[2][2] =-sin(-x);
	
	frame->lg_dqdqdqdq[0][0] = cos(x);
	frame->lg_dqdqdqdq[0][2] = sin(x);
	frame->lg_dqdqdqdq[2][0] =-sin(x);
	frame->lg_dqdqdqdq[2][2] = cos(x);
	// RY(x)^-1'''' = RY''''(-x)
	frame->lg_inv_dqdqdqdq[0][0] = cos(-x);
	frame->lg_inv_dqdqdqdq[0][2] = sin(-x);
	frame->lg_inv_dqdqdqdq[2][0] =-sin(-x);
	frame->lg_inv_dqdqdqdq[2][2] = cos(-x);

	frame->twist_hat[0][2] = 1;
	frame->twist_hat[2][0] =-1;
    }
    else if(frame->transform == TREP_RZ) {
	frame->lg[0][0] = cos(x);
	frame->lg[0][1] =-sin(x);
	frame->lg[1][0] = sin(x);
	frame->lg[1][1] = cos(x);
	// RZ(x)^-1 = RZ(-x)
	frame->lg_inv[0][0] = cos(-x);
	frame->lg_inv[0][1] =-sin(-x);
	frame->lg_inv[1][0] = sin(-x);
	frame->lg_inv[1][1] = cos(-x);

	frame->lg_dq[0][0] =-sin(x);
	frame->lg_dq[0][1] =-cos(x);
	frame->lg_dq[1][0] = cos(x);
	frame->lg_dq[1][1] =-sin(x);
	// RZ(x)^-1' = -RZ'(-x)
	frame->lg_inv_dq[0][0] = sin(-x);
	frame->lg_inv_dq[0][1] = cos(-x);
	frame->lg_inv_dq[1][0] =-cos(-x);
	frame->lg_inv_dq[1][1] = sin(-x);

	frame->lg_dqdq[0][0] =-cos(x);
	frame->lg_dqdq[0][1] = sin(x);
	frame->lg_dqdq[1][0] =-sin(x);
	frame->lg_dqdq[1][1] =-cos(x);
	// RZ(x)^-1'' = RZ''(-x)
	frame->lg_inv_dqdq[0][0] =-cos(-x);
	frame->lg_inv_dqdq[0][1] = sin(-x);
	frame->lg_inv_dqdq[1][0] =-sin(-x);
	frame->lg_inv_dqdq[1][1] =-cos(-x);
	
	frame->lg_dqdqdq[0][0] = sin(x);
	frame->lg_dqdqdq[0][1] = cos(x);
	frame->lg_dqdqdq[1][0] =-cos(x);
	frame->lg_dqdqdq[1][1] = sin(x);
	// RZ(x)^-1''' = -RZ'''(-x)
	frame->lg_inv_dqdqdq[0][0] =-sin(-x);
	frame->lg_inv_dqdqdq[0][1] =-cos(-x);
	frame->lg_inv_dqdqdq[1][0] = cos(-x);
	frame->lg_inv_dqdqdq[1][1] =-sin(-x);

	frame->lg_dqdqdqdq[0][0] = cos(x);
	frame->lg_dqdqdqdq[0][1] =-sin(x);
	frame->lg_dqdqdqdq[1][0] = sin(x);
	frame->lg_dqdqdqdq[1][1] = cos(x);
	// RZ(x)^-1'''' = RZ''''(-x)
	frame->lg_inv_dqdqdqdq[0][0] = cos(-x);
	frame->lg_inv_dqdqdqdq[0][1] =-sin(-x);
	frame->lg_inv_dqdqdqdq[1][0] = sin(-x);
	frame->lg_inv_dqdqdqdq[1][1] = cos(-x);

	frame->twist_hat[0][1] =-1;
	frame->twist_hat[1][0] = 1;
    }
    else if(frame->transform == TREP_CONST_SE3) {
	/* The SE3 transformation is stored in lg, so we don't have to
	 * do anything. All derivatives are zero. */
    }
    else {
	PyErr_SetString(PyExc_ValueError, "Unknown frame type");
    }

    for(i = 0; i < Frame_CHILD_SIZE(frame); i++)
	build_lg_cache_int(Frame_CHILD(frame, i));
}

void build_lg_cache(System *system)
{
    if(system->cache & SYSTEM_CACHE_LG)
	return;
    build_lg_cache_int(system->world_frame);
    system->cache |= SYSTEM_CACHE_LG;
}

static void build_g_cache_int(Frame *frame)
{
    int i;

    if(frame->transform == TREP_WORLD)
	eye_mat4x4(frame->g);
    else 
	frame->multiply_gk(frame, frame->g, *g_cached(frame->parent), 0);

    frame->p[0] = frame->g[0][3];
    frame->p[1] = frame->g[1][3];
    frame->p[2] = frame->g[2][3];
    frame->p[3] = frame->g[3][3];    

    for(i = 0; i < Frame_CHILD_SIZE(frame); i++)
	build_g_cache_int(Frame_CHILD(frame, i));
}

void build_g_cache(System *system)
{
    if(system->cache & SYSTEM_CACHE_G)
	return;
    build_lg_cache(system);
    build_g_cache_int(system->world_frame);
    system->cache |= SYSTEM_CACHE_G;    
}

static void build_g_dq_cache_int(Frame *frame)
{
    int i1 = 0;
    Config *q1 = NULL;

    for(i1 = 0; i1 < Frame_CACHE_SIZE(frame); i1++) {
	q1 = Frame_CACHE(frame, i1);
	
	// Equation (3) from Johnson and Murphey, ICRA 2008
	if(frame->transform == TREP_WORLD)
	    clear_mat4x4(G_DQ(i1));
	else if(frame->config != q1) {
	    frame->multiply_gk(frame, G_DQ(i1), *g_dq_cached(frame->parent, q1), 0);
	}
	else {  //if(frame->config == config)
	    frame->multiply_gk(frame,
			       G_DQ(i1),
			       *g_cached(frame->parent), 1);
	}

	// Write p_dq[] values
	P_DQ(i1)[0] = G_DQ(i1)[0][3];
        P_DQ(i1)[1] = G_DQ(i1)[1][3];
	P_DQ(i1)[2] = G_DQ(i1)[2][3];
	P_DQ(i1)[3] = G_DQ(i1)[3][3];
    }

    for(i1 = 0; i1 < Frame_CHILD_SIZE(frame); i1++)
	build_g_dq_cache_int(Frame_CHILD(frame, i1));
}

void build_g_dq_cache(System *system)
{
    if(system->cache & SYSTEM_CACHE_G_DQ)
	return;
    build_g_cache(system);
    build_g_dq_cache_int(system->world_frame);
    system->cache |= SYSTEM_CACHE_G_DQ;
}

static void build_g_dqdq_cache_int(Frame *frame)
{
    int i1 = 0;
    int i2 = 0;
    Config *q1 = NULL;
    Config *q2 = NULL;

    if (PyArray_DIM(frame->g_dqdq, 0) != Frame_CACHE_SIZE(frame))
    {
      Py_DECREF(frame->g_dqdq);
      Py_DECREF(frame->p_dqdq);
      
      npy_intp dims[] = {
        Frame_CACHE_SIZE(frame),
        Frame_CACHE_SIZE(frame),
        4,
        4
      };
      frame->g_dqdq = (PyArrayObject*)PyArray_ZEROS(
        sizeof(dims)/sizeof(npy_intp),
        dims,
        NPY_DOUBLE,
        0);
      frame->p_dqdq = (PyArrayObject*)PyArray_ZEROS(
        sizeof(dims)/sizeof(npy_intp) - 1,
        dims,
        NPY_DOUBLE,
        0);
    }

    for(i1 = 0; i1 < Frame_CACHE_SIZE(frame); i1++) {
	q1 = Frame_CACHE(frame, i1);
	for(i2 = i1; i2 < Frame_CACHE_SIZE(frame); i2++) {
	    q2 = Frame_CACHE(frame, i2);
	    // Equation (4) from Johnson and Murphey, ICRA 2008
	    if(frame->transform == TREP_WORLD)
		clear_mat4x4(G_DQDQ(i1, i2));
	    else if(frame->config == q1 && frame->config == q2) {
		frame->multiply_gk(frame,
				   G_DQDQ(i1, i2),
				   *g_cached(frame->parent), 2);
	    }
	    else if(frame->config == q1 && frame->config != q2) {
		frame->multiply_gk(frame,
				   G_DQDQ(i1, i2),
				   *g_dq_cached(frame->parent, q2), 1);
	    }
	    else if(frame->config != q1 && frame->config == q2) {
		frame->multiply_gk(frame,
				   G_DQDQ(i1, i2),
				   *g_dq_cached(frame->parent, q1), 1);
	    }
	    else { // none equal 
		frame->multiply_gk(frame,
				   G_DQDQ(i1, i2), *
				   g_dqdq_cached(frame->parent, q1, q2), 0);
	    }

	    P_DQDQ(i1, i2)[0] = G_DQDQ(i1, i2)[0][3];
	    P_DQDQ(i1, i2)[1] = G_DQDQ(i1, i2)[1][3];
	    P_DQDQ(i1, i2)[2] = G_DQDQ(i1, i2)[2][3];
	    P_DQDQ(i1, i2)[3] = G_DQDQ(i1, i2)[3][3];
	}
    }

    for(i1 = 0; i1 < Frame_CHILD_SIZE(frame); i1++)
	build_g_dqdq_cache_int(Frame_CHILD(frame, i1));
}

void build_g_dqdq_cache(System *system)
{
    if(system->cache & SYSTEM_CACHE_G_DQDQ)
	return;
    build_g_dq_cache(system);
    build_g_dqdq_cache_int(system->world_frame);
    system->cache |= SYSTEM_CACHE_G_DQDQ;
}

static void build_g_dqdqdq_cache_int(Frame *frame)
{
    int i = 0;
    int j = 0;
    int k = 0;
    Config *q1 = NULL;
    Config *q2 = NULL;
    Config *q3 = NULL;

    if (PyArray_DIM(frame->g_dqdqdq, 0) != Frame_CACHE_SIZE(frame))
    {
      Py_DECREF(frame->g_dqdqdq);
      Py_DECREF(frame->p_dqdqdq);
      
      npy_intp dims[] = {
        Frame_CACHE_SIZE(frame),
        Frame_CACHE_SIZE(frame),
        Frame_CACHE_SIZE(frame),
        4,
        4
      };
      frame->g_dqdqdq = (PyArrayObject*)PyArray_ZEROS(
        sizeof(dims)/sizeof(npy_intp),
        dims,
        NPY_DOUBLE,
        0);
      frame->p_dqdqdq = (PyArrayObject*)PyArray_ZEROS(
        sizeof(dims)/sizeof(npy_intp) - 1,
        dims,
        NPY_DOUBLE,
        0);
    }

    for(i = 0; i < Frame_CACHE_SIZE(frame); i++) {
	q1 = Frame_CACHE(frame, i);
	for(j = i; j < Frame_CACHE_SIZE(frame); j++) {
	    q2 = Frame_CACHE(frame, j);
	    for(k = j; k < Frame_CACHE_SIZE(frame); k++) {
		q3 = Frame_CACHE(frame, k);
	    
		// Equation (4) from Johnson and Murphey, ICRA 2008
		if(frame->transform == TREP_WORLD)
		    clear_mat4x4(G_DQDQDQ(i, j, k));
		else if(frame->config == q1 && 
			frame->config == q2 &&
		        frame->config == q3)
		{
		    frame->multiply_gk(frame,
				       G_DQDQDQ(i, j, k),
				       *g_cached(frame->parent), 3);
		}
		else if(frame->config == q1 &&
			frame->config == q2 &&
			frame->config != q3)
		{
		    frame->multiply_gk(frame,
				       G_DQDQDQ(i, j, k),
				       *g_dq_cached(frame->parent, q3), 2);
		}
		else if(frame->config == q1 &&
			frame->config != q2 &&
			frame->config == q3)
		{
		    frame->multiply_gk(frame,
				       G_DQDQDQ(i, j, k),
				       *g_dq_cached(frame->parent, q2), 2);
		}
		else if(frame->config != q1 &&
			frame->config == q2 &&
			frame->config == q3)
		{
		    frame->multiply_gk(frame,
				       G_DQDQDQ(i, j, k),
				       *g_dq_cached(frame->parent, q1), 2);
		}
		else if(frame->config == q1 &&
			frame->config != q2 &&
			frame->config != q3)
		{
		    frame->multiply_gk(frame,
				       G_DQDQDQ(i, j, k),
				       *g_dqdq_cached(frame->parent, q2, q3), 1);
		}
		else if(frame->config != q1 &&
			frame->config == q2 &&
			frame->config != q3)
		{
		    frame->multiply_gk(frame,
				       G_DQDQDQ(i, j, k),
				       *g_dqdq_cached(frame->parent, q1, q3), 1);
		}
		else if(frame->config != q1 &&
			frame->config != q2 &&
			frame->config == q3)
		{
		    frame->multiply_gk(frame,
				       G_DQDQDQ(i, j, k),
				       *g_dqdq_cached(frame->parent, q1, q2), 1);
		}
		else // none equal
		{
		    frame->multiply_gk(frame, G_DQDQDQ(i,j, k),
				       *g_dqdqdq_cached(frame->parent, q1, q2, q3), 0);
		}

		P_DQDQDQ(i, j, k)[0] = G_DQDQDQ(i, j, k)[0][3];
		P_DQDQDQ(i, j, k)[1] = G_DQDQDQ(i, j, k)[1][3];
		P_DQDQDQ(i, j, k)[2] = G_DQDQDQ(i, j, k)[2][3];
		P_DQDQDQ(i, j, k)[3] = G_DQDQDQ(i, j, k)[3][3];
	    }
	}
    }

    for(i = 0; i < Frame_CHILD_SIZE(frame); i++)
	build_g_dqdqdq_cache_int(Frame_CHILD(frame, i));
}

void build_g_dqdqdq_cache(System *system)
{
    if(system->cache & SYSTEM_CACHE_G_DQDQDQ)
	return;
    build_g_dqdq_cache(system);
    build_g_dqdqdq_cache_int(system->world_frame);
    system->cache |= SYSTEM_CACHE_G_DQDQDQ;
}

static void build_g_dqdqdqdq_cache_int(Frame *frame)
{
    int i1 = 0;
    int i2 = 0;
    int i3 = 0;
    int i4 = 0;
    Config *q1 = NULL;
    Config *q2 = NULL;
    Config *q3 = NULL;
    Config *q4 = NULL;

    if (PyArray_DIM(frame->g_dqdqdqdq, 0) != Frame_CACHE_SIZE(frame))
    {
      Py_DECREF(frame->g_dqdqdqdq);
      Py_DECREF(frame->p_dqdqdqdq);
      
      npy_intp dims[] = {
        Frame_CACHE_SIZE(frame),
        Frame_CACHE_SIZE(frame),
        Frame_CACHE_SIZE(frame),
        Frame_CACHE_SIZE(frame),
        4,
        4
      };
      frame->g_dqdqdqdq = (PyArrayObject*)PyArray_ZEROS(
        sizeof(dims)/sizeof(npy_intp),
        dims,
        NPY_DOUBLE,
        0);
      frame->p_dqdqdqdq = (PyArrayObject*)PyArray_ZEROS(
        sizeof(dims)/sizeof(npy_intp) - 1,
        dims,
        NPY_DOUBLE,
        0);
    }

    for(i1 = 0; i1 < Frame_CACHE_SIZE(frame); i1++) {
	q1 = Frame_CACHE(frame, i1);
	for(i2 = i1; i2 < Frame_CACHE_SIZE(frame); i2++) {
	    q2 = Frame_CACHE(frame, i2);
	    for(i3 = i2; i3 < Frame_CACHE_SIZE(frame); i3++) {
		q3 = Frame_CACHE(frame, i3);
		for(i4 = i3; i4 < Frame_CACHE_SIZE(frame); i4++) {
		    q4 = Frame_CACHE(frame, i4);

		    if(frame->transform == TREP_WORLD)
			clear_mat4x4(G_DQDQDQDQ(i1, i2, i3, i4));
		    else if(frame->config == q1 &&
			    frame->config == q2 &&
			    frame->config == q3 &&
			    frame->config == q4)
		    {
			frame->multiply_gk(frame,
					   G_DQDQDQDQ(i1, i2, i3, i4),
					   *g_cached(frame->parent), 4);
		    }
		    else if(frame->config != q1 &&
			    frame->config == q2 &&
			    frame->config == q3 &&
			    frame->config == q4)
		    {
			frame->multiply_gk(frame,
					   G_DQDQDQDQ(i1, i2, i3, i4),
					   *g_dq_cached(frame->parent, q1), 3);
		    }
		    else if(frame->config == q1 &&
			    frame->config != q2 &&
			    frame->config == q3 &&
			    frame->config == q4)
		    {
			frame->multiply_gk(frame,
					   G_DQDQDQDQ(i1, i2, i3, i4),
					   *g_dq_cached(frame->parent, q2), 3);
		    }
		    else if(frame->config == q1 &&
			    frame->config == q2 &&
			    frame->config != q3 &&
			    frame->config == q4)
		    {
			frame->multiply_gk(frame,
					   G_DQDQDQDQ(i1, i2, i3, i4),
					   *g_dq_cached(frame->parent, q3), 3);
		    }
		    else if(frame->config == q1 &&
			    frame->config == q2 &&
			    frame->config == q3 &&
			    frame->config != q4)
		    {
			frame->multiply_gk(frame,
					   G_DQDQDQDQ(i1, i2, i3, i4),
					   *g_dq_cached(frame->parent, q4), 3);
		    }
		    else if(frame->config != q1 &&
			    frame->config != q2 &&
			    frame->config == q3 &&
			    frame->config == q4)
		    {
			frame->multiply_gk(frame,
					   G_DQDQDQDQ(i1, i2, i3, i4),
					   *g_dqdq_cached(frame->parent, q1, q2), 2);
		    }
		    else if(frame->config != q1 &&
			    frame->config == q2 &&
			    frame->config != q3 &&
			    frame->config == q4)
		    {
			frame->multiply_gk(frame,
					   G_DQDQDQDQ(i1, i2, i3, i4),
					   *g_dqdq_cached(frame->parent, q1, q3), 2);
		    }
		    else if(frame->config != q1 &&
			    frame->config == q2 &&
			    frame->config == q3 &&
			    frame->config != q4)
		    {
			frame->multiply_gk(frame,
					   G_DQDQDQDQ(i1, i2, i3, i4),
					   *g_dqdq_cached(frame->parent, q1, q4), 2);
		    }
		    else if(frame->config == q1 &&
			    frame->config != q2 &&
			    frame->config != q3 &&
			    frame->config == q4)
		    {
			frame->multiply_gk(frame,
					   G_DQDQDQDQ(i1, i2, i3, i4),
					   *g_dqdq_cached(frame->parent, q2, q3), 2);
		    }
		    else if(frame->config == q1 &&
			    frame->config != q2 &&
			    frame->config == q3 &&
			    frame->config != q4)
		    {
			frame->multiply_gk(frame,
					   G_DQDQDQDQ(i1, i2, i3, i4),
					   *g_dqdq_cached(frame->parent, q2, q4), 2);
		    }
		    else if(frame->config == q1 &&
			    frame->config == q2 &&
			    frame->config != q3 &&
			    frame->config != q4)
		    {
			frame->multiply_gk(frame,
					   G_DQDQDQDQ(i1, i2, i3, i4),
					   *g_dqdq_cached(frame->parent, q3, q4), 2);
		    }
		    else if(frame->config != q1 &&
			    frame->config != q2 &&
			    frame->config != q3 &&
			    frame->config == q4)
		    {
			frame->multiply_gk(frame,
					   G_DQDQDQDQ(i1, i2, i3, i4),
					   *g_dqdqdq_cached(frame->parent, q1, q2, q3), 1);
		    }
		    else if(frame->config != q1 &&
			    frame->config != q2 &&
			    frame->config == q3 &&
			    frame->config != q4)
		    {
			frame->multiply_gk(frame,
					   G_DQDQDQDQ(i1, i2, i3, i4),
					   *g_dqdqdq_cached(frame->parent, q1, q2, q4), 1);
		    }
		    else if(frame->config != q1 &&
			    frame->config == q2 &&
			    frame->config != q3 &&
			    frame->config != q4)
		    {
			frame->multiply_gk(frame,
					   G_DQDQDQDQ(i1, i2, i3, i4),
					   *g_dqdqdq_cached(frame->parent, q1, q3, q4), 1);
		    }
		    else if(frame->config == q1 &&
			    frame->config != q2 &&
			    frame->config != q3 &&
			    frame->config != q4)
		    {
			frame->multiply_gk(frame,
					   G_DQDQDQDQ(i1, i2, i3, i4),
					   *g_dqdqdq_cached(frame->parent, q2, q3, q4), 1);
		    }
		    else { // if none equal
			frame->multiply_gk(frame,
					   G_DQDQDQDQ(i1, i2, i3, i4),
					   *g_dqdqdqdq_cached(frame->parent, q1, q2, q3, q4), 0);
		    }

		    P_DQDQDQDQ(i1, i2, i3, i4)[0] = G_DQDQDQDQ(i1, i2, i3, i4)[0][3];
		    P_DQDQDQDQ(i1, i2, i3, i4)[1] = G_DQDQDQDQ(i1, i2, i3, i4)[1][3];
		    P_DQDQDQDQ(i1, i2, i3, i4)[2] = G_DQDQDQDQ(i1, i2, i3, i4)[2][3];
		    P_DQDQDQDQ(i1, i2, i3, i4)[3] = G_DQDQDQDQ(i1, i2, i3, i4)[3][3];
		}
	    }
	}
    }

    for(i1 = 0; i1 < Frame_CHILD_SIZE(frame); i1++)
	build_g_dqdqdqdq_cache_int(Frame_CHILD(frame, i1));
}

void build_g_dqdqdqdq_cache(System *system)
{
    if(system->cache & SYSTEM_CACHE_G_DQDQDQDQ)
	return;
    build_g_dqdqdq_cache(system);
    build_g_dqdqdqdq_cache_int(system->world_frame);
    system->cache |= SYSTEM_CACHE_G_DQDQDQDQ;
}


static void build_g_inv_cache_int(Frame *frame)
{
    int i;
    if(frame->transform == TREP_WORLD)
    	eye_mat4x4(frame->g_inv);
    else 
    	mul_mm4(frame->g_inv, frame->lg_inv, *g_inv_cached(frame->parent));
    
    for(i = 0; i < Frame_CHILD_SIZE(frame); i++)
	build_g_inv_cache_int(Frame_CHILD(frame, i));
}

void build_g_inv_cache(System *system)
{
    if(system->cache & SYSTEM_CACHE_G_INV)
	return;
    build_lg_cache(system);
    build_g_inv_cache_int(system->world_frame);
    system->cache |= SYSTEM_CACHE_G_INV;    
}

static void build_g_inv_dq_cache_int(Frame *frame)
{
    int i1 = 0;
    Config *q1 = NULL;

    for(i1 = 0; i1 < Frame_CACHE_SIZE(frame); i1++) {
	q1 = Frame_CACHE(frame, i1);
	
	if(frame->transform == TREP_WORLD)
	    clear_mat4x4(G_INV_DQ(i1));
	else if(frame->config != q1) {
	    mul_mm4(G_INV_DQ(i1), frame->lg_inv, *g_inv_dq_cached(frame->parent, q1));
	}
	else {  //if(frame->config == config) 
	    mul_mm4(G_INV_DQ(i1), frame->lg_inv_dq, *g_inv_cached(frame->parent));
	}
    }

    for(i1 = 0; i1 < Frame_CHILD_SIZE(frame); i1++)
	build_g_inv_dq_cache_int(Frame_CHILD(frame, i1));
}

void build_g_inv_dq_cache(System *system)
{
    if(system->cache & SYSTEM_CACHE_G_INV_DQ)
	return;
    build_g_inv_cache(system);
    build_g_inv_dq_cache_int(system->world_frame);
    system->cache |= SYSTEM_CACHE_G_INV_DQ;
}

static void build_g_inv_dqdq_cache_int(Frame *frame)
{
    int i1 = 0;
    int i2 = 0;
    Config *q1 = NULL;
    Config *q2 = NULL;

    for(i1 = 0; i1 < Frame_CACHE_SIZE(frame); i1++) {
	q1 = Frame_CACHE(frame, i1);
	for(i2 = i1; i2 < Frame_CACHE_SIZE(frame); i2++) {
	    q2 = Frame_CACHE(frame, i2);

	    if(frame->transform == TREP_WORLD)
		clear_mat4x4(G_INV_DQDQ(i1, i2));
	    else if(frame->config == q1 && frame->config == q2) {
		mul_mm4(G_INV_DQDQ(i1, i2),
			frame->lg_inv_dqdq,
			*g_inv_cached(frame->parent));
	    }
	    else if(frame->config == q1 && frame->config != q2) {
		mul_mm4(G_INV_DQDQ(i1, i2),
			frame->lg_inv_dq,
			*g_inv_dq_cached(frame->parent, q2));
	    }
	    else if(frame->config != q1 && frame->config == q2) {
		mul_mm4(G_INV_DQDQ(i1, i2),
			frame->lg_inv_dq,
			*g_inv_dq_cached(frame->parent, q1));
	    }
	    else { // none equal 
		mul_mm4(G_INV_DQDQ(i1, i2),
			frame->lg_inv,
			*g_inv_dqdq_cached(frame->parent, q1, q2));
	    }
	}
    }

    for(i1 = 0; i1 < Frame_CHILD_SIZE(frame); i1++)
	build_g_inv_dqdq_cache_int(Frame_CHILD(frame, i1));
}

void build_g_inv_dqdq_cache(System *system)
{
    if(system->cache & SYSTEM_CACHE_G_INV_DQDQ)
	return;
    build_g_inv_dq_cache(system);
    build_g_inv_dqdq_cache_int(system->world_frame);
    system->cache |= SYSTEM_CACHE_G_INV_DQDQ;
}

static void build_vb_cache_int(Frame *frame)
{
    int i;

    // Equation (6) from Johnson and Murphey, ICRA 2008
    // Vb = g^-1 * Vb(parent) * g + twist*dq
    if(frame->transform == TREP_WORLD)
	clear_mat4x4(frame->vb);
    else {
	if(frame->config == NULL) 
	    clear_mat4x4(frame->vb); 
	else 
	    mul_dm4(frame->vb, frame->config->dq, frame->twist_hat);
	frame->add_sandwich_gk(frame, frame->vb, *vb_cached(frame->parent), 0, 0);
    }
    
    for(i = 0; i < Frame_CHILD_SIZE(frame); i++)
	build_vb_cache_int(Frame_CHILD(frame, i));
}

void build_vb_cache(System *system)
{
    if(system->cache & SYSTEM_CACHE_VB)
	return;
    build_lg_cache(system);
    build_vb_cache_int(system->world_frame);
    system->cache |= SYSTEM_CACHE_VB;
}

static void build_vb_dq_cache_int(Frame *frame)
{
    int i = 0;
    Config *q1 = NULL;

    for(i = 0; i < Frame_CACHE_SIZE(frame); i++) {
	q1 = Frame_CACHE(frame, i);
	
	// Equation (7) from Johnson and Murphey, ICRA 2008
	if(frame->transform == TREP_WORLD)
	    clear_mat4x4(VB_DQ(i));
	else if(frame->config == q1) {
	    clear_mat4x4(VB_DQ(i));
	    frame->add_sandwich_gk(frame, VB_DQ(i), *vb_cached(frame->parent), 0, 1);
	}
	else { //if(frame->config != q1)
	    clear_mat4x4(VB_DQ(i));
	    frame->add_sandwich_gk(frame, VB_DQ(i), *vb_dq_cached(frame->parent, q1), 0, 0);
	}
    }

    for(i = 0; i < Frame_CHILD_SIZE(frame); i++)
	build_vb_dq_cache_int(Frame_CHILD(frame, i));
}

void build_vb_dq_cache(System *system)
{
    if(system->cache & SYSTEM_CACHE_VB_DQ)
	return;
    build_vb_cache(system);
    build_vb_dq_cache_int(system->world_frame);
    system->cache |= SYSTEM_CACHE_VB_DQ;
}

static void build_vb_dqdq_cache_int(Frame *frame)
{
    int i = 0;
    int j = 0;
    Config *q1 = NULL;
    Config *q2 = NULL;
    
    if (PyArray_DIM(frame->vb_dqdq, 0) != Frame_CACHE_SIZE(frame))
    {
      Py_DECREF(frame->vb_dqdq);
      
      npy_intp dims[] = {
        Frame_CACHE_SIZE(frame),
        Frame_CACHE_SIZE(frame),
        4,
        4
      };
      frame->vb_dqdq = (PyArrayObject*)PyArray_ZEROS(
        sizeof(dims)/sizeof(npy_intp),
        dims,
        NPY_DOUBLE,
        0);
    }

    for(i = 0; i < Frame_CACHE_SIZE(frame); i++) {
	q1 = Frame_CACHE(frame, i);
	for(j = i; j < Frame_CACHE_SIZE(frame); j++) {
	    q2 = Frame_CACHE(frame, j);
	    
	    // Equation (8) from Johnson and Murphey, ICRA 2008
	    if(frame->transform == TREP_WORLD)
		clear_mat4x4(VB_DQDQ(i, j));
	    else if(frame->config == q1 && frame->config == q2)
	    {
		clear_mat4x4(VB_DQDQ(i, j));
		frame->add_sandwich_gk(frame, VB_DQDQ(i, j), *vb_cached(frame->parent), 1, 1);
		mul_dm4(VB_DQDQ(i, j), 2.0, VB_DQDQ(i, j));
		frame->add_sandwich_gk(frame, VB_DQDQ(i, j), *vb_cached(frame->parent), 0, 2);
	    }
	    else if(frame->config != q1 && frame->config == q2)
	    {
		clear_mat4x4(VB_DQDQ(i, j));
		frame->add_sandwich_gk(frame, VB_DQDQ(i, j), *vb_dq_cached(frame->parent, q1), 0, 1);
	    }
	    else if(frame->config == q1 && frame->config != q2)
	    {
		clear_mat4x4(VB_DQDQ(i, j));
		frame->add_sandwich_gk(frame, VB_DQDQ(i, j), *vb_dq_cached(frame->parent, q2), 0, 1);
	    }
	    else //if(frame->config != q1 && frame->config != q2)
	    {
		clear_mat4x4(VB_DQDQ(i, j));
		frame->add_sandwich_gk(frame, VB_DQDQ(i, j), *vb_dqdq_cached(frame->parent, q1, q2), 0, 0);
	    }
	}
    }

    for(i = 0; i < Frame_CHILD_SIZE(frame); i++)
	build_vb_dqdq_cache_int(Frame_CHILD(frame, i));
}

void build_vb_dqdq_cache(System *system)
{
    if(system->cache & SYSTEM_CACHE_VB_DQDQ)
	return;
    build_vb_dq_cache(system);
    build_vb_dqdq_cache_int(system->world_frame);
    system->cache |= SYSTEM_CACHE_VB_DQDQ;
}

static void build_vb_dqdqdq_cache_int(Frame *frame)
{
    int i = 0;
    int j = 0;
    int k = 0;
    Config *q1 = NULL;
    Config *q2 = NULL;
    Config *q3 = NULL;
    
    if (PyArray_DIM(frame->vb_dqdqdq, 0) != Frame_CACHE_SIZE(frame))
    {
      Py_DECREF(frame->vb_dqdqdq);
      
      npy_intp dims[] = {
        Frame_CACHE_SIZE(frame),
        Frame_CACHE_SIZE(frame),
        Frame_CACHE_SIZE(frame),
        4,
        4
      };
      frame->vb_dqdqdq = (PyArrayObject*)PyArray_ZEROS(
        sizeof(dims)/sizeof(npy_intp),
        dims,
        NPY_DOUBLE,
        0);
    }

    for(i = 0; i < Frame_CACHE_SIZE(frame); i++) {
	q1 = Frame_CACHE(frame, i);
	for(j = i; j < Frame_CACHE_SIZE(frame); j++) {
	    q2 = Frame_CACHE(frame, j);
	    for(k = j; k < Frame_CACHE_SIZE(frame); k++) {
		q3 = Frame_CACHE(frame, k);
	    
		// Equation (8) from Johnson and Murphey, ICRA 2008
		if(frame->transform == TREP_WORLD)
		    clear_mat4x4(VB_DQDQDQ(i, j, k));
		else if(frame->config == q1 &&
			frame->config == q2 &&
			frame->config == q3)
		{
		    clear_mat4x4(VB_DQDQDQ(i, j, k));
		    frame->add_sandwich_gk(frame, VB_DQDQDQ(i, j, k), *vb_cached(frame->parent), 1, 2);
		    mul_dm4(VB_DQDQDQ(i, j, k), 3.0, VB_DQDQDQ(i, j, k));
		    frame->add_sandwich_gk(frame, VB_DQDQDQ(i, j, k), *vb_cached(frame->parent), 0, 3);
		}
		else if(frame->config != q1 &&
			frame->config == q2 &&
		        frame->config == q3)
		{
		    clear_mat4x4(VB_DQDQDQ(i, j, k));
		    frame->add_sandwich_gk(frame, VB_DQDQDQ(i, j, k), *vb_dq_cached(frame->parent, q1), 1, 1);
		    mul_dm4(VB_DQDQDQ(i, j, k), 2.0, VB_DQDQDQ(i, j, k));
		    frame->add_sandwich_gk(frame, VB_DQDQDQ(i, j, k), *vb_dq_cached(frame->parent, q1), 0, 2);
		}
		else if(frame->config == q1 &&
			frame->config != q2 &&
		        frame->config == q3)
		{
		    clear_mat4x4(VB_DQDQDQ(i, j, k));
		    frame->add_sandwich_gk(frame, VB_DQDQDQ(i, j, k), *vb_dq_cached(frame->parent, q2), 1, 1);
		    mul_dm4(VB_DQDQDQ(i, j, k), 2.0, VB_DQDQDQ(i, j, k));
		    frame->add_sandwich_gk(frame, VB_DQDQDQ(i, j, k), *vb_dq_cached(frame->parent, q2), 0, 2);
		}
		else if(frame->config == q1 &&
			frame->config == q2 &&
		        frame->config != q3)
		{
		    clear_mat4x4(VB_DQDQDQ(i, j, k));
		    frame->add_sandwich_gk(frame, VB_DQDQDQ(i, j, k), *vb_dq_cached(frame->parent, q3), 1, 1);
		    mul_dm4(VB_DQDQDQ(i, j, k), 2.0, VB_DQDQDQ(i, j, k));
		    frame->add_sandwich_gk(frame, VB_DQDQDQ(i, j, k), *vb_dq_cached(frame->parent, q3), 0, 2);
		}
		else if(frame->config == q1 &&
			frame->config != q2 &&
		        frame->config != q3)
		{
		    clear_mat4x4(VB_DQDQDQ(i, j, k));
		    frame->add_sandwich_gk(frame, VB_DQDQDQ(i, j, k),
                                           *vb_dqdq_cached(frame->parent, q2, q3), 0, 1);
		}
		else if(frame->config != q1 &&
			frame->config == q2 &&
		        frame->config != q3)
		{
		    clear_mat4x4(VB_DQDQDQ(i, j, k));
		    frame->add_sandwich_gk(frame, VB_DQDQDQ(i, j, k),
                                           *vb_dqdq_cached(frame->parent, q1, q3), 0, 1);
		}
		else if(frame->config != q1 &&
			frame->config != q2 &&
		        frame->config == q3)
		{
		    clear_mat4x4(VB_DQDQDQ(i, j, k));
		    frame->add_sandwich_gk(frame, VB_DQDQDQ(i, j, k),
                                           *vb_dqdq_cached(frame->parent, q1, q2), 0, 1);
		}
		else // none equal
		{
		    clear_mat4x4(VB_DQDQDQ(i, j, k));
		    frame->add_sandwich_gk(frame, VB_DQDQDQ(i, j, k),
                                           *vb_dqdqdq_cached(frame->parent, q1, q2, q3), 0, 0);
		}
	    }
	}
    }

    for(i = 0; i < Frame_CHILD_SIZE(frame); i++)
	build_vb_dqdqdq_cache_int(Frame_CHILD(frame, i));
}

void build_vb_dqdqdq_cache(System *system)
{
    if(system->cache & SYSTEM_CACHE_VB_DQDQDQ)
	return;
    build_vb_dqdq_cache(system);
    build_vb_dqdqdq_cache_int(system->world_frame);
    system->cache |= SYSTEM_CACHE_VB_DQDQDQ;
}

static void build_vb_ddq_cache_int(Frame *frame) 
{
    int i = 0;
    Config *dq1 = NULL;

    for(i = 0; i < Frame_CACHE_SIZE(frame); i++) {
	dq1 = Frame_CACHE(frame, i);
	// Equation (10) from Johnson and Murphey, ICRA 2008
	if(frame->transform == TREP_WORLD)
	    clear_mat4x4(VB_DDQ(i));
	else if(frame->config == dq1) {
	    copy_mat4x4(VB_DDQ(i), frame->twist_hat);
	}
	else { //if(frame->config != dq1)
	    clear_mat4x4(VB_DDQ(i));
	    frame->add_sandwich_gk(frame, VB_DDQ(i), *vb_ddq_cached(frame->parent, dq1), 0, 0);
	}
    }

    for(i = 0; i < Frame_CHILD_SIZE(frame); i++)
	build_vb_ddq_cache_int(Frame_CHILD(frame, i));
}

void build_vb_ddq_cache(System *system)
{
    if(system->cache & SYSTEM_CACHE_VB_DDQ)
	return;
    build_lg_cache(system);
    build_vb_ddq_cache_int(system->world_frame);
    system->cache |= SYSTEM_CACHE_VB_DDQ;
}

static void build_vb_ddqdq_cache_int(Frame *frame)
{
    int i = 0;
    int j = 0;
    Config *dq1 = NULL;
    Config *q2 = NULL;

    if (PyArray_DIM(frame->vb_ddqdq, 0) != Frame_CACHE_SIZE(frame))
    {
      Py_DECREF(frame->vb_ddqdq);
      
      npy_intp dims[] = {
        Frame_CACHE_SIZE(frame),
        Frame_CACHE_SIZE(frame),
        4,
        4
      };
      frame->vb_ddqdq = (PyArrayObject*)PyArray_ZEROS(
        sizeof(dims)/sizeof(npy_intp),
        dims,
        NPY_DOUBLE,
        0);
    }

    for(i = 0; i < Frame_CACHE_SIZE(frame); i++) {
	dq1 = Frame_CACHE(frame, i);
	for(j = 0; j < Frame_CACHE_SIZE(frame); j++) {
	    q2 = Frame_CACHE(frame, j);
	    
	    // Equation (9) from Johnson and Murphey, ICRA 2008
	    if(frame->transform == TREP_WORLD || frame->config == dq1)
		clear_mat4x4(VB_DDQDQ(i, j));
	    else if(frame->config == q2)
	    {
		clear_mat4x4(VB_DDQDQ(i, j));
		frame->add_sandwich_gk(frame, VB_DDQDQ(i, j), *vb_ddq_cached(frame->parent, dq1), 0, 1);
	    }
	    else //if(frame->config != q2)
	    {
		clear_mat4x4(VB_DDQDQ(i, j));
		frame->add_sandwich_gk(frame, VB_DDQDQ(i, j), *vb_ddqdq_cached(frame->parent, dq1, q2), 0, 0);
	    }
	}
    }

    for(i = 0; i < Frame_CHILD_SIZE(frame); i++)
	build_vb_ddqdq_cache_int(Frame_CHILD(frame, i));
}

void build_vb_ddqdq_cache(System *system)
{
    if(system->cache & SYSTEM_CACHE_VB_DDQDQ)
	return;
    build_vb_ddq_cache(system);
    build_vb_ddqdq_cache_int(system->world_frame);
    system->cache |= SYSTEM_CACHE_VB_DDQDQ;
}

static void build_vb_ddqdqdq_cache_int(Frame *frame)
{
    int i = 0;
    int j = 0;
    int k = 0;
    Config *dq1 = NULL;
    Config *q2 = NULL;
    Config *q3 = NULL;

    if (PyArray_DIM(frame->vb_ddqdqdq, 0) != Frame_CACHE_SIZE(frame))
    {
      Py_DECREF(frame->vb_ddqdqdq);
      
      npy_intp dims[] = {
        Frame_CACHE_SIZE(frame),
        Frame_CACHE_SIZE(frame),
        Frame_CACHE_SIZE(frame),
        4,
        4
      };
      frame->vb_ddqdqdq = (PyArrayObject*)PyArray_ZEROS(
        sizeof(dims)/sizeof(npy_intp),
        dims,
        NPY_DOUBLE,
        0);
    }

    for(i = 0; i < Frame_CACHE_SIZE(frame); i++) {
	dq1 = Frame_CACHE(frame, i);
	for(j = 0; j < Frame_CACHE_SIZE(frame); j++) {
	    q2 = Frame_CACHE(frame, j);
	    for(k = j; k < Frame_CACHE_SIZE(frame); k++) {
		q3 = Frame_CACHE(frame, k);

		if(frame->transform == TREP_WORLD || frame->config == dq1)
		    clear_mat4x4(VB_DDQDQDQ(i, j, k));
		else if(frame->config == q2 && frame->config == q3)
		{
		    clear_mat4x4(VB_DDQDQDQ(i, j, k));
		    frame->add_sandwich_gk(frame, VB_DDQDQDQ(i, j, k), *vb_ddq_cached(frame->parent, dq1), 1, 1);
		    mul_dm4(VB_DDQDQDQ(i, j, k), 2.0, VB_DDQDQDQ(i, j, k));
		    frame->add_sandwich_gk(frame, VB_DDQDQDQ(i, j, k), *vb_ddq_cached(frame->parent, dq1), 0, 2);
		}
		else if(frame->config != q2 && frame->config == q3)
		{
		    clear_mat4x4(VB_DDQDQDQ(i, j, k));
		    frame->add_sandwich_gk(frame, VB_DDQDQDQ(i, j, k),
                                           *vb_ddqdq_cached(frame->parent, dq1, q2), 0, 1);
		}
		else if(frame->config == q2 && frame->config != q3)
		{
		    clear_mat4x4(VB_DDQDQDQ(i, j, k));
		    frame->add_sandwich_gk(frame, VB_DDQDQDQ(i, j, k),
                                           *vb_ddqdq_cached(frame->parent, dq1, q3), 0, 1);
		}
		else // none equal
		{
		    clear_mat4x4(VB_DDQDQDQ(i, j, k));
		    frame->add_sandwich_gk(frame, VB_DDQDQDQ(i, j, k),
                                           *vb_ddqdqdq_cached(frame->parent, dq1, q2, q3), 0, 0);
		}
	    }
	}
    }

    for(i = 0; i < Frame_CHILD_SIZE(frame); i++)
	build_vb_ddqdqdq_cache_int(Frame_CHILD(frame, i));
}

void build_vb_ddqdqdq_cache(System *system)
{
    if(system->cache & SYSTEM_CACHE_VB_DDQDQDQ)
	return;
    build_vb_ddqdq_cache(system);
    build_vb_ddqdqdq_cache_int(system->world_frame);
    system->cache |= SYSTEM_CACHE_VB_DDQDQDQ;
}

static void build_vb_ddqdqdqdq_cache_int(Frame *frame)
{
    int i = 0;
    int j = 0;
    int k = 0;
    int l = 0;
    Config *dq1 = NULL;
    Config *q2 = NULL;
    Config *q3 = NULL;
    Config *q4 = NULL;

    if (PyArray_DIM(frame->vb_ddqdqdqdq, 0) != Frame_CACHE_SIZE(frame))
    {
      Py_DECREF(frame->vb_ddqdqdqdq);
      
      npy_intp dims[] = {
        Frame_CACHE_SIZE(frame),
        Frame_CACHE_SIZE(frame),
        Frame_CACHE_SIZE(frame),
        Frame_CACHE_SIZE(frame),
        4,
        4
      };
      frame->vb_ddqdqdqdq = (PyArrayObject*)PyArray_ZEROS(
        sizeof(dims)/sizeof(npy_intp),
        dims,
        NPY_DOUBLE,
        0);
    }

    for(i = 0; i < Frame_CACHE_SIZE(frame); i++) {
	dq1 = Frame_CACHE(frame, i);
	for(j = 0; j < Frame_CACHE_SIZE(frame); j++) {
	    q2 = Frame_CACHE(frame, j);
	    for(k = j; k < Frame_CACHE_SIZE(frame); k++) {
		q3 = Frame_CACHE(frame, k);
		for(l = k; l < Frame_CACHE_SIZE(frame); l++) {
		    q4 = Frame_CACHE(frame, l);
		
		    if(frame->transform == TREP_WORLD || frame->config == dq1)
			clear_mat4x4(VB_DDQDQDQDQ(i, j, k, l));
		    else if(frame->config == q2 && frame->config == q3 && frame->config == q4) {		    
			clear_mat4x4(VB_DDQDQDQDQ(i, j, k, l));
			frame->add_sandwich_gk(frame, VB_DDQDQDQDQ(i, j, k, l),
                                               *vb_ddq_cached(frame->parent, dq1), 1, 2);
			mul_dm4(VB_DDQDQDQDQ(i, j, k, l), 3.0, VB_DDQDQDQDQ(i, j, k, l));
			frame->add_sandwich_gk(frame, VB_DDQDQDQDQ(i, j, k, l),
                                               *vb_ddq_cached(frame->parent, dq1), 0, 3);
		    }
		    else if(frame->config == q2 && frame->config == q3 && frame->config != q4) {		    
			clear_mat4x4(VB_DDQDQDQDQ(i, j, k, l));
			frame->add_sandwich_gk(frame, VB_DDQDQDQDQ(i, j, k, l),
                                               *vb_ddqdq_cached(frame->parent, dq1, q4), 1, 1);
			mul_dm4(VB_DDQDQDQDQ(i, j, k, l), 2.0, VB_DDQDQDQDQ(i, j, k, l));
			frame->add_sandwich_gk(frame, VB_DDQDQDQDQ(i, j, k, l),
                                               *vb_ddqdq_cached(frame->parent, dq1, q4), 0, 2);
		    }
		    else if(frame->config == q2 && frame->config != q3 && frame->config == q4) {		    
			clear_mat4x4(VB_DDQDQDQDQ(i, j, k, l));
			frame->add_sandwich_gk(frame, VB_DDQDQDQDQ(i, j, k, l),
                                               *vb_ddqdq_cached(frame->parent, dq1, q3), 1, 1);
			mul_dm4(VB_DDQDQDQDQ(i, j, k, l), 2.0, VB_DDQDQDQDQ(i, j, k, l));
			frame->add_sandwich_gk(frame, VB_DDQDQDQDQ(i, j, k, l),
                                               *vb_ddqdq_cached(frame->parent, dq1, q3), 0, 2);
		    }
		    else if(frame->config != q2 && frame->config == q3 && frame->config == q4) {		    
			clear_mat4x4(VB_DDQDQDQDQ(i, j, k, l));
			frame->add_sandwich_gk(frame, VB_DDQDQDQDQ(i, j, k, l),
                                               *vb_ddqdq_cached(frame->parent, dq1, q2), 1, 1);
			mul_dm4(VB_DDQDQDQDQ(i, j, k, l), 2.0, VB_DDQDQDQDQ(i, j, k, l));
			frame->add_sandwich_gk(frame, VB_DDQDQDQDQ(i, j, k, l),
                                               *vb_ddqdq_cached(frame->parent, dq1, q2), 0, 2);
		    }
		    else if(frame->config == q2 && frame->config != q3 && frame->config != q4) {		    
			clear_mat4x4(VB_DDQDQDQDQ(i, j, k, l));
			frame->add_sandwich_gk(frame, VB_DDQDQDQDQ(i, j, k, l),
                                               *vb_ddqdqdq_cached(frame->parent, dq1, q3, q4), 0, 1);
		    }
		    else if(frame->config != q2 && frame->config == q3 && frame->config != q4) {		    
			clear_mat4x4(VB_DDQDQDQDQ(i, j, k, l));
			frame->add_sandwich_gk(frame, VB_DDQDQDQDQ(i, j, k, l),
                                               *vb_ddqdqdq_cached(frame->parent, dq1, q2, q4), 0, 1);
		    }
		    else if(frame->config != q2 && frame->config != q3 && frame->config == q4) {		    
			clear_mat4x4(VB_DDQDQDQDQ(i, j, k, l));
			frame->add_sandwich_gk(frame, VB_DDQDQDQDQ(i, j, k, l),
                                               *vb_ddqdqdq_cached(frame->parent, dq1, q2, q3), 0, 1);
		    }
		    else // none equal
		    {
			clear_mat4x4(VB_DDQDQDQDQ(i, j, k, l));
			frame->add_sandwich_gk(frame, VB_DDQDQDQDQ(i, j, k, l),
                                               *vb_ddqdqdqdq_cached(frame->parent, dq1, q2, q3, q4), 0, 0);
		    }
		}
	    }
	}
    }

    for(i = 0; i < Frame_CHILD_SIZE(frame); i++)
	build_vb_ddqdqdqdq_cache_int(Frame_CHILD(frame, i));
}

void build_vb_ddqdqdqdq_cache(System *system)
{
    if(system->cache & SYSTEM_CACHE_VB_DDQDQDQDQ)
	return;
    build_vb_ddqdqdq_cache(system);
    build_vb_ddqdqdqdq_cache_int(system->world_frame);
    system->cache |= SYSTEM_CACHE_VB_DDQDQDQDQ;
}

///////////////////////////////
// Cache retrieval functions //
///////////////////////////////

static mat4x4* g_cached(Frame *frame)
{
    return &frame->g;
}

static mat4x4* g_dq_cached(Frame *frame, Config *q1)
{
    int i = Frame_get_cache_index(frame, q1);
    if(i == -1)
	return &zero_mat4x4;
    else
	return (mat4x4*)G_DQ(i);
}

static mat4x4* g_dqdq_cached(Frame *frame, Config *q1, Config *q2)
{
    int i1 = Frame_get_cache_index(frame, q1);
    int i2 = Frame_get_cache_index(frame, q2);

    if(i1 == -1 || i2 == -1)
	return &zero_mat4x4;
    else {
	assert(i1 <= i2);		
	return (mat4x4*)G_DQDQ(i1, i2);
    }
}

static mat4x4* g_dqdqdq_cached(Frame *frame, Config *q1, Config *q2, Config *q3)
{
    int i1 = Frame_get_cache_index(frame, q1);
    int i2 = Frame_get_cache_index(frame, q2);
    int i3 = Frame_get_cache_index(frame, q3);

    if(i1 == -1 || i2 == -1 || i3 == -1)
	return &zero_mat4x4;
    else {
	assert(i1 <= i2 && i2 <= i3);
	return (mat4x4*)G_DQDQDQ(i1, i2, i3);
    }
}

static mat4x4* g_dqdqdqdq_cached(Frame *frame, Config *q1, Config *q2, Config *q3, Config *q4)
{
    int i1 = Frame_get_cache_index(frame, q1);
    int i2 = Frame_get_cache_index(frame, q2);
    int i3 = Frame_get_cache_index(frame, q3);
    int i4 = Frame_get_cache_index(frame, q4);

    if(i1 == -1 || i2 == -1 || i3 == -1 || i4 == -1) 
	return &zero_mat4x4;
    else {
	assert(i1 <= i2 && i2 <= i3 && i3 <= i4);
	return (mat4x4*)G_DQDQDQDQ(i1, i2, i3, i4);
    }
}

static mat4x4* g_inv_cached(Frame *frame)
{
    return &frame->g_inv;
}

static mat4x4* g_inv_dq_cached(Frame *frame, Config *q1)
{
    int i = Frame_get_cache_index(frame, q1);
    if(i == -1)
	return &zero_mat4x4;
    else
	return (mat4x4*)G_INV_DQ(i);
}

static mat4x4* g_inv_dqdq_cached(Frame *frame, Config *q1, Config *q2)
{
    int i1 = Frame_get_cache_index(frame, q1);
    int i2 = Frame_get_cache_index(frame, q2);

    if(i1 == -1 || i2 == -1)
	return &zero_mat4x4;
    else {
	assert(i1 <= i2);		
	return (mat4x4*)G_INV_DQDQ(i1, i2);
    }
}

static vec4* p_dq_cached(Frame *frame, Config *q1)
{
    int i = Frame_get_cache_index(frame, q1);

    if(i == -1)
	return &zero_vec4;
    else
	return (vec4*)P_DQ(i);
}

static vec4* p_dqdq_cached(Frame *frame, Config *q1, Config *q2)
{
    int i1 = Frame_get_cache_index(frame, q1);
    int i2 = Frame_get_cache_index(frame, q2);
    if(i1 == -1 || i2 == -1)
	return &zero_vec4;
    else {
	assert(i1 <= i2);
	return (vec4*)P_DQDQ(i1, i2);
    }
}

static vec4* p_dqdqdq_cached(Frame *frame, Config *q1, Config *q2, Config *q3)
{
    int i1 = Frame_get_cache_index(frame, q1);
    int i2 = Frame_get_cache_index(frame, q2);
    int i3 = Frame_get_cache_index(frame, q3);
    
    if(i1 == -1 || i2 == -1 || i3 == -1)
	return &zero_vec4;
    else {
	assert(i1 <= i2 && i2 <= i3);
	return (vec4*)P_DQDQDQ(i1, i2, i3);
    }
}

static vec4* p_dqdqdqdq_cached(Frame *frame, Config *q1, Config *q2, Config *q3, Config *q4)
{
    int i1 = Frame_get_cache_index(frame, q1);
    int i2 = Frame_get_cache_index(frame, q2);
    int i3 = Frame_get_cache_index(frame, q3);
    int i4 = Frame_get_cache_index(frame, q4);
    
    if(i1 == -1 || i2 == -1 || i3 == -1 || i4 == -1)
	return &zero_vec4;
    else {
	assert(i1 <= i2 && i2 <= i3 && i3 <= i4);
	return (vec4*)P_DQDQDQDQ(i1, i2, i3, i4);
    }
}

static mat4x4* vb_cached(Frame *frame)
{
    return &frame->vb;
}

static mat4x4* vb_dq_cached(Frame *frame, Config *q1)
{
    int i = Frame_get_cache_index(frame, q1);
    if(i == -1)
	return &zero_mat4x4;    
    else
	return (mat4x4*)VB_DQ(i);
}

static mat4x4* vb_dqdq_cached(Frame *frame, Config *q1, Config *q2)
{
    int i1 = Frame_get_cache_index(frame, q1);
    int i2 = Frame_get_cache_index(frame, q2);
    if(i1 == -1 || i2 == -1)
	return &zero_mat4x4;
    else {
	assert(i1 <= i2);
	return (mat4x4*)VB_DQDQ(i1, i2);
    }
}

static mat4x4* vb_dqdqdq_cached(Frame *frame, Config *q1, Config *q2, Config *q3)
{
    int i1 = Frame_get_cache_index(frame, q1);
    int i2 = Frame_get_cache_index(frame, q2);
    int i3 = Frame_get_cache_index(frame, q3);
    if(i1 == -1 || i2 == -1 || i3 == -1)
	return &zero_mat4x4;
    else {
	assert(i1 <= i2 && i2 <= i3);
	return (mat4x4*)VB_DQDQDQ(i1, i2, i3);
    }
}

static mat4x4* vb_ddq_cached(Frame *frame, Config *q1) 
{
    int i = Frame_get_cache_index(frame, q1);
    if(i == -1)
	return &zero_mat4x4;
    else
	return (mat4x4*)VB_DDQ(i);
}

static mat4x4* vb_ddqdq_cached(Frame *frame, Config *q1, Config *q2)
{
    int i1 = Frame_get_cache_index(frame, q1);
    int i2 = Frame_get_cache_index(frame, q2);
    if(i1 == -1 || i2 == -1)
	return &zero_mat4x4;
    else
	return (mat4x4*)VB_DDQDQ(i1, i2);
}

static mat4x4* vb_ddqdqdq_cached(Frame *frame, Config *q1, Config *q2, Config *q3)
{
    int i1 = Frame_get_cache_index(frame, q1);
    int i2 = Frame_get_cache_index(frame, q2);
    int i3 = Frame_get_cache_index(frame, q3);
    if(i1 == -1 || i2 == -1 || i3 == -1)
	return &zero_mat4x4;
    else {
	assert(i2 <= i3);
	return (mat4x4*)VB_DDQDQDQ(i1, i2, i3);
    }
}

static mat4x4* vb_ddqdqdqdq_cached(Frame *frame, Config *dq1, Config *q2, Config *q3, Config *q4)
{
    int i1 = Frame_get_cache_index(frame, dq1);
    int i2 = Frame_get_cache_index(frame, q2);
    int i3 = Frame_get_cache_index(frame, q3);
    int i4 = Frame_get_cache_index(frame, q4);
    if(i1 == -1 || i2 == -1 || i3 == -1 || i4 == -1)
	return &zero_mat4x4;
    else {
	assert(i2 <= i3 && i3 <= i4);
	return (mat4x4*)VB_DDQDQDQDQ(i1, i2, i3, i4);
    }
}

/***********************************************************************
 * Public Frame_* functions.
 *
 * These are the functions for accessing frame
 * position/configuration/derivative data.  They ensure the cache
 * contains updated values and handle finding the values in the cache
 * tables.  This involves making sure the configuration variable
 * affects the frame, and ordering the configuration variables to only
 * access the upper diagonal section of symmetric tables.
 **********************************************************************/  
 
// 
//
// ///////////////////////////////////////
//////////////////////////////////////////
// caching external interface functions //
//////////////////////////////////////////

#define swap(a,b) do{ temp = a; a = b; b = temp; } while(0)
static inline void sort_configs_2(Config **q1, Config **q2) {
    Config *temp;
    if((*q2)->config_gen < (*q1)->config_gen)
	swap(*q1, *q2);
}

static inline void sort_configs_3(Config **q1, Config **q2, Config **q3) {
    Config *temp;
    if((*q2)->config_gen < (*q1)->config_gen)
	swap(*q1, *q2);
    if((*q3)->config_gen < (*q2)->config_gen) {
	swap(*q2, *q3);
	if((*q2)->config_gen < (*q1)->config_gen)
	    swap(*q1, *q2);
    }
}

static inline void sort_configs_4(Config **q1, Config **q2, Config **q3, Config **q4) {
    Config *temp;
    
    if((*q2)->config_gen < (*q1)->config_gen)
	swap(*q1, *q2);
    if((*q4)->config_gen < (*q3)->config_gen)
	swap(*q3, *q4);
    if((*q3)->config_gen < (*q2)->config_gen) {
	swap(*q2, *q3);
	if((*q2)->config_gen < (*q1)->config_gen)
	    swap(*q1, *q2);
	if((*q4)->config_gen < (*q3)->config_gen)
	    swap(*q3, *q4);
	if((*q3)->config_gen < (*q2)->config_gen)
	    swap(*q2, *q3);
    }
}
#undef swap

mat4x4* Frame_lg(Frame *frame)
{
    if(!(frame->system->cache & SYSTEM_CACHE_LG))
	build_lg_cache(frame->system);    
    return &frame->lg;
}

mat4x4* Frame_lg_inv(Frame *frame)
{
    if(!(frame->system->cache & SYSTEM_CACHE_LG))
	build_lg_cache(frame->system);
    return &frame->lg_inv;
}

mat4x4* Frame_lg_dq(Frame *frame)
{
    if(!(frame->system->cache & SYSTEM_CACHE_LG))
	build_lg_cache(frame->system);
    return &frame->lg_dq;
}

mat4x4* Frame_lg_inv_dq(Frame *frame)
{
    if(!(frame->system->cache & SYSTEM_CACHE_LG))
	build_lg_cache(frame->system);
    return &frame->lg_inv_dq;
}

mat4x4* Frame_lg_dqdq(Frame *frame)
{
    if(!(frame->system->cache & SYSTEM_CACHE_LG))
	build_lg_cache(frame->system);
    return &frame->lg_dqdq;
}

mat4x4* Frame_lg_inv_dqdq(Frame *frame)
{
    if(!(frame->system->cache & SYSTEM_CACHE_LG))
	build_lg_cache(frame->system);
    return &frame->lg_inv_dqdq;
}

mat4x4* Frame_lg_dqdqdq(Frame *frame)
{
    if(!(frame->system->cache & SYSTEM_CACHE_LG))
	build_lg_cache(frame->system);
    return &frame->lg_dqdqdq;
}

mat4x4* Frame_lg_inv_dqdqdq(Frame *frame)
{
    if(!(frame->system->cache & SYSTEM_CACHE_LG))
	build_lg_cache(frame->system);
    return &frame->lg_inv_dqdqdq;
}

mat4x4* Frame_lg_dqdqdqdq(Frame *frame)
{
    if(!(frame->system->cache & SYSTEM_CACHE_LG))
	build_lg_cache(frame->system);
    return &frame->lg_dqdqdqdq;
}

mat4x4* Frame_lg_inv_dqdqdqdq(Frame *frame)
{
    if(!(frame->system->cache & SYSTEM_CACHE_LG))
	build_lg_cache(frame->system);
    return &frame->lg_inv_dqdqdqdq;
}

mat4x4* Frame_twist_hat(Frame *frame)
{
    if(!(frame->system->cache & SYSTEM_CACHE_LG))
	build_lg_cache(frame->system);
    return &frame->twist_hat;
}

mat4x4* Frame_g(Frame *frame)
{
    if(!(frame->system->cache & SYSTEM_CACHE_G))
	build_g_cache(frame->system);
    return &frame->g;
}

mat4x4* Frame_g_dq(Frame *frame, Config *q1)
{
    if(!(frame->system->cache & SYSTEM_CACHE_G_DQ))
	build_g_dq_cache(frame->system);
    return g_dq_cached(frame, q1);
}

mat4x4* Frame_g_dqdq(Frame *frame, Config *q1, Config *q2)
{
    if(!(frame->system->cache & SYSTEM_CACHE_G_DQDQ))
	build_g_dqdq_cache(frame->system);
    sort_configs_2(&q1, &q2);
    return g_dqdq_cached(frame, q1, q2);
}

mat4x4* Frame_g_dqdqdq(Frame *frame, Config *q1, Config *q2, Config *q3)
{
    if(!(frame->system->cache & SYSTEM_CACHE_G_DQDQDQ))
	build_g_dqdqdq_cache(frame->system);
    sort_configs_3(&q1, &q2, &q3);
    return g_dqdqdq_cached(frame, q1, q2, q3);
}

mat4x4* Frame_g_dqdqdqdq(Frame *frame, Config *q1, Config *q2, Config *q3, Config *q4)
{
    if(!(frame->system->cache & SYSTEM_CACHE_G_DQDQDQDQ)) 
	build_g_dqdqdqdq_cache(frame->system);
    sort_configs_4(&q1, &q2, &q3, &q4);
    return g_dqdqdqdq_cached(frame, q1, q2, q3, q4);
}

mat4x4* Frame_g_inv(Frame *frame)
{
    if(!(frame->system->cache & SYSTEM_CACHE_G_INV))
       build_g_inv_cache(frame->system);    
    return &frame->g_inv;
}

mat4x4* Frame_g_inv_dq(Frame *frame, Config *q1)
{
    if(!(frame->system->cache & SYSTEM_CACHE_G_INV_DQ))
	build_g_inv_dq_cache(frame->system);
    return g_inv_dq_cached(frame, q1);
}

mat4x4* Frame_g_inv_dqdq(Frame *frame, Config *q1, Config *q2)
{
    if(!(frame->system->cache & SYSTEM_CACHE_G_INV_DQDQ))
	build_g_inv_dqdq_cache(frame->system);
    sort_configs_2(&q1, &q2);
    return g_inv_dqdq_cached(frame, q1, q2);
}

vec4* Frame_p(Frame *frame)
{
    if(!(frame->system->cache & SYSTEM_CACHE_G))
	build_g_cache(frame->system);

    return &frame->p;
}

vec4* Frame_p_dq(Frame *frame, Config *q1)
{
    if(!(frame->system->cache & SYSTEM_CACHE_G_DQ))
	build_g_dq_cache(frame->system);
    return p_dq_cached(frame, q1);
}

vec4* Frame_p_dqdq(Frame *frame, Config *q1, Config *q2)
{
    if(!(frame->system->cache & SYSTEM_CACHE_G_DQDQ))
	build_g_dqdq_cache(frame->system);
    sort_configs_2(&q1, &q2);
    return p_dqdq_cached(frame, q1, q2);
}

vec4* Frame_p_dqdqdq(Frame *frame, Config *q1, Config *q2, Config *q3)
{
    if(!(frame->system->cache & SYSTEM_CACHE_G_DQDQDQ))
	build_g_dqdqdq_cache(frame->system);
    sort_configs_3(&q1, &q2, &q3);
    return p_dqdqdq_cached(frame, q1, q2, q3);
}

vec4* Frame_p_dqdqdqdq(Frame *frame, Config *q1, Config *q2, Config *q3, Config *q4)
{
    if(!(frame->system->cache & SYSTEM_CACHE_G_DQDQDQDQ))
	build_g_dqdqdqdq_cache(frame->system);
    sort_configs_4(&q1, &q2, &q3, &q4);
    return p_dqdqdqdq_cached(frame, q1, q2, q3, q4);
}

mat4x4* Frame_vb(Frame *frame)
{
    if(!(frame->system->cache & SYSTEM_CACHE_VB))
	build_vb_cache(frame->system);
    return vb_cached(frame);
}

mat4x4* Frame_vb_dq(Frame *frame, Config *q1)
{
    if(!(frame->system->cache & SYSTEM_CACHE_VB_DQ))
	build_vb_dq_cache(frame->system);
    return vb_dq_cached(frame, q1);
}

mat4x4* Frame_vb_dqdq(Frame *frame, Config *q1, Config *q2)
{
    if(!(frame->system->cache & SYSTEM_CACHE_VB_DQDQ))
	build_vb_dqdq_cache(frame->system);
    sort_configs_2(&q1, &q2);
    return vb_dqdq_cached(frame, q1, q2);
}

mat4x4* Frame_vb_dqdqdq(Frame *frame, Config *q1, Config *q2, Config *q3)
{
    if(!(frame->system->cache & SYSTEM_CACHE_VB_DQDQDQ))
	build_vb_dqdqdq_cache(frame->system);
    sort_configs_3(&q1, &q2, &q3);
    return vb_dqdqdq_cached(frame, q1, q2, q3);
}

mat4x4* Frame_vb_ddq(Frame *frame, Config *dq1)
{
    if(!(frame->system->cache & SYSTEM_CACHE_VB_DDQ))
	build_vb_ddq_cache(frame->system);
    return vb_ddq_cached(frame, dq1);
}

mat4x4* Frame_vb_ddqdq(Frame *frame, Config *dq1, Config *q2)
{
    if(!(frame->system->cache & SYSTEM_CACHE_VB_DDQDQ))
	build_vb_ddqdq_cache(frame->system);

    return vb_ddqdq_cached(frame, dq1, q2);
}

mat4x4* Frame_vb_ddqdqdq(Frame *frame, Config *dq1, Config *q2, Config *q3)
{
    if(!(frame->system->cache & SYSTEM_CACHE_VB_DDQDQDQ))
	build_vb_ddqdqdq_cache(frame->system);
    sort_configs_2(&q2, &q3);
    return vb_ddqdqdq_cached(frame, dq1, q2, q3);
}

mat4x4* Frame_vb_ddqdqdqdq(Frame *frame, Config *dq1, Config *q2, Config *q3, Config *q4)
{
    if(!(frame->system->cache & SYSTEM_CACHE_VB_DDQDQDQDQ))
	build_vb_ddqdqdqdq_cache(frame->system);
    sort_configs_3(&q2, &q3, &q4);
    return vb_ddqdqdqdq_cached(frame, dq1, q2, q3, q4);
}


/***********************************************************************
 * Python API
 **********************************************************************/


static void dealloc(Frame *self)
{
    Py_CLEAR(self->system);
    Py_CLEAR(self->transform);
    Py_CLEAR(self->config);
    Py_CLEAR(self->parent);
    Py_CLEAR(self->children);
    Py_CLEAR(self->cache_index);

    Py_CLEAR(self->g_dq);
    Py_CLEAR(self->g_dqdq);
    Py_CLEAR(self->g_dqdqdq);
    Py_CLEAR(self->g_dqdqdqdq);
    
    Py_CLEAR(self->g_inv_dq);
    Py_CLEAR(self->g_inv_dqdq);
    
    Py_CLEAR(self->p_dq);
    Py_CLEAR(self->p_dqdq);
    Py_CLEAR(self->p_dqdqdq);
    Py_CLEAR(self->p_dqdqdqdq);
    
    Py_CLEAR(self->vb_dq);
    Py_CLEAR(self->vb_dqdq);
    Py_CLEAR(self->vb_dqdqdq);
    Py_CLEAR(self->vb_ddq);
    Py_CLEAR(self->vb_ddqdq);
    Py_CLEAR(self->vb_ddqdqdq);
    Py_CLEAR(self->vb_ddqdqdqdq);
    self->ob_type->tp_free((PyObject*)self);
}

static int init(Frame *self, PyObject *args, PyObject *kwds)
{
    self->value = 0.0;
    self->mass = 0.0;
    self->Ixx = 0.0;
    self->Iyy = 0.0;
    self->Izz = 0.0;

    eye_mat4x4(self->lg);
    eye_mat4x4(self->lg_inv);
    clear_mat4x4(self->lg_dq);
    clear_mat4x4(self->lg_inv_dq);
    clear_mat4x4(self->lg_dqdq);
    clear_mat4x4(self->lg_inv_dqdq);
    clear_mat4x4(self->lg_dqdqdq);
    clear_mat4x4(self->lg_inv_dqdqdq);
    clear_mat4x4(self->lg_dqdqdqdq);
    clear_mat4x4(self->lg_inv_dqdqdqdq);
    clear_mat4x4(self->twist_hat);    
    eye_mat4x4(self->g);
    set_vec4(self->p, 0.0, 0.0, 0.0, 1.0);
    clear_mat4x4(self->vb);

    self->cos_param = 0.0;
    self->sin_param = 0.0;
    self->multiply_gk = tx_multiply_gk;
    self->add_sandwich_gk = tx_add_sandwich_gk;

    return 0;
}

/* Updates internal structures when the transform type is changed. */
static PyObject* update_transform(Frame *self)
{
    if(self->transform == TREP_TX) {
	self->multiply_gk = tx_multiply_gk;
	self->add_sandwich_gk = tx_add_sandwich_gk;
    }
    else if(self->transform == TREP_TY) {
	self->multiply_gk = ty_multiply_gk;
	self->add_sandwich_gk = ty_add_sandwich_gk;
    }
    else if(self->transform == TREP_TZ) {
	self->multiply_gk = tz_multiply_gk;
	self->add_sandwich_gk = tz_add_sandwich_gk;
    }
    else if(self->transform == TREP_RX) {
	self->multiply_gk = rx_multiply_gk;
	self->add_sandwich_gk = rx_add_sandwich_gk;
    }
    else if(self->transform == TREP_RY) {
	self->multiply_gk = ry_multiply_gk;
	self->add_sandwich_gk = ry_add_sandwich_gk;
    }
    else if(self->transform == TREP_RZ) {
	self->multiply_gk = rz_multiply_gk;
	self->add_sandwich_gk = rz_add_sandwich_gk;
    }
    else if(self->transform == TREP_CONST_SE3) {
	self->multiply_gk = const_se3_multiply_gk;
	self->add_sandwich_gk = const_se3_add_sandwich_gk;
    }
    
    /* build_lg_cache_int doesn't set the constant elements of local
     * transforms, so we need to be sure they are initialized correctly.
     */
    eye_mat4x4(self->lg);
    eye_mat4x4(self->lg_inv);
    clear_mat4x4(self->lg_dq);
    clear_mat4x4(self->lg_inv_dq);
    clear_mat4x4(self->lg_dqdq);
    clear_mat4x4(self->lg_inv_dqdq);
    clear_mat4x4(self->lg_dqdqdq);
    clear_mat4x4(self->lg_inv_dqdqdq);
    clear_mat4x4(self->lg_dqdqdqdq);
    clear_mat4x4(self->lg_inv_dqdqdqdq);
    clear_mat4x4(self->twist_hat);

    /* Changing the transform type outdates the cache as well. */
    self->system->cache = SYSTEM_CACHE_NONE;

    Py_RETURN_NONE;
}

static PyObject* set_SE3(Frame *self, PyObject *args)
{
    if(!PyArg_ParseTuple(args, "(ddd)(ddd)(ddd)(ddd)",
			 &(self->lg[0][0]), &(self->lg[1][0]), &(self->lg[2][0]), 
			 &(self->lg[0][1]), &(self->lg[1][1]), &(self->lg[2][1]), 
			 &(self->lg[0][2]), &(self->lg[1][2]), &(self->lg[2][2]), 
			 &(self->lg[0][3]), &(self->lg[1][3]), &(self->lg[2][3])))
        return NULL;
    invert_se3(self->lg_inv, self->lg);    
    Py_RETURN_NONE;
}

static PyObject* lg(Frame* self)          { return array_from_mat4x4(*Frame_lg(self)); }
static PyObject* lg_dq(Frame* self)       { return array_from_mat4x4(*Frame_lg_dq(self)); }
static PyObject* lg_dqdq(Frame* self)     { return array_from_mat4x4(*Frame_lg_dqdq(self)); }
static PyObject* lg_dqdqdq(Frame* self)   { return array_from_mat4x4(*Frame_lg_dqdqdq(self)); }
static PyObject* lg_dqdqdqdq(Frame* self) { return array_from_mat4x4(*Frame_lg_dqdqdqdq(self)); }

static PyObject* lg_inv(Frame* self)          { return array_from_mat4x4(*Frame_lg_inv(self)); }
static PyObject* lg_inv_dq(Frame* self)       { return array_from_mat4x4(*Frame_lg_inv_dq(self)); }
static PyObject* lg_inv_dqdq(Frame* self)     { return array_from_mat4x4(*Frame_lg_inv_dqdq(self)); }
static PyObject* lg_inv_dqdqdq(Frame* self)   { return array_from_mat4x4(*Frame_lg_inv_dqdqdq(self)); }
static PyObject* lg_inv_dqdqdqdq(Frame* self) { return array_from_mat4x4(*Frame_lg_inv_dqdqdqdq(self)); }

static PyObject* g(Frame* self)          { return array_from_mat4x4(*Frame_g(self)); }
static PyObject* g_inv(Frame* self)      { return array_from_mat4x4(*Frame_g_inv(self)); }
static PyObject* p(Frame* self)          { return array_from_vec4(*Frame_p(self)); }
static PyObject* twist_hat(Frame* self)  { return array_from_mat4x4(*Frame_twist_hat(self)); }
static PyObject* vb(Frame* self)         { return array_from_mat4x4(*Frame_vb(self));}

static PyMethodDef methods_list[] = {
    {"_update_transform", (PyCFunction)update_transform, METH_NOARGS, trep_internal_doc},
    {"_set_SE3", (PyCFunction)set_SE3, METH_VARARGS, trep_internal_doc},

    {"_lg", (PyCFunction)lg, METH_NOARGS, trep_internal_doc},
    {"_lg_dq", (PyCFunction)lg_dq, METH_NOARGS, trep_internal_doc},
    {"_lg_dqdq", (PyCFunction)lg_dqdq, METH_NOARGS, trep_internal_doc},
    {"_lg_dqdqdq", (PyCFunction)lg_dqdqdq, METH_NOARGS, trep_internal_doc},
    {"_lg_dqdqdqdq", (PyCFunction)lg_dqdqdqdq, METH_NOARGS, trep_internal_doc},

    {"_lg_inv", (PyCFunction)lg_inv, METH_NOARGS, trep_internal_doc},
    {"_lg_inv_dq", (PyCFunction)lg_inv_dq, METH_NOARGS, trep_internal_doc},
    {"_lg_inv_dqdq", (PyCFunction)lg_inv_dqdq, METH_NOARGS, trep_internal_doc},
    {"_lg_inv_dqdqdq", (PyCFunction)lg_inv_dqdqdq, METH_NOARGS, trep_internal_doc},
    {"_lg_inv_dqdqdqdq", (PyCFunction)lg_inv_dqdqdqdq, METH_NOARGS, trep_internal_doc},

    {"_twist_hat", (PyCFunction)twist_hat, METH_NOARGS, trep_internal_doc},
    {"_g", (PyCFunction)g, METH_NOARGS, trep_internal_doc},
    {"_g_inv", (PyCFunction)g_inv, METH_NOARGS, trep_internal_doc},
    {"_p", (PyCFunction)p, METH_NOARGS, trep_internal_doc},
    {"_vb", (PyCFunction)vb, METH_NOARGS, trep_internal_doc},
    {NULL}  /* Sentinel */
};

static PyMemberDef members_list[] = {
    {"_system", T_OBJECT_EX, offsetof(Frame, system), 0, trep_internal_doc},
    {"_config", T_OBJECT_EX, offsetof(Frame, config), 0, trep_internal_doc},
    {"_parent", T_OBJECT_EX, offsetof(Frame, parent), 0, trep_internal_doc},
    {"_transform", T_OBJECT_EX, offsetof(Frame, transform), 0, trep_internal_doc},
    {"_children", T_OBJECT_EX, offsetof(Frame, children), 0, trep_internal_doc},
    {"_value", T_DOUBLE, offsetof(Frame, value), 0, trep_internal_doc},
    {"_cache_size", T_INT, offsetof(Frame, cache_size), 0, trep_internal_doc},
    {"_cache_index", T_OBJECT_EX, offsetof(Frame, cache_index), 0, trep_internal_doc},
    {"_mass", T_DOUBLE, offsetof(Frame, mass), 0, trep_internal_doc},
    {"_Ixx", T_DOUBLE, offsetof(Frame, Ixx), 0, trep_internal_doc},
    {"_Iyy", T_DOUBLE, offsetof(Frame, Iyy), 0, trep_internal_doc},
    {"_Izz", T_DOUBLE, offsetof(Frame, Izz), 0, trep_internal_doc},
    {"_g_dq",       T_OBJECT_EX, offsetof(Frame, g_dq),       0, trep_internal_doc},
    {"_g_dqdq",     T_OBJECT_EX, offsetof(Frame, g_dqdq),     0, trep_internal_doc},
    {"_g_dqdqdq",   T_OBJECT_EX, offsetof(Frame, g_dqdqdq),   0, trep_internal_doc},
    {"_g_dqdqdqdq", T_OBJECT_EX, offsetof(Frame, g_dqdqdqdq), 0, trep_internal_doc},
    
    {"_g_inv_dq",   T_OBJECT_EX, offsetof(Frame, g_inv_dq),   0, trep_internal_doc},
    {"_g_inv_dqdq", T_OBJECT_EX, offsetof(Frame, g_inv_dqdq), 0, trep_internal_doc},
    
    {"_p_dq",       T_OBJECT_EX, offsetof(Frame, p_dq),       0, trep_internal_doc},
    {"_p_dqdq",     T_OBJECT_EX, offsetof(Frame, p_dqdq),     0, trep_internal_doc},
    {"_p_dqdqdq",   T_OBJECT_EX, offsetof(Frame, p_dqdqdq),   0, trep_internal_doc},
    {"_p_dqdqdqdq", T_OBJECT_EX, offsetof(Frame, p_dqdqdqdq), 0, trep_internal_doc},
    
    {"_vb_dq",        T_OBJECT_EX, offsetof(Frame, vb_dq),     0, trep_internal_doc},
    {"_vb_dqdq",      T_OBJECT_EX, offsetof(Frame, vb_dqdq),   0, trep_internal_doc},
    {"_vb_dqdqdq",    T_OBJECT_EX, offsetof(Frame, vb_dqdqdq), 0, trep_internal_doc},    
    {"_vb_ddq",       T_OBJECT_EX, offsetof(Frame, vb_ddq),       0, trep_internal_doc},
    {"_vb_ddqdq",     T_OBJECT_EX, offsetof(Frame, vb_ddqdq),     0, trep_internal_doc},
    {"_vb_ddqdqdq",   T_OBJECT_EX, offsetof(Frame, vb_ddqdqdq),   0, trep_internal_doc},
    {"_vb_ddqdqdqdq", T_OBJECT_EX, offsetof(Frame, vb_ddqdqdqdq), 0, trep_internal_doc},
    
    {NULL}  /* Sentinel */
};

PyTypeObject FrameType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_trep._Frame",            /*tp_name*/
    sizeof(Frame),             /*tp_basicsize*/
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
