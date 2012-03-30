#include <Python.h>
#include "trep_internal.h"

// Used to return zero values
mat4x4 zero_mat4x4 = {
    {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};    
vec4 zero_vec4 = {0.0, 0.0, 0.0, 0.0};

PyObject* array_from_mat4x4(mat4x4 mat) {
    double *data = NULL;
    PyObject *array = NULL;
    npy_intp size[2] = {4, 4};

    array = PyArray_SimpleNew(2, size, NPY_DOUBLE);
    if(array == NULL)
        return NULL;

    data = (double*)PyArray_DATA(array);
    memcpy(data, mat, 16*sizeof(double));
    return array;    
}

PyObject* array_from_vec4(vec4 vec) {
    double *data = NULL;
    PyObject *array = NULL;
    npy_intp size[1] = {4};

    array = PyArray_SimpleNew(1, size, NPY_DOUBLE);
    if(array == NULL)
        return NULL;

    data = (double*)PyArray_DATA(array);
    memcpy(data, vec, 4*sizeof(double));
    return array;    
}

void copy_vec4(vec4 dest, vec4 src)
{
    dest[0] = src[0];
    dest[1] = src[1];
    dest[2] = src[2];
    dest[2] = src[3];    
}

void set_vec4(vec4 dest, double x, double y, double z, double w)
{
    dest[0] = x;
    dest[1] = y;
    dest[2] = z;
    dest[3] = w;
}

void clear_vec4(vec4 dest)
{
    dest[0] = 0.0;
    dest[1] = 0.0;
    dest[2] = 0.0;
    dest[3] = 0.0;
}

double dot_vec4(vec4 op1, vec4 op2)
{
    return op1[0]*op2[0] + op1[1]*op2[1] + op1[2]*op2[2] + op1[3]*op2[3];
}

void sub_vec4(vec4 dest, vec4 op1, vec4 op2)
{
    dest[0] = op1[0] - op2[0];
    dest[1] = op1[1] - op2[1];
    dest[2] = op1[2] - op2[2];
    dest[3] = op1[3] - op2[3];
}

void invert_se3(mat4x4 dest, mat4x4 src)
{
    /* The inverse of a SE3 matrix is the SO(3) component transposed,
     * the translation part negated, and the last row left untouched.
     */
    
    dest[0][0] =  src[0][0];
    dest[0][1] =  src[1][0];
    dest[0][2] =  src[2][0];
    
    dest[1][0] =  src[0][1];
    dest[1][1] =  src[1][1];
    dest[1][2] =  src[2][1];
    
    dest[2][0] =  src[0][2];
    dest[2][1] =  src[1][2];
    dest[2][2] =  src[2][2];
       
    dest[0][3] = -(dest[0][0] * src[0][3] +
		   dest[0][1] * src[1][3] +
		   dest[0][2] * src[2][3]);
    dest[1][3] = -(dest[1][0] * src[0][3] +
		   dest[1][1] * src[1][3] +
		   dest[1][2] * src[2][3]);
    dest[2][3] = -(dest[2][0] * src[0][3] +
		   dest[2][1] * src[1][3] +
		   dest[2][2] * src[2][3]);

    dest[3][0] =  src[3][0];
    dest[3][1] =  src[3][1];
    dest[3][2] =  src[3][2];
    dest[3][3] =  src[3][3];
}

void mul_mm4(mat4x4 dest, mat4x4 op1, mat4x4 op2)
{
    dest[0][0] =
	op1[0][0]*op2[0][0] + op1[0][1]*op2[1][0] +
	op1[0][2]*op2[2][0] + op1[0][3]*op2[3][0];
    dest[0][1] =
	op1[0][0]*op2[0][1] + op1[0][1]*op2[1][1] +
	op1[0][2]*op2[2][1] + op1[0][3]*op2[3][1];
    dest[0][2] =
	op1[0][0]*op2[0][2] + op1[0][1]*op2[1][2] +
	op1[0][2]*op2[2][2] + op1[0][3]*op2[3][2];
    dest[0][3] =
	op1[0][0]*op2[0][3] + op1[0][1]*op2[1][3] +
	op1[0][2]*op2[2][3] + op1[0][3]*op2[3][3];
    dest[1][0] =
	op1[1][0]*op2[0][0] + op1[1][1]*op2[1][0] +
	op1[1][2]*op2[2][0] + op1[1][3]*op2[3][0];
    dest[1][1] =
	op1[1][0]*op2[0][1] + op1[1][1]*op2[1][1] +
	op1[1][2]*op2[2][1] + op1[1][3]*op2[3][1];
    dest[1][2] =
	op1[1][0]*op2[0][2] + op1[1][1]*op2[1][2] +
	op1[1][2]*op2[2][2] + op1[1][3]*op2[3][2];
    dest[1][3] =
	op1[1][0]*op2[0][3] + op1[1][1]*op2[1][3] +
	op1[1][2]*op2[2][3] + op1[1][3]*op2[3][3];
    dest[2][0] =
	op1[2][0]*op2[0][0] + op1[2][1]*op2[1][0] +
	op1[2][2]*op2[2][0] + op1[2][3]*op2[3][0];
    dest[2][1] =
	op1[2][0]*op2[0][1] + op1[2][1]*op2[1][1] +
	op1[2][2]*op2[2][1] + op1[2][3]*op2[3][1];
    dest[2][2] =
	op1[2][0]*op2[0][2] + op1[2][1]*op2[1][2] +
	op1[2][2]*op2[2][2] + op1[2][3]*op2[3][2];
    dest[2][3] =
	op1[2][0]*op2[0][3] + op1[2][1]*op2[1][3] +
	op1[2][2]*op2[2][3] + op1[2][3]*op2[3][3];
    dest[3][0] =
	op1[3][0]*op2[0][0] + op1[3][1]*op2[1][0] +
	op1[3][2]*op2[2][0] + op1[3][3]*op2[3][0];
    dest[3][1] =
	op1[3][0]*op2[0][1] + op1[3][1]*op2[1][1] +
	op1[3][2]*op2[2][1] + op1[3][3]*op2[3][1];	
    dest[3][2] =
	op1[3][0]*op2[0][2] + op1[3][1]*op2[1][2] +
	op1[3][2]*op2[2][2] + op1[3][3]*op2[3][2];
    dest[3][3] =
	op1[3][0]*op2[0][3] + op1[3][1]*op2[1][3] +
	op1[3][2]*op2[2][3] + op1[3][3]*op2[3][3];	
}

void mul_mv4(vec4 dest, mat4x4 m, vec4 v)
{
    dest[0] = m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2] + m[0][3]*v[3];
    dest[1] = m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2] + m[1][3]*v[3];
    dest[2] = m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2] + m[2][3]*v[3];
    dest[3] = m[3][0]*v[0] + m[3][1]*v[1] + m[3][2]*v[2] + m[3][3]*v[3];
}

void mul_dm4(mat4x4 dest, double op1, mat4x4 op2)
{
    dest[0][0] = op1*op2[0][0];
    dest[0][1] = op1*op2[0][1];
    dest[0][2] = op1*op2[0][2];
    dest[0][3] = op1*op2[0][3];
    dest[1][0] = op1*op2[1][0];
    dest[1][1] = op1*op2[1][1];
    dest[1][2] = op1*op2[1][2];
    dest[1][3] = op1*op2[1][3];
    dest[2][0] = op1*op2[2][0];
    dest[2][1] = op1*op2[2][1];
    dest[2][2] = op1*op2[2][2];
    dest[2][3] = op1*op2[2][3];
    dest[3][0] = op1*op2[3][0];
    dest[3][1] = op1*op2[3][1];
    dest[3][2] = op1*op2[3][2];
    dest[3][3] = op1*op2[3][3];
}

void add_mm4(mat4x4 dest, mat4x4 op1, mat4x4 op2)
{
    dest[0][0] = op1[0][0] + op2[0][0];
    dest[0][1] = op1[0][1] + op2[0][1];
    dest[0][2] = op1[0][2] + op2[0][2];
    dest[0][3] = op1[0][3] + op2[0][3];
    dest[1][0] = op1[1][0] + op2[1][0];
    dest[1][1] = op1[1][1] + op2[1][1];
    dest[1][2] = op1[1][2] + op2[1][2];
    dest[1][3] = op1[1][3] + op2[1][3];
    dest[2][0] = op1[2][0] + op2[2][0];
    dest[2][1] = op1[2][1] + op2[2][1];
    dest[2][2] = op1[2][2] + op2[2][2];
    dest[2][3] = op1[2][3] + op2[2][3];
    dest[3][0] = op1[3][0] + op2[3][0];
    dest[3][1] = op1[3][1] + op2[3][1];
    dest[3][2] = op1[3][2] + op2[3][2];
    dest[3][3] = op1[3][3] + op2[3][3];
}

void sub_mm4(mat4x4 dest, mat4x4 op1, mat4x4 op2)
{
    dest[0][0] = op1[0][0] - op2[0][0];
    dest[0][1] = op1[0][1] - op2[0][1];
    dest[0][2] = op1[0][2] - op2[0][2];
    dest[0][3] = op1[0][3] - op2[0][3];
    dest[1][0] = op1[1][0] - op2[1][0];
    dest[1][1] = op1[1][1] - op2[1][1];
    dest[1][2] = op1[1][2] - op2[1][2];
    dest[1][3] = op1[1][3] - op2[1][3];
    dest[2][0] = op1[2][0] - op2[2][0];
    dest[2][1] = op1[2][1] - op2[2][1];
    dest[2][2] = op1[2][2] - op2[2][2];
    dest[2][3] = op1[2][3] - op2[2][3];
    dest[3][0] = op1[3][0] - op2[3][0];
    dest[3][1] = op1[3][1] - op2[3][1];
    dest[3][2] = op1[3][2] - op2[3][2];
    dest[3][3] = op1[3][3] - op2[3][3];
}

void copy_mat4x4(mat4x4 dest, mat4x4 src)
{
    dest[0][0] = src[0][0];
    dest[0][1] = src[0][1];
    dest[0][2] = src[0][2];
    dest[0][3] = src[0][3];
    dest[1][0] = src[1][0];
    dest[1][1] = src[1][1];
    dest[1][2] = src[1][2];
    dest[1][3] = src[1][3];
    dest[2][0] = src[2][0];
    dest[2][1] = src[2][1];
    dest[2][2] = src[2][2];
    dest[2][3] = src[2][3];
    dest[3][0] = src[3][0];
    dest[3][1] = src[3][1];
    dest[3][2] = src[3][2];
    dest[3][3] = src[3][3];
}

void eye_mat4x4(mat4x4 mat)
{
    mat[0][0] = 1.0;
    mat[0][1] = 0.0;
    mat[0][2] = 0.0;
    mat[0][3] = 0.0;
    mat[1][0] = 0.0;
    mat[1][1] = 1.0;
    mat[1][2] = 0.0;
    mat[1][3] = 0.0;
    mat[2][0] = 0.0;
    mat[2][1] = 0.0;
    mat[2][2] = 1.0;
    mat[2][3] = 0.0;
    mat[3][0] = 0.0;
    mat[3][1] = 0.0;
    mat[3][2] = 0.0;
    mat[3][3] = 1.0;
}

void clear_mat4x4(mat4x4 mat)
{    
    memset(mat, 0, sizeof(mat4x4));
}

void unhat(vec6 dest, mat4x4 src)
{
    //assert(SO3 part is skew symmetric);
    dest[0] = src[0][3];
    dest[1] = src[1][3];
    dest[2] = src[2][3];
    dest[3] = src[2][1];
    dest[4] = src[0][2];
    dest[5] = src[1][0];    
}

void copy_np_matrix(PyArrayObject *dest, PyArrayObject *src, int rows, int cols)
{
    int i1, i2;

    assert(dest != NULL);
    assert(src != NULL);
    assert(PyArray_NDIM(dest) == 2);
    //assert(PyArray_SAMESHAPE(dest, src));

    for(i1 = 0; i1 < rows; i1++)
        for(i2 = 0; i2 < cols; i2++) 
            IDX2_DBL(dest, i1, i2) = IDX2_DBL(src, i1, i2);
}

void transpose_np_matrix(PyArrayObject *dest, PyArrayObject *src)
{
    int i1, i2;
    int rows, cols;

    assert(dest != NULL);
    assert(src != NULL);
    assert(PyArray_NDIM(dest) == 2);
    assert(PyArray_DIMS(dest)[0] == PyArray_DIMS(src)[1]);
    assert(PyArray_DIMS(dest)[1] == PyArray_DIMS(src)[0]);

    rows = PyArray_DIMS(src)[0];
    cols = PyArray_DIMS(src)[1];
    
    for(i1 = 0; i1 < rows; i1++)
        for(i2 = 0; i2 < cols; i2++) 
            IDX2_DBL(dest, i2, i1) = IDX2_DBL(src, i1, i2);
}

void copy_vector(double *dest, double *src, int length)
{
    assert(!(dest == 0 || src == 0));
    memcpy(dest, src, sizeof(double)*length);
}

double norm_vector(double *vec, int length)
{
    double norm = 0.0;
    int i;

    assert(!(length != 0 && vec == NULL));

    for(i = 0; i < length; i++)
	norm += vec[i]*vec[i];
    return sqrt(norm);
}

int LU_decomp(PyArrayObject *A, int n, PyArrayObject *np_index, double tolerance)
{
    int i, j, k;
    double pivot_value; /* Used for partial pivoting */
    int pivot_index = 0;
    int int_temp;
    int *index = (int*)PyArray_DATA(np_index);
    double *array_temp = (double*)malloc(sizeof(double)*n);
    double *scales = (double*)malloc(sizeof(double)*n);   /* Used for implicit pivoting */

    /* Find the largest elements in each row for implicit pivoting. */
    for(i = 0; i < n; i++)
    {
	scales[i] = -1.0;
	for(j = 0; j < n; j++)
	{
	    if(fabs(IDX2_DBL(A, i, j)) > scales[i])
		scales[i] = fabs(IDX2_DBL(A, i, j));
	}
	scales[i] = 1.0/scales[i];
	index[i] = i;
    }

    /* Crout's algorithm iterates over the matrix columns. */
    for(j = 0; j < n; j++)
    {
	/* First solve for Beta values in the current column.  No
	 * pivoting is involved. (Do not find Bjj yet).
	 */
	for(i = 0; i < j; i++)
	{
	    /* Beta_ij = A_ij - (Sum k=0 to i-1 of alpha_ik * Beta_kj) */
	    for(k = 0; k <= i-1; k++)
		IDX2_DBL(A, i, j) -= IDX2_DBL(A, i, k)*IDX2_DBL(A, k, j);
	}

	/* Now we calculate Bjj and alpha_*j*Bjj.  Along the way, keep
	 * track of the largest element seen.  We'll use this as the
	 * pivot, swapping the rows and applying the Bjj division
	 * afterwards.
	 */
	pivot_value = -1.0;
	for(i = j; i < n; i++)
	{
	    /* Calculate the value at this position */
	    for(k = 0; k <= j-1; k++)
		IDX2_DBL(A, i, j) -= IDX2_DBL(A, i, k)*IDX2_DBL(A, k, j);
	    
	    /* Pivot search */
	    if(fabs(IDX2_DBL(A, i, j)*scales[i]) > pivot_value)
	    {
		pivot_value = fabs(IDX2_DBL(A, i, j)*scales[i]);
		pivot_index = i;
	    }
	}

	if(pivot_value <= tolerance)
	{
	    PyErr_Format(PyExc_ValueError,
			 "Matrix is singular and cannot be LU decomposed.");
            goto fail;
	}

	/* Do we need to swap rows? */
	//pivot_index = j;
	if(pivot_index != j)
	{
	    int_temp = index[j];
	    index[j] = index[pivot_index];
	    index[pivot_index] = int_temp;
	
	    /* Swap rows */
            memcpy(array_temp, &IDX2_DBL(A, j, 0), sizeof(double)*n);
            memcpy(&IDX2_DBL(A, j, 0), &IDX2_DBL(A, pivot_index, 0), sizeof(double)*n);
            memcpy(&IDX2_DBL(A, pivot_index, 0), array_temp, sizeof(double)*n);

	    /* Swap scales, but we don't need scales[j] anymore, so
	     * just ignore.
	     */
	    scales[pivot_index] = scales[j];
	}

	/* Divide alphas by Bjj */
	for(i = j+1; i < n; i++)
	    IDX2_DBL(A, i, j) /= IDX2_DBL(A, j, j);
    }
    
    free(scales);
    free(array_temp);
    return 0;

fail:
    free(scales);
    free(array_temp);
    return -1;
}

void LU_solve_vec(PyArrayObject *A, int n, PyArrayObject *np_index, double *b)
{
    int i, j;
    double *x;
    int *index = (int*)PyArray_DATA(np_index);

    x = (double*)malloc(sizeof(double)*n);

    /* Solve L*y = b */
    for(i = 0; i < n; i++)
    {
	/* y_i = b_i - sum j=0 to i=1 alpha_ij * y_j */
	x[i] = b[index[i]];
	for(j = 0; j <= i-1; j++)
	    x[i] -= IDX2_DBL(A, i, j)*x[j];
    }
        
    /* Solve U*x = y */
    for(i = n-1; i >= 0; i--)
    {
	/* x_i = 1/beta_ii * (y_i - sum j=i+1 to n-1  beta_ij * x_j */
	for(j = i+1; j < n; j++)
	    x[i] -= IDX2_DBL(A, i, j)*x[j];
	x[i] = x[i]/IDX2_DBL(A, i, i);
	b[i] = x[i];
    }
    free(x);
}

void LU_solve_mat(PyArrayObject *A, int n, PyArrayObject *np_index, PyArrayObject *b, int m)
{
    int i, j, k;
    double *x;
    int *index = (int*)PyArray_DATA(np_index);

    x = (double*)malloc(sizeof(double)*n);

    for(k = 0; k < m; k++)
    {
	/* Solve L*y = b */
	for(i = 0; i < n; i++)
	{
	    /* y_i = b_i - sum j=0 to i=1 alpha_ij * y_j */
	    x[i] = IDX2_DBL(b, index[i], k);
	    for(j = 0; j <= i-1; j++)
		x[i] -= IDX2_DBL(A, i, j)*x[j];
	}

	/* Solve U*x = y */
	for(i = n-1; i >= 0; i--)
	{
	    /* x_i = 1/beta_ii * (y_i - sum j=i+1 to n-1  beta_ij * x_j */
	    for(j = i+1; j < n; j++)
		x[i] -= IDX2_DBL(A, i, j)*x[j];
	    x[i] = x[i]/IDX2_DBL(A, i, i);
	    IDX2_DBL(b, i, k) = x[i];
	}
    }
    free(x);
}

// Performs dest = op1 * op2 where
//   dest is [rows x 1]
//   op1 is [rows x inner]
//   op2 is [inner x 1]
void mul_matvec_c_np_c(double *dest, int rows, PyArrayObject *op1, double *op2, int inner)
{
    int i = 0;
    int j = 0;
    
    assert(dest != NULL);
    assert(op1 != NULL);
    assert(op2 != NULL);
    assert(PyArray_NDIM(op1) == 2);
    assert(PyArray_DIMS(op1)[0] == rows);
    assert(PyArray_DIMS(op1)[1] >= inner);

    for(i = 0; i < rows; i++)
    {
	dest[i] = 0.0;
	for(j = 0; j < inner; j++)
	    dest[i] += IDX2_DBL(op1, i, j)*op2[j];
    }
}

// Peforms dest = op1 * op2 where
//   dest is [rows x cols]
//   op1 is [rows x inner]
//   op2 is [inner x cols]
void mul_matmat_np_np_np(PyArrayObject *dest, int rows, int cols, PyArrayObject *op1, PyArrayObject *op2, int inner)
{
    int i = 0;
    int j = 0;
    int k = 0;

    assert(dest != NULL);
    assert(op1 != NULL);
    assert(op2 != NULL);
    assert(PyArray_NDIM(op1) == 2);
    assert(PyArray_DIMS(op1)[0] == rows);
    assert(PyArray_DIMS(op1)[1] >= inner);

    for(i = 0; i < rows; i++)
	for(j = 0; j < cols; j++)
	{
	    IDX2_DBL(dest, i, j) = 0.0;
	    for(k = 0; k < inner; k++)
		IDX2_DBL(dest, i, j) += IDX2_DBL(op1, i, k)*IDX2_DBL(op2, k, j);
	}
}
