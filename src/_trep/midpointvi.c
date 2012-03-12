#include <Python.h>
#include "structmember.h"
#include <pthread.h>
#include "trep.h"

#define LU_tolerance 1.0e-20

/***********************
 * multithreading code
 **********************/

//#define MP_DEBUG(...) do { printf(__VA_ARGS__); fflush(stdout) } while(0)
#define MP_DEBUG(...) {}

typedef int (*calc_func_t)(MidpointVI *mvi, int j);
typedef struct mvi_thread_s mvi_thread_t;
typedef struct mvi_threading_s mvi_threading_t;
typedef struct mvi_job_s mvi_job_t;

// This structure describes a job waiting to be processed
struct mvi_job_s {
    calc_func_t func;  // Function to be called to process job
    int i; // Argument passed to func
    mvi_job_t *next;      // Used to link the job in an appropriate list.
};

// This structure describes each individual thread.
struct mvi_thread_s {
    int id;
    pthread_t pthread;
    MidpointVI *mvi;
    mvi_threading_t *pool;    
    // Used to link the thread in the active or idle list
    mvi_thread_t *prev, *next;
};

struct mvi_threading_s {
    pthread_mutex_t mutex;
    pthread_cond_t idle_cond;
    pthread_cond_t queue_cond;
    int num_threads;
    int quit;
        
    mvi_job_t* unused_jobs;   // Forward linked list of available job structures
    mvi_job_t* pending_jobs, *last_pending_job;  // Forward linked list of pending jobs
    mvi_thread_t* active_threads;  // Double linked list of active threads.
    mvi_thread_t* all_threads;     // Array of all available threads
};

/* Add a job to the queue and wake up an idle thread */
static void queue_job(MidpointVI *mvi, calc_func_t func, int i)
{
    mvi_threading_t *pool = mvi->threading;
    mvi_job_t *job;

    if(pool == NULL) {
	/* We're not using multi-threading.  Do the job now. */
	func(mvi, i);
	return;
    }
    
    pthread_mutex_lock(&(pool->mutex));

    /* Fill in an available job structure */
    if(pool->unused_jobs) {
	job = pool->unused_jobs;
	pool->unused_jobs = job->next;
    } else 
	job = (mvi_job_t*)malloc(sizeof(mvi_job_t));

    job->func = func;
    job->i = i;
    job->next = NULL;
    
    if(pool->last_pending_job == NULL) 
	pool->pending_jobs = job;
    else 
	pool->last_pending_job->next = job;
    pool->last_pending_job = job;

    /* Wake up an idle thread if there is one */
    MP_DEBUG("queue_job adds a job, sends a signal\n");
    pthread_cond_signal(&(pool->queue_cond));
    pthread_mutex_unlock(&(pool->mutex));
}


/* Waits until there are no pending jobs and no active threads */
static void wait_for_jobs_to_finish(MidpointVI *mvi)
{
    if(mvi->threading == NULL) {
	// We're not using multi-threading.  Jobs are already done. */
	return;
    }
    
    pthread_mutex_lock(&(mvi->threading->mutex));
    while(mvi->threading->pending_jobs || mvi->threading->active_threads) {
	MP_DEBUG("wait for jobs to finish sleeps\n"); 
	pthread_cond_wait(&(mvi->threading->idle_cond), &(mvi->threading->mutex));
	MP_DEBUG("wait_for_jobs_to_finish wakes! it finds: %p and %p\n",
	       mvi->threading->pending_jobs,
	       mvi->threading->active_threads);
    }
    MP_DEBUG("wait for jobs to finish returns\n"); 
    pthread_mutex_unlock(&(mvi->threading->mutex));    
}

/* Adds thread to the active list */
static void make_thread_active(mvi_thread_t *thread)
{
    mvi_thread_t *old_head;

    old_head = thread->pool->active_threads;
    thread->prev = NULL;
    thread->next = old_head;
    thread->pool->active_threads = thread;
    if(old_head)
	old_head->prev = thread;
}
    
/* Removes the thread from the active list */
static void make_thread_idle(mvi_thread_t *thread)
{
    mvi_thread_t *prev, *next;
    
    prev = thread->prev;
    next = thread->next;

    if(next)
	next->prev = prev;
    if(prev)
	prev->next = next;
    else
        thread->pool->active_threads = next;
    thread->next = NULL;
    thread->prev = NULL;
}

/* Threads call this function to get a new job.  First it looks for
 * jobs in the current queue and returns the first one.  If there are
 * none, it places the thread in an idle state (and sends an idle
 * signal), and then waits on a condition which is signaled when new
 * jobs are added or when it's time to quit.
 */  
static mvi_job_t* get_next_job(mvi_thread_t *thread, mvi_job_t *old_job)
{
    mvi_job_t *new_job;
    mvi_threading_t *pool = thread->pool; 

    pthread_mutex_lock(&(pool->mutex)); 

    MP_DEBUG("thread %d releases job\n", thread->id); 
    
    if(old_job) {
	/* Reclaim this job structure */
	old_job->next = pool->unused_jobs;
	pool->unused_jobs = old_job;
    }

    if(pool->pending_jobs != NULL) {
	/* Found a pending job, take ownership, release mutex, and
	 * send it to the thread. */
	new_job = pool->pending_jobs;
	pool->pending_jobs = new_job->next;
	if(new_job == pool->last_pending_job)
	    pool->last_pending_job = NULL;
	 
	pthread_mutex_unlock(&(pool->mutex));
	return new_job;
    }	
	    
    /* No jobs are waiting, make thread idle and wait for new job. */
    make_thread_idle(thread);
    MP_DEBUG("thread %d sends idle broadcast\n", thread->id); 
    pthread_cond_broadcast(&(pool->idle_cond));
    
    /* Wait for a new job to appear in the queue */
    while(pool->pending_jobs == NULL && !pool->quit) {
	pthread_cond_wait(&(pool->queue_cond), &(pool->mutex));
	MP_DEBUG("a sleeping thread wakes! it finds %p and %d\n", pool->pending_jobs, pool->quit);
    }

    if(pool->quit) {
	pthread_mutex_unlock(&(pool->mutex));
	return NULL;
    }

    make_thread_active(thread);
    new_job = pool->pending_jobs;
    pool->pending_jobs = new_job->next;
    if(new_job == pool->last_pending_job)
	pool->last_pending_job = NULL;
    
    pthread_mutex_unlock(&(pool->mutex));
    return new_job;
}

/* This is the processing function for each thread.  It waits for jobs
 * and executes them.  It quits when get_next_job returns NULL.
 */ 
static void* mvi_thread_func(void *arg)
{
    mvi_thread_t *info = (mvi_thread_t*)arg;
    mvi_job_t *job = NULL;
    MP_DEBUG("thread %d starts.\n", info->id); 
    while(1) {
	job = get_next_job(info, job);
	if(job == NULL)
	    break;
	MP_DEBUG("thread %d received new job!\n", info->id); 
	job->func(info->mvi, job->i);
    }
    MP_DEBUG("thread %d finished.\n", info->id); 
    return NULL;
}

void mvi_init_threading(MidpointVI *mvi, int num_threads)
{
    int i;
    mvi_threading_t *pool;
    
    if(num_threads <= 1) {
	/* Don't use threading. This is mostly useful for debugging and profiling. */
	mvi->threading = NULL;
	return;
    };    

    pool = (mvi_threading_t*)malloc(sizeof(mvi_threading_t));
    mvi->threading = pool;

    pool->num_threads = num_threads;
    pool->all_threads = (mvi_thread_t*)malloc(sizeof(mvi_thread_t)*num_threads);
    
    pthread_mutex_init(&(pool->mutex), NULL);
    pthread_cond_init(&(pool->idle_cond), NULL);
    pthread_cond_init(&(pool->queue_cond), NULL);
    pool->quit = 0;

    pool->unused_jobs = NULL;
    pool->pending_jobs = NULL;
    pool->last_pending_job = NULL;
    pool->active_threads = NULL;

    for(i = 0; i < pool->num_threads; i ++) {
	pool->all_threads[i].id = i;
	pool->all_threads[i].mvi = mvi;
	pool->all_threads[i].pool = pool;
	pool->all_threads[i].prev = NULL;
	pool->all_threads[i].next = NULL;

	/* Threads need to start out in the active list to be
	 * compatible with get_next_job() */
	make_thread_active(&(pool->all_threads[i]));
	pthread_create(&(pool->all_threads[i].pthread), NULL, mvi_thread_func, &(pool->all_threads[i]));
    }    
}

void mvi_kill_threading(MidpointVI *mvi)
{
    int i;
    mvi_job_t *job;
    mvi_threading_t *pool = mvi->threading;
    /* Assume there are no active threads. */

    if(pool == NULL) {
	/* We didn't use threading, just return. */
	return;
    }

    pthread_mutex_lock(&(pool->mutex));
    pool->quit = 1;
    pthread_cond_broadcast(&(pool->queue_cond));
    pthread_mutex_unlock(&(pool->mutex));

    for(i = 0; i < pool->num_threads; i ++) 
	pthread_join(pool->all_threads[i].pthread, NULL);

    /* Release the job structures */
    while(pool->unused_jobs) {
	job = pool->unused_jobs;
	pool->unused_jobs = job->next;
	free(job);
    }

    free(pool->all_threads);
    free(pool);
    mvi->threading = NULL;
}

/*****************************************************************
 * end of multi-thread implementation
 ****************************************************************/


#define Q1(i1)                        F_IDX1_DBL(__q1, i1)
#define Q2(i1)                        F_IDX1_DBL(__q2, i1)
#define P1(i1)                        F_IDX1_DBL(__p1, i1)
#define P2(i1)                        F_IDX1_DBL(__p2, i1)
#define U1(i1)                        F_IDX1_DBL(__u1, i1)
#define LAMBDA1(i1)                   F_IDX1_DBL(__lambda1, i1)

#define DH1T(i1, i2)                  F_IDX2_DBL(Dh1T, i1, i2)
#define DH2(i1, i2)                   F_IDX2_DBL(Dh2, i1, i2)
#define F(i1)                         F_IDX1_DBL(__f, i1) 
#define DF(i1, i2)                    F_IDX2_DBL(Df, i1, i2) 
#define M2_LU(i1, i2)                 F_IDX2_DBL(M2_lu, i1, i2)
#define PROJ_LU(i1, i2)               F_IDX2_DBL(proj_lu, i1, i2)

#define Q2_DQ1(i1, i2)                F_IDX2_DBL(q2_dq1, i1, i2)
#define Q2_DP1(i1, i2)                F_IDX2_DBL(q2_dp1, i1, i2)
#define Q2_DU1(i1, i2)                F_IDX2_DBL(q2_du1, i1, i2)
#define Q2_DK2(i1, i2)                F_IDX2_DBL(q2_dk2, i1, i2)
#define P2_DQ1(i1, i2)                F_IDX2_DBL(p2_dq1, i1, i2)
#define P2_DP1(i1, i2)                F_IDX2_DBL(p2_dp1, i1, i2)
#define P2_DU1(i1, i2)                F_IDX2_DBL(p2_du1, i1, i2)
#define P2_DK2(i1, i2)                F_IDX2_DBL(p2_dk2, i1, i2)
#define L1_DQ1(i1, i2)                F_IDX2_DBL(l1_dq1, i1, i2)
#define L1_DP1(i1, i2)                F_IDX2_DBL(l1_dp1, i1, i2)
#define L1_DU1(i1, i2)                F_IDX2_DBL(l1_du1, i1, i2)
#define L1_DK2(i1, i2)                F_IDX2_DBL(l1_dk2, i1, i2)

#define Q2_DQ1DQ1(i1, i2, i3)         F_IDX3_DBL(q2_dq1dq1, i1, i2, i3)
#define Q2_DQ1DP1(i1, i2, i3)         F_IDX3_DBL(q2_dq1dp1, i1, i2, i3)
#define Q2_DQ1DU1(i1, i2, i3)         F_IDX3_DBL(q2_dq1du1, i1, i2, i3)
#define Q2_DQ1DK2(i1, i2, i3)         F_IDX3_DBL(q2_dq1dk2, i1, i2, i3)
#define Q2_DP1DP1(i1, i2, i3)         F_IDX3_DBL(q2_dp1dp1, i1, i2, i3)
#define Q2_DP1DU1(i1, i2, i3)         F_IDX3_DBL(q2_dp1du1, i1, i2, i3)
#define Q2_DP1DK2(i1, i2, i3)         F_IDX3_DBL(q2_dp1dk2, i1, i2, i3)
#define Q2_DU1DU1(i1, i2, i3)         F_IDX3_DBL(q2_du1du1, i1, i2, i3)
#define Q2_DU1DK2(i1, i2, i3)         F_IDX3_DBL(q2_du1dk2, i1, i2, i3)
#define Q2_DK2DK2(i1, i2, i3)         F_IDX3_DBL(q2_dk2dk2, i1, i2, i3)

#define P2_DQ1DQ1(i1, i2, i3)         F_IDX3_DBL(p2_dq1dq1, i1, i2, i3)
#define P2_DQ1DP1(i1, i2, i3)         F_IDX3_DBL(p2_dq1dp1, i1, i2, i3)
#define P2_DQ1DU1(i1, i2, i3)         F_IDX3_DBL(p2_dq1du1, i1, i2, i3)
#define P2_DQ1DK2(i1, i2, i3)         F_IDX3_DBL(p2_dq1dk2, i1, i2, i3)
#define P2_DP1DP1(i1, i2, i3)         F_IDX3_DBL(p2_dp1dp1, i1, i2, i3)
#define P2_DP1DU1(i1, i2, i3)         F_IDX3_DBL(p2_dp1du1, i1, i2, i3)
#define P2_DP1DK2(i1, i2, i3)         F_IDX3_DBL(p2_dp1dk2, i1, i2, i3)
#define P2_DU1DU1(i1, i2, i3)         F_IDX3_DBL(p2_du1du1, i1, i2, i3)
#define P2_DU1DK2(i1, i2, i3)         F_IDX3_DBL(p2_du1dk2, i1, i2, i3)
#define P2_DK2DK2(i1, i2, i3)         F_IDX3_DBL(p2_dk2dk2, i1, i2, i3)

#define L1_DQ1DQ1(i1, i2, i3)         F_IDX3_DBL(l1_dq1dq1, i1, i2, i3)
#define L1_DQ1DP1(i1, i2, i3)         F_IDX3_DBL(l1_dq1dp1, i1, i2, i3)
#define L1_DQ1DU1(i1, i2, i3)         F_IDX3_DBL(l1_dq1du1, i1, i2, i3)
#define L1_DQ1DK2(i1, i2, i3)         F_IDX3_DBL(l1_dq1dk2, i1, i2, i3)
#define L1_DP1DP1(i1, i2, i3)         F_IDX3_DBL(l1_dp1dp1, i1, i2, i3)
#define L1_DP1DU1(i1, i2, i3)         F_IDX3_DBL(l1_dp1du1, i1, i2, i3)
#define L1_DP1DK2(i1, i2, i3)         F_IDX3_DBL(l1_dp1dk2, i1, i2, i3)
#define L1_DU1DU1(i1, i2, i3)         F_IDX3_DBL(l1_du1du1, i1, i2, i3)
#define L1_DU1DK2(i1, i2, i3)         F_IDX3_DBL(l1_du1dk2, i1, i2, i3)
#define L1_DK2DK2(i1, i2, i3)         F_IDX3_DBL(l1_dk2dk2, i1, i2, i3)

#define DDH1T(i1, i2, i3)             F_IDX3_DBL(DDh1T, i1, i2, i3)
#define DDH2(i1, i2, i3)              F_IDX3_DBL(DDh2, i1, i2, i3)
#define DDDH1T(i1, i2, i3, i4)        F_IDX4_DBL(DDDh1T, i1, i2, i3, i4)

#define D1D1L2_D1FM2(i1, i2)          F_IDX2_DBL(D1D1L2_D1fm2, i1, i2)
#define D2D1L2_D2FM2(i1, i2)          F_IDX2_DBL(D2D1L2_D2fm2, i1, i2)
#define D1D2L2(i1, i2)                F_IDX2_DBL(D1D2L2, i1, i2)
#define D2D2L2(i1, i2)                F_IDX2_DBL(D2D2L2, i1, i2)
#define D3FM2(i1, i2)                 F_IDX2_DBL(D3fm2, i1, i2)

#define D1D1D1L2_D1D1FM2(i1, i2, i3)  F_IDX3_DBL(D1D1D1L2_D1D1fm2, i1, i2, i3)
#define D1D2D1L2_D1D2FM2(i1, i2, i3)  F_IDX3_DBL(D1D2D1L2_D1D2fm2, i1, i2, i3)
#define _D2D2D1L2_D2D2FM2(i1, i2, i3) F_IDX3_DBL(_D2D2D1L2_D2D2fm2, i1, i2, i3)

#define D1D1D2L2(i1, i2, i3)          F_IDX3_DBL(D1D1D2L2, i1, i2, i3)
#define D1D2D2L2(i1, i2, i3)          F_IDX3_DBL(D1D2D2L2, i1, i2, i3)
#define _D2D2D2L2(i1, i2, i3)         F_IDX3_DBL(_D2D2D2L2, i1, i2, i3)

#define D1D3FM2(i1, i2, i3)           F_IDX3_DBL(D1D3fm2, i1, i2, i3)
#define D2D3FM2(i1, i2, i3)           F_IDX3_DBL(D2D3fm2, i1, i2, i3)
#define D3D3FM2(i1, i2, i3)           F_IDX3_DBL(D3D3fm2, i1, i2, i3)

#define DQ2_DQ1_OP(i1, i2, i3)        F_IDX3_DBL(dq2_dq1_op, i1, i2, i3)
#define DL1_DQ1_OP(i1, i2, i3)        F_IDX3_DBL(dl1_dq1_op, i1, i2, i3)
#define DP2_DQ1_OP(i1, i2, i3)        F_IDX3_DBL(dp2_dq1_op, i1, i2, i3)
#define DQ2_DP1_OP(i1, i2, i3)        F_IDX3_DBL(dq2_dp1_op, i1, i2, i3)
#define DL1_DP1_OP(i1, i2, i3)        F_IDX3_DBL(dl1_dp1_op, i1, i2, i3)
#define DP2_DP1_OP(i1, i2, i3)        F_IDX3_DBL(dp2_dp1_op, i1, i2, i3)
#define DQ2_DU1_OP(i1, i2, i3)        F_IDX3_DBL(dq2_du1_op, i1, i2, i3)
#define DL1_DU1_OP(i1, i2, i3)        F_IDX3_DBL(dl1_du1_op, i1, i2, i3)
#define DP2_DU1_OP(i1, i2, i3)        F_IDX3_DBL(dp2_du1_op, i1, i2, i3)
#define DQ2_DK2_OP(i1, i2, i3)        F_IDX3_DBL(dq2_dk2_op, i1, i2, i3)
#define DL1_DK2_OP(i1, i2, i3)        F_IDX3_DBL(dl1_dk2_op, i1, i2, i3)
#define DP2_DK2_OP(i1, i2, i3)        F_IDX3_DBL(dp2_dk2_op, i1, i2, i3)


/* Set the integrator's system state to q[k] (using q[k-1] to
 * approximate velocities):
 *
 *    q = q[k]
 *    u = u[k]
 *   dq = (q[k] - q[k-1]) / (t[k] - t[k-1])
 *
 *  where k is 1 or 2.
 */
static void MidpointVI_set_state(MidpointVI *mvi, int k)
{
    int i;
    double *state;
    Config *q;
    Input *u;
    assert(k == 1 || k == 2);
    DECLARE_F_IDX1(mvi->q1, __q1);
    DECLARE_F_IDX1(mvi->q2, __q2);
    DECLARE_F_IDX1(mvi->u1, __u1);                      
    
    mvi->system->cache = SYSTEM_CACHE_NONE;
    if(k == 1) {
        mvi->system->time = mvi->t1;
        state = &Q1(0);
    } else {
        mvi->system->time = mvi->t2;
        state = &Q2(0);
    }        
    for(i = 0; i < System_CONFIGS(mvi->system); i++) {
	q = System_CONFIG(mvi->system, i);
	q->q = state[i];
	q->dq = (Q2(i) - Q1(i)) / (mvi->t2 - mvi->t1);
    }
    for(i = 0; i < System_INPUTS(mvi->system); i++) {
	u = System_INPUT(mvi->system, i);
	u->u = U1(i);
    }
}

/* Set the integrator's system state to the midpoint between q1 and
 * q2:
 *
 *    q = (q2 + q1)/2
 *   dq = (q2 - q1) / (t2 - t1)
 */
static void MidpointVI_set_midpoint(MidpointVI *mvi)
{
    int i;
    Config *q;
    Input *u;
    DECLARE_F_IDX1(mvi->q1, __q1);
    DECLARE_F_IDX1(mvi->q2, __q2);
    DECLARE_F_IDX1(mvi->u1, __u1);                      
    
    mvi->system->cache = SYSTEM_CACHE_NONE;
    mvi->system->time = 0.5* (mvi->t2 + mvi->t1);
    for(i = 0; i < System_CONFIGS(mvi->system); i++) {
	q = System_CONFIG(mvi->system, i);
	q->q = 0.5*(Q2(i) + Q1(i));
	q->dq = (Q2(i) - Q1(i)) / (mvi->t2 - mvi->t1);
    }
    for(i = 0; i < System_INPUTS(mvi->system); i++) {
	u = System_INPUT(mvi->system, i);
	u->u = U1(i);
    }
}


static double D1L2(MidpointVI *mvi, Config *q1_1)
{
    System *sys = mvi->system;
    double dt = mvi->t2 - mvi->t1;

    return 0.5 * dt * System_L_dq(sys, q1_1)
	- System_L_ddq(sys, q1_1);
}

static double D2L2(MidpointVI *mvi, Config *q2_1)
{
    System *sys = mvi->system;
    double dt = mvi->t2 - mvi->t1;

    return 0.5 * dt * System_L_dq(sys, q2_1)
	+ System_L_ddq(sys, q2_1);
}

static double fm2(MidpointVI *mvi, Config *q)
{
    double dt = mvi->t2 - mvi->t1;
    return dt * System_F(mvi->system, q);
}
	
static double D2fm2(MidpointVI *mvi, Config *q, Config *q2_1)
{
    double dt = mvi->t2 - mvi->t1;
    return (0.5 * dt * System_F_dq(mvi->system, q, q2_1)
	    + System_F_ddq(mvi->system, q, q2_1));
}

static int MidpointVI_calc_p2(MidpointVI *mvi)
{
    int i;
    Config *q;
    DECLARE_F_IDX1(mvi->p2, __p2);

    for(i = 0; i < System_DYN_CONFIGS(mvi->system); i++) {
	q = System_DYN_CONFIG(mvi->system, i);
	P2(i) = D2L2(mvi, q);
    }
    if(PyErr_Occurred())
	return -1;    
    return 0;
}

static int calc_Dh(MidpointVI *mvi, PyArrayObject *dest)
{
    int i1, i2;
    Constraint *constraint;
    Config *q2;

    int nq = System_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);

    assert(dest != NULL);
    assert(PyArray_NDIM(dest) == 2);
    assert(PyArray_DIMS(dest)[0] == nc);
    assert(PyArray_DIMS(dest)[1] == nq);

    for(i1 = 0; i1 < nc; i1++) {
	constraint = System_CONSTRAINT(mvi->system, i1);
	for(i2 = 0; i2 < nq; i2++) {
	    q2 = System_CONFIG(mvi->system, i2);
	    IDX2_DBL(dest, i1, i2) = constraint->h_dq(constraint, q2);
	}
    }
    
    if(PyErr_Occurred())
	return -1;
    return 0;
}

static int calc_f(MidpointVI *mvi)
{
    int i1, i2;
    Config *q1;
    Constraint *constraint;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    DECLARE_F_IDX2(mvi->Dh1T, Dh1T);
    DECLARE_F_IDX1(mvi->lambda1, __lambda1);
    DECLARE_F_IDX1(mvi->p1, __p1);
    DECLARE_F_IDX1(mvi->f, __f);                        

    MidpointVI_set_midpoint(mvi);
    for(i1 = 0; i1 < nd; i1++) {
	q1 = System_DYN_CONFIG(mvi->system, i1);
	F(i1) = P1(i1) + D1L2(mvi, q1) + fm2(mvi, q1);
	for(i2 = 0; i2 < nc; i2++) 
	    F(i1) -= DH1T(i1, i2)*LAMBDA1(i2);
    }	    
    if(PyErr_Occurred())
	return -1;

    MidpointVI_set_state(mvi, 2);

    for(i1 = 0; i1 < nc; i1++) {
	constraint = System_CONSTRAINT(mvi->system, i1);
	F(nd + i1) = constraint->h(constraint);
    }
    if(PyErr_Occurred())
	return -1;

    return 0;
}

static int MidpointVI_calc_f(MidpointVI *mvi)
{    
    MidpointVI_set_state(mvi, 1);
    if(calc_Dh(mvi, mvi->Dh2)) return -1;
    transpose_np_matrix(mvi->Dh1T, mvi->Dh2);
    if(calc_f(mvi))
	return -1;
    return 0;
}

static int calc_bar_Df_11(MidpointVI *mvi)
{
    int i, k;
    Config *qi, *qk;
    double val;
    double dt = mvi->t2 - mvi->t1;
    int nd = System_DYN_CONFIGS(mvi->system);
    DECLARE_F_IDX2(mvi->Df, Df);

    // Df_11[k][i] is D2iD1kL2 + D2if2(k)    

    // To help with profiling, otherwise not needed otherwise but
    // doesn't affect speed to leave in anyways.
    build_g_dqdq_cache(mvi->system);
    build_vb_dqdq_cache(mvi->system);
    build_vb_ddqdq_cache(mvi->system);

    // Initialize the table with the force component.
    for(k = 0; k < nd; k++) {
	qk = System_DYN_CONFIG(mvi->system, k);
	for(i = 0; i < nd; i++) {
	    qi = System_DYN_CONFIG(mvi->system, i);
	    DF(k, i) = D2fm2(mvi, qk, qi);
	}
    }

    for(k = 0; k < nd; k++) {
	qk = System_DYN_CONFIG(mvi->system, k);

	DF(k, k) += 0.25*dt*System_L_dqdq(mvi->system, qk, qk);
	DF(k, k) -= 1.0/dt*System_L_ddqddq(mvi->system, qk, qk);

	for(i = 0; i < nd; i++) {
	    qi = System_DYN_CONFIG(mvi->system, i);
	    
	    val = 0.5*System_L_ddqdq(mvi->system, qi, qk);
	    DF(k, i) += val;
	    DF(i, k) -= val;
	}
	for(i = 0; i < k; i++) {
	    qi = System_DYN_CONFIG(mvi->system, i);

	    val = 0.25*dt*System_L_dqdq(mvi->system, qk, qi);
	    DF(k, i) += val;
	    DF(i, k) += val;

	    val = 1.0/dt*System_L_ddqddq(mvi->system, qk, qi);
	    DF(k, i) -= val;
	    DF(i, k) -= val;
	}
	
    }
    if(PyErr_Occurred())
	return -1;
    return 0;
}

static void calc_bar_Df_12(MidpointVI *mvi)
{
    int i1, i2;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    DECLARE_F_IDX2(mvi->Dh1T, Dh1T);
    DECLARE_F_IDX2(mvi->Df, Df);

    for(i1 = 0; i1 < nd; i1++)
	for(i2 = 0; i2 < nc; i2++)
	    DF(i1, nd+i2) = -DH1T(i1, i2);
}

static void calc_bar_Df_21(MidpointVI *mvi)
{
    int i1, i2;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    DECLARE_F_IDX2(mvi->Dh2, Dh2);
    DECLARE_F_IDX2(mvi->Df, Df);

    for(i1 = 0; i1 < nc; i1++)
	for(i2 = 0; i2 < nd; i2++)
	    DF(nd+i1, i2) = DH2(i1, i2);
}

static void calc_bar_Df_22(MidpointVI *mvi)
{
    int i1, i2;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    DECLARE_F_IDX2(mvi->Df, Df);

    for(i1 = 0; i1 < nc; i1++)
	for(i2 = 0; i2 < nc; i2++)
	    DF(nd+i1, nd+i2) = 0.0;
}

static int DEL_solved(MidpointVI *mvi)
{
    int i;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    DECLARE_F_IDX1(mvi->f, __f);                        

    /* The equations are considered 'solved' if: 
     * || DEL || < tolerance                       
     * |g_i(q)| < constraint[i]->tolerance for all i
     */
    if(norm_vector(&F(0), nd) > mvi->tolerance)
	return 0;
    for(i = 0; i < nc; i++) 
	if(fabs(F(nd+i)) > System_CONSTRAINT(mvi->system, i)->tolerance)
	    return 0;
    return 1;  
}

static int MidpointVI_solve_DEL(MidpointVI *mvi, int max_iterations)
{
    int iterations;
    int row;
    int k;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    int nf = nd + nc;
    DECLARE_F_IDX1(mvi->q2, __q2);
    DECLARE_F_IDX1(mvi->lambda1, __lambda1);
    DECLARE_F_IDX1(mvi->f, __f);                        

    iterations = 0;

    MidpointVI_set_state(mvi, 1);
    if(calc_Dh(mvi, mvi->Dh2)) return -1;
    transpose_np_matrix(mvi->Dh1T, mvi->Dh2);
    
    while(1) {
	if(calc_f(mvi)) return -1;

	if(DEL_solved(mvi))
	    break;

	if(iterations > max_iterations) {
	    PyErr_Format(PyExc_StandardError, "failed to converge after %d iterations", iterations);
	    return -1;
	}	

	// Not solved, need to calculate Df^-1 and improve guess
	MidpointVI_set_midpoint(mvi);
	if(calc_bar_Df_11(mvi)) return -1;
	MidpointVI_set_state(mvi, 2);
	calc_Dh(mvi, mvi->Dh2);
	calc_bar_Df_12(mvi);
	calc_bar_Df_21(mvi);
	calc_bar_Df_22(mvi);

	if(LU_decomp(mvi->Df, nf, mvi->Df_index, LU_tolerance))
	    return -1;
	LU_solve_vec(mvi->Df, nf, mvi->Df_index, &F(0));

	row = 0;
	for(k = 0; k < nd; k++, row++)
	    Q2(k) -= F(row);
	for(k = 0; k < nc; k++, row++)
	    LAMBDA1(k) -= F(row);	

	iterations++;
    }

    MidpointVI_set_midpoint(mvi);
    if(MidpointVI_calc_p2(mvi)) return -1;
    
    mvi->cache = MIDPOINTVI_CACHE_SOLUTION;
    return iterations;
}

static int calc_deriv1_cache(MidpointVI *mvi)
{
    int i1, i2;
    Config *q1, *q2;
    Input *u1;
    double val1, val2;
    double dt = mvi->t2 - mvi->t1;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nk = System_KIN_CONFIGS(mvi->system);
    int nu = System_INPUTS(mvi->system);
    DECLARE_F_IDX2(mvi->D1D1L2_D1fm2, D1D1L2_D1fm2);
    DECLARE_F_IDX2(mvi->D2D1L2_D2fm2, D2D1L2_D2fm2);
    DECLARE_F_IDX2(mvi->D1D2L2, D1D2L2);
    DECLARE_F_IDX2(mvi->D2D2L2, D2D2L2);
    DECLARE_F_IDX2(mvi->D3fm2, D3fm2);

    // To help with profiling, otherwise not needed otherwise but
    // doesn't affect speed to leave in anyways.
    build_g_dqdq_cache(mvi->system);
    build_vb_dqdq_cache(mvi->system);
    build_vb_ddqdq_cache(mvi->system);    

    // Initialize the tables
    for(i1 = 0; i1 < nd+nk; i1++) {
	q1 = System_CONFIG(mvi->system, i1);
	for(i2 = 0; i2 < nd; i2++) {
	    q2 = System_CONFIG(mvi->system, i2);
	    val1 = 0.5 * dt * System_F_dq(mvi->system, q2, q1);
	    val2 = System_F_ddq(mvi->system, q2, q1);
	    D1D1L2_D1FM2(i1, i2) = val1 - val2;
	    D2D1L2_D2FM2(i1, i2) = val1 + val2;
	    D1D2L2(i1, i2) = 0;
	    D2D2L2(i1, i2) = 0;
	}
    }

    for(i1 = 0; i1 < nd; i1++) {
	q1 = System_CONFIG(mvi->system, i1);

	val1 = 0.25*dt*System_L_dqdq(mvi->system, q1, q1);
	val2 = 1.0/dt*System_L_ddqddq(mvi->system, q1, q1);
	D1D1L2_D1FM2(i1, i1) += val1 + val2;
	D2D1L2_D2FM2(i1, i1) += val1 - val2;
	D1D2L2(i1, i1) += val1 - val2;
	D2D2L2(i1, i1) += val1 + val2;
	
	for(i2 = 0; i2 < nd+nk; i2++) {
	    q2 = System_CONFIG(mvi->system, i2);
	    val1 = 0.5*System_L_ddqdq(mvi->system, q1, q2);
	    D1D1L2_D1FM2(i2, i1) -= val1;
	    D2D1L2_D2FM2(i2, i1) -= val1;
	    D1D2L2(i2, i1) += val1;
	    D2D2L2(i2, i1) += val1;
	    if(i2 < nd) {
		D1D1L2_D1FM2(i1, i2) -= val1;
		D2D1L2_D2FM2(i1, i2) += val1;
		D1D2L2(i1, i2) -= val1;
		D2D2L2(i1, i2) += val1;
	    }
	}

	for(i2 = 0; i2 < i1; i2++) {
	    q2 = System_CONFIG(mvi->system, i2);
	    val1 = 0.25*dt*System_L_dqdq(mvi->system, q1, q2);
	    val2 = 1.0/dt*System_L_ddqddq(mvi->system, q1, q2);
	    D1D1L2_D1FM2(i2, i1) += val1 + val2;
	    D2D1L2_D2FM2(i2, i1) += val1 - val2;
	    D1D2L2(i2, i1) += val1 - val2;
	    D2D2L2(i2, i1) += val1 + val2;
	    if(i2 < nd) {
		D1D1L2_D1FM2(i1, i2) += val1 + val2;
		D2D1L2_D2FM2(i1, i2) += val1 - val2;
		D1D2L2(i1, i2) += val1 - val2;
		D2D2L2(i1, i2) += val1 + val2;
	    }
	}
    }
	
    for(i1 = nd; i1 < nd+nk; i1++) {
	q1 = System_CONFIG(mvi->system, i1);

	for(i2 = 0; i2 < nd; i2++) {
	    q2 = System_CONFIG(mvi->system, i2);
	    val1 = 0.5*System_L_ddqdq(mvi->system, q1, q2);
	    D1D1L2_D1FM2(i1, i2) -= val1;
	    D2D1L2_D2FM2(i1, i2) += val1;
	    D1D2L2(i1, i2) -= val1;
	    D2D2L2(i1, i2) += val1;
	}

	for(i2 = 0; i2 < i1 && i2 < nd; i2++) {
	    q2 = System_CONFIG(mvi->system, i2);
	    val1 = 0.25*dt*System_L_dqdq(mvi->system, q1, q2);
	    val2 = 1.0/dt*System_L_ddqddq(mvi->system, q1, q2);
	    D1D1L2_D1FM2(i1, i2) += val1 + val2;
	    D2D1L2_D2FM2(i1, i2) += val1 - val2;
	    D1D2L2(i1, i2) += val1 - val2;
	    D2D2L2(i1, i2) += val1 + val2;
	}
    }

    for(i1 = 0; i1 < nu; i1++) {
	u1 = System_INPUT(mvi->system, i1);
	for(i2 = 0; i2 < nd; i2++) {
	    q2 = System_CONFIG(mvi->system, i2);
	    D3FM2(i1, i2) = dt * System_F_du(mvi->system, q2, u1);
	}
    }

    if(PyErr_Occurred())
	return -1;
    return 0;
}


static int calc_h1_deriv1(MidpointVI *mvi)
{
    int i1, i2, i3;
    Config *q1;
    Config *q2;
    Constraint *constraint;
    
    int nd = System_DYN_CONFIGS(mvi->system);
    int nk = System_KIN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    DECLARE_F_IDX3(mvi->DDh1T, DDh1T);
    
    for(i1 = 0; i1 < nd+nk; i1++) {
	q1 = System_CONFIG(mvi->system, i1);
	for(i2 = 0; i2 < nd+nk; i2++) {
	    q2 = System_CONFIG(mvi->system, i2);
	    for(i3 = 0; i3 < nc; i3++) {
		constraint = System_CONSTRAINT(mvi->system, i3);
		DDH1T(i1, i2, i3) = constraint->h_dqdq(constraint, q1, q2);
	    }
	}
    }
    if(PyErr_Occurred())
	return -1;
    return 0;    
}

static int calc_M2(MidpointVI *mvi)
{
    int i1, i2;
    int nd = System_DYN_CONFIGS(mvi->system);
    DECLARE_F_IDX2(mvi->D2D1L2_D2fm2, D2D1L2_D2fm2);
    DECLARE_F_IDX2(mvi->M2_lu, M2_lu);

    for(i1 = 0; i1 < nd; i1++) {
	for(i2 = 0; i2 < nd; i2++) {
	    M2_LU(i1, i2) = D2D1L2_D2FM2(i2, i1);
	}
    }
    if(PyErr_Occurred())
	return -1;
    if(LU_decomp(mvi->M2_lu, nd, mvi->M2_lu_index, LU_tolerance))
	return -1;
    return 0;
}

static int calc_proj_inv(MidpointVI *mvi)
{
    int i1, i2;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    DECLARE_F_IDX2(mvi->proj_lu, proj_lu);

    copy_np_matrix(mvi->temp_ndnc, mvi->Dh1T, nd, nc);
    LU_solve_mat(mvi->M2_lu, nd, mvi->M2_lu_index, mvi->temp_ndnc, nc);
    mul_matmat_np_np_np(mvi->proj_lu, nc, nc, mvi->Dh2, mvi->temp_ndnc, nd);
    for(i1 = 0; i1 < nc; i1++) { 
	for(i2 = 0; i2 < nc; i2++)
	    PROJ_LU(i1, i2) = -PROJ_LU(i1, i2);
    }
    if(LU_decomp(mvi->proj_lu, nc, mvi->proj_lu_index, LU_tolerance))
	return -1;
    return 0;    
}

static int calc_deriv1(MidpointVI *mvi)
{
    int i, i1, i2;

    int nd = System_DYN_CONFIGS(mvi->system);
    int nk = System_KIN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    int nu = System_INPUTS(mvi->system);
    double *temp_nd = (double*)malloc(sizeof(double)*nd);
    DECLARE_F_IDX2(mvi->q2_dq1, q2_dq1);
    DECLARE_F_IDX2(mvi->q2_dp1, q2_dp1);
    DECLARE_F_IDX2(mvi->q2_du1, q2_du1);
    DECLARE_F_IDX2(mvi->q2_dk2, q2_dk2);
    DECLARE_F_IDX2(mvi->p2_dq1, p2_dq1);
    DECLARE_F_IDX2(mvi->p2_dp1, p2_dp1);
    DECLARE_F_IDX2(mvi->p2_du1, p2_du1);
    DECLARE_F_IDX2(mvi->p2_dk2, p2_dk2);
    DECLARE_F_IDX2(mvi->l1_dq1, l1_dq1);
    DECLARE_F_IDX2(mvi->l1_dp1, l1_dp1);
    DECLARE_F_IDX2(mvi->l1_du1, l1_du1);
    DECLARE_F_IDX2(mvi->l1_dk2, l1_dk2);
    DECLARE_F_IDX3(mvi->DDh1T, DDh1T);
    DECLARE_F_IDX2(mvi->D1D1L2_D1fm2, D1D1L2_D1fm2);
    DECLARE_F_IDX2(mvi->D2D1L2_D2fm2, D2D1L2_D2fm2);
    DECLARE_F_IDX2(mvi->D1D2L2, D1D2L2);
    DECLARE_F_IDX2(mvi->D2D2L2, D2D2L2);
    DECLARE_F_IDX2(mvi->D3fm2, D3fm2);
    DECLARE_F_IDX2(mvi->Dh1T, Dh1T);
    DECLARE_F_IDX2(mvi->Dh2, Dh2);
    DECLARE_F_IDX1(mvi->lambda1, __lambda1);
    
    // Calculate d/dq1
    for(i = 0; i < nd+nk; i++) {
	for(i1 = 0; i1 < nd; i1++) {
            
	    temp_nd[i1] = - D1D1L2_D1FM2(i, i1);
	    for(i2 = 0; i2 < nc; i2++) 
		temp_nd[i1] += DDH1T(i, i1, i2)*LAMBDA1(i2);
	}
	if(PyErr_Occurred())  // fmk could raise an exception
            goto fail;
	// Save c_dq1 to q2_dq1
	copy_vector(&Q2_DQ1(i, 0), temp_nd, nd);	
        if(nc) {
            // Solve for lambda_dq1
            // NOTE: could we write c_dq1 directly to q2_dq1, and only
            // copy it over to temp if we need to?
            LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, temp_nd);
            mul_matvec_c_np_c(&L1_DQ1(i, 0), nc, mvi->Dh2, temp_nd, nd);
            LU_solve_vec(mvi->proj_lu, nc, mvi->proj_lu_index, &L1_DQ1(i, 0));
            // Solve for q2_dq1
            for(i1 = 0; i1 < nd; i1++) {
                for(i2 = 0; i2 < nc; i2++)
                    Q2_DQ1(i, i1) += DH1T(i1, i2)*L1_DQ1(i, i2);
            }
        }
	LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, &Q2_DQ1(i, 0));

	// Calculate p2/dq1
	for(i1 = 0; i1 < nd; i1++) {
	    P2_DQ1(i, i1) = D1D2L2(i, i1);
	    for(i2 = 0; i2 < nd; i2++) {
		P2_DQ1(i, i1) += D2D2L2(i2, i1)*Q2_DQ1(i, i2);
	    }
	}
    }

    // Calculate d/dp1
    for(i = 0; i < nd; i++) {
	for(i1 = 0; i1 < nd; i1++) {
	    temp_nd[i1] = (i1 == i ? -1.0 : 0.0);
	}
	if(PyErr_Occurred())  // fmk could raise an exception
            goto fail;
	// Save c_dp1 to q2_dp1
	copy_vector(&Q2_DP1(i, 0), temp_nd, nd);
        if(nc) {
            // Solve for lambda_dp1
            LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, temp_nd);
            mul_matvec_c_np_c(&L1_DP1(i, 0), nc, mvi->Dh2, temp_nd, nd);
            LU_solve_vec(mvi->proj_lu, nc, mvi->proj_lu_index, &L1_DP1(i, 0));
            // Solve for q2_dp1
            for(i1 = 0; i1 < nd; i1++) {
                for(i2 = 0; i2 < nc; i2++)
                    Q2_DP1(i, i1) += DH1T(i1, i2)*L1_DP1(i, i2);
            }
        }
	LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, &Q2_DP1(i, 0));

	// Calculate p2/dp1
	for(i1 = 0; i1 < nd; i1++) {
	    P2_DP1(i, i1) = 0.0;
	    for(i2 = 0; i2 < nd; i2++) {
		P2_DP1(i, i1) += D2D2L2(i2, i1)*Q2_DP1(i, i2);
	    }
	}
    }

    // Calculate d/du1
    for(i = 0; i < nu; i++) {
	for(i1 = 0; i1 < nd; i1++) {
	    temp_nd[i1] = -D3FM2(i, i1);
	}
	if(PyErr_Occurred())  // fmk could raise an exception
            goto fail;
	// Save c_du1 to q2_du1
	copy_vector(&Q2_DU1(i, 0), temp_nd, nd);	
        if(nc) {
            // Solve for lambda_du1
            LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, temp_nd);
            mul_matvec_c_np_c(&L1_DU1(i, 0), nc, mvi->Dh2, temp_nd, nd);
            LU_solve_vec(mvi->proj_lu, nc, mvi->proj_lu_index, &L1_DU1(i, 0));
            // Solve for q2_du1
            for(i1 = 0; i1 < nd; i1++) {
	    for(i2 = 0; i2 < nc; i2++)
		Q2_DU1(i, i1) += DH1T(i1, i2)*L1_DU1(i, i2);
            }
        }
	LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, &Q2_DU1(i, 0));	

	// Calculate p2/du1
	for(i1 = 0; i1 < nd; i1++) {
	    P2_DU1(i, i1) = 0;
	    for(i2 = 0; i2 < nd; i2++) {
		P2_DU1(i, i1) += D2D2L2(i2, i1)*Q2_DU1(i, i2);
	    }
	}
    }

    // Calculate d/dk2
    for(i = 0; i < nk; i++) {
	for(i1 = 0; i1 < nd; i1++) {
	    temp_nd[i1] = - D2D1L2_D2FM2(nd+i, i1);
	}
	if(PyErr_Occurred())  // fmk could raise an exception
            goto fail;
	// Save c_dk2 to q2_dk2
	copy_vector(&Q2_DK2(i, 0), temp_nd, nd);
        if(nc) {
            // Solve for lambda_dk2
            LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, temp_nd);
            mul_matvec_c_np_c(&L1_DK2(i, 0), nc, mvi->Dh2, temp_nd, nd);
            for(i1 = 0; i1 < nc; i1++) 
                L1_DK2(i, i1) += DH2(i1, nd+i);
            LU_solve_vec(mvi->proj_lu, nc, mvi->proj_lu_index, &L1_DK2(i, 0));
            // Solve for q2_dk2
            for(i1 = 0; i1 < nd; i1++) {
                for(i2 = 0; i2 < nc; i2++)
                    Q2_DK2(i, i1) += DH1T(i1, i2)*L1_DK2(i, i2);
            }
        }
        LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, &Q2_DK2(i, 0));
            
	// Calculate p2/dk1
	for(i1 = 0; i1 < nd; i1++) {
	    P2_DK2(i, i1) = D2D2L2(nd+i, i1);
	    for(i2 = 0; i2 < nd; i2++) {
		P2_DK2(i, i1) += D2D2L2(i2, i1)*Q2_DK2(i, i2);
	    }
	}
	
    }

    free(temp_nd);
    return 0;
    
fail:
    free(temp_nd);
    return -1;
}

static int MidpointVI_calc_deriv1(MidpointVI *mvi)
{
    if(mvi->cache & MIDPOINTVI_CACHE_SOLUTION_DERIV1)
	return 0;
    if(!(mvi->cache & MIDPOINTVI_CACHE_SOLUTION)) {
	PyErr_Format(PyExc_StandardError, "Integrator has not solved of the next time step yet.");
	return -1;
    }
    MidpointVI_set_state(mvi, 1);
    if(calc_h1_deriv1(mvi)) return -1;
    MidpointVI_set_state(mvi, 2);
    if(calc_Dh(mvi, mvi->Dh2)) return -1;
    MidpointVI_set_midpoint(mvi);    
    if(calc_deriv1_cache(mvi)) return -1;
    if(calc_M2(mvi)) return -1;
    if(calc_proj_inv(mvi)) return -1;
    if(calc_deriv1(mvi)) return -1;

    mvi->cache |= MIDPOINTVI_CACHE_SOLUTION_DERIV1;
    return 0;
}

static void calc_deriv2_cache_dqdqdq(MidpointVI *mvi)
{
    int i1, i2, i3;
    Config *q1, *q2, *q3;
    double dt = mvi->t2 - mvi->t1;
    double val1;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nk = System_KIN_CONFIGS(mvi->system);
    DECLARE_F_IDX3(mvi->D1D1D1L2_D1D1fm2, D1D1D1L2_D1D1fm2);
    DECLARE_F_IDX3(mvi->D1D2D1L2_D1D2fm2, D1D2D1L2_D1D2fm2);          
    DECLARE_F_IDX3(mvi->_D2D2D1L2_D2D2fm2, _D2D2D1L2_D2D2fm2);
    DECLARE_F_IDX3(mvi->D1D1D2L2, D1D1D2L2);
    DECLARE_F_IDX3(mvi->D1D2D2L2, D1D2D2L2);
    DECLARE_F_IDX3(mvi->_D2D2D2L2, _D2D2D2L2);

    for(i1 = 0; i1 < nd+nk; i1++) { 
	q1 = System_CONFIG(mvi->system, i1);
	
    	val1 = 0.125*dt*System_L_dqdqdq(mvi->system, q1, q1, q1);
	if(i1 < nd) {
	    D1D1D1L2_D1D1FM2(i1, i1, i1) = val1;
	    D1D2D1L2_D1D2FM2(i1, i1, i1) = val1;
	    _D2D2D1L2_D2D2FM2(i1, i1, i1) = val1;
	    D1D1D2L2(i1, i1, i1) = val1;
	    D1D2D2L2(i1, i1, i1) = val1;
	    _D2D2D2L2(i1, i1, i1) = val1;
	}

    	for(i2 = 0; i2 < i1; i2++) {
    	    q2 = System_CONFIG(mvi->system, i2);

    	    val1 = 0.125*dt*System_L_dqdqdq(mvi->system, q1, q1, q2);
	    if(i1 < nd) {
		D1D1D1L2_D1D1FM2(i1, i2, i1) = val1;
		D1D1D1L2_D1D1FM2(i2, i1, i1) = val1;
		D1D2D1L2_D1D2FM2(i1, i2, i1) = val1;
		D1D2D1L2_D1D2FM2(i2, i1, i1) = val1;
		_D2D2D1L2_D2D2FM2(i1, i1, i2) = val1;
		_D2D2D1L2_D2D2FM2(i1, i2, i1) = val1;
		D1D1D2L2(i1, i2, i1) = val1;
		D1D1D2L2(i2, i1, i1) = val1;
		D1D2D2L2(i1, i2, i1) = val1;
		D1D2D2L2(i2, i1, i1) = val1;
		_D2D2D2L2(i1, i1, i2) = val1;
		_D2D2D2L2(i1, i2, i1) = val1;
	    }
	    if(i2 < nd) {
		D1D1D1L2_D1D1FM2(i1, i1, i2) = val1;
		D1D2D1L2_D1D2FM2(i1, i1, i2) = val1;
		_D2D2D1L2_D2D2FM2(i2, i1, i1) = val1;
		D1D1D2L2(i1, i1, i2) = val1;
		D1D2D2L2(i1, i1, i2) = val1;
		_D2D2D2L2(i2, i1, i1) = val1;
	    }

	    val1 = 0.125*dt*System_L_dqdqdq(mvi->system, q1, q2, q2);
	    if(i1 < nd) {
		D1D1D1L2_D1D1FM2(i2, i2, i1) = val1;
		D1D2D1L2_D1D2FM2(i2, i2, i1) = val1;
		_D2D2D1L2_D2D2FM2(i1, i2, i2) = val1;
		D1D1D2L2(i2, i2, i1) = val1;
		D1D2D2L2(i2, i2, i1) = val1;
		_D2D2D2L2(i1, i2, i2) = val1;
	    }
	    if(i2 < nd) {
		D1D1D1L2_D1D1FM2(i1, i2, i2) = val1;
		D1D1D1L2_D1D1FM2(i2, i1, i2) = val1;
		D1D2D1L2_D1D2FM2(i1, i2, i2) = val1;
		D1D2D1L2_D1D2FM2(i2, i1, i2) = val1;
		_D2D2D1L2_D2D2FM2(i2, i1, i2) = val1;
		_D2D2D1L2_D2D2FM2(i2, i2, i1) = val1;
		D1D1D2L2(i1, i2, i2) = val1;
		D1D1D2L2(i2, i1, i2) = val1;
		D1D2D2L2(i1, i2, i2) = val1;
		D1D2D2L2(i2, i1, i2) = val1;
		_D2D2D2L2(i2, i1, i2) = val1;
		_D2D2D2L2(i2, i2, i1) = val1;
	    }

	    for(i3 = 0; i3 < i2; i3++) {
		q3 = System_CONFIG(mvi->system, i3);

		if(i1 >= nd && i2 >= nd && i3 >= nd)
		    continue;
		
		val1 = 0.125*dt*System_L_dqdqdq(mvi->system, q1, q2, q3);
		if(i1 < nd) {
		    D1D1D1L2_D1D1FM2(i2, i3, i1) = val1;
		    D1D1D1L2_D1D1FM2(i3, i2, i1) = val1;
		    D1D2D1L2_D1D2FM2(i2, i3, i1) = val1;
		    D1D2D1L2_D1D2FM2(i3, i2, i1) = val1;
		    _D2D2D1L2_D2D2FM2(i1, i2, i3) = val1;
		    _D2D2D1L2_D2D2FM2(i1, i3, i2) = val1;
		    D1D1D2L2(i2, i3, i1) = val1;
		    D1D1D2L2(i3, i2, i1) = val1;
		    D1D2D2L2(i2, i3, i1) = val1;
		    D1D2D2L2(i3, i2, i1) = val1;
		    _D2D2D2L2(i1, i2, i3) = val1;
		    _D2D2D2L2(i1, i3, i2) = val1;
		}
		if(i2 < nd) {
		    D1D1D1L2_D1D1FM2(i1, i3, i2) = val1;
		    D1D1D1L2_D1D1FM2(i3, i1, i2) = val1;
		    D1D2D1L2_D1D2FM2(i1, i3, i2) = val1;
		    D1D2D1L2_D1D2FM2(i3, i1, i2) = val1;
		    _D2D2D1L2_D2D2FM2(i2, i1, i3) = val1;
		    _D2D2D1L2_D2D2FM2(i2, i3, i1) = val1;
		    D1D1D2L2(i1, i3, i2) = val1;
		    D1D1D2L2(i3, i1, i2) = val1;
		    D1D2D2L2(i1, i3, i2) = val1;
		    D1D2D2L2(i3, i1, i2) = val1;
		    _D2D2D2L2(i2, i1, i3) = val1;
		    _D2D2D2L2(i2, i3, i1) = val1;
		}
		if(i3 < nd) {
		    D1D1D1L2_D1D1FM2(i1, i2, i3) = val1;
		    D1D1D1L2_D1D1FM2(i2, i1, i3) = val1;
		    D1D2D1L2_D1D2FM2(i1, i2, i3) = val1;
		    D1D2D1L2_D1D2FM2(i2, i1, i3) = val1;
		    _D2D2D1L2_D2D2FM2(i3, i1, i2) = val1;
		    _D2D2D1L2_D2D2FM2(i3, i2, i1) = val1;
		    D1D1D2L2(i1, i2, i3) = val1;
		    D1D1D2L2(i2, i1, i3) = val1;
		    D1D2D2L2(i1, i2, i3) = val1;
		    D1D2D2L2(i2, i1, i3) = val1;
		    _D2D2D2L2(i3, i1, i2) = val1;
		    _D2D2D2L2(i3, i2, i1) = val1;
		}
	    }
	}
    }
}
  
static void calc_deriv2_cache_ddqdqdq(MidpointVI *mvi)
{
    int i1, i2, i3;
    Config *q1, *q2, *q3;
    double val2;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nk = System_KIN_CONFIGS(mvi->system);
    DECLARE_F_IDX3(mvi->D1D1D1L2_D1D1fm2, D1D1D1L2_D1D1fm2);
    DECLARE_F_IDX3(mvi->D1D2D1L2_D1D2fm2, D1D2D1L2_D1D2fm2);          
    DECLARE_F_IDX3(mvi->_D2D2D1L2_D2D2fm2, _D2D2D1L2_D2D2fm2);
    DECLARE_F_IDX3(mvi->D1D1D2L2, D1D1D2L2);
    DECLARE_F_IDX3(mvi->D1D2D2L2, D1D2D2L2);
    DECLARE_F_IDX3(mvi->_D2D2D2L2, _D2D2D2L2);

    for(i1 = 0; i1 < nd+nk; i1++) { 
	q1 = System_CONFIG(mvi->system, i1);
	
    	for(i2 = 0; i2 < nd+nk; i2++) {
    	    q2 = System_CONFIG(mvi->system, i2);

	    val2 = 0.25*System_L_ddqdqdq(mvi->system, q1, q2, q2);

	    if(i1 < nd) {
		D1D1D1L2_D1D1FM2(i2, i2, i1) -= val2;
		D1D2D1L2_D1D2FM2(i2, i2, i1) -= val2;
		_D2D2D1L2_D2D2FM2(i1, i2, i2) -= val2;
		D1D1D2L2(i2, i2, i1) += val2;
		D1D2D2L2(i2, i2, i1) += val2;
		_D2D2D2L2(i1, i2, i2) += val2;
	    }
	    if(i2 < nd) {
		D1D1D1L2_D1D1FM2(i1, i2, i2) -= val2;
		D1D1D1L2_D1D1FM2(i2, i1, i2) -= val2;
		D1D2D1L2_D1D2FM2(i1, i2, i2) -= val2;
		D1D2D1L2_D1D2FM2(i2, i1, i2) += val2;
		_D2D2D1L2_D2D2FM2(i2, i1, i2) += val2;
		_D2D2D1L2_D2D2FM2(i2, i2, i1) += val2;
		D1D1D2L2(i2, i1, i2) -= val2;
		D1D1D2L2(i1, i2, i2) -= val2;
		D1D2D2L2(i1, i2, i2) -= val2;
		D1D2D2L2(i2, i1, i2) += val2;
		_D2D2D2L2(i2, i1, i2) += val2;
		_D2D2D2L2(i2, i2, i1) += val2;
	    }
		
	    for(i3 = 0; i3 < i2; i3++) { 
		q3 = System_CONFIG(mvi->system, i3);

		if(i1 >= nd && i2 >= nd && i3 >= nd)
		    continue;
		
		val2 = 0.25*System_L_ddqdqdq(mvi->system, q1, q2, q3);
		if(i1 < nd) {
		    D1D1D1L2_D1D1FM2(i2, i3, i1) -= val2;
		    D1D1D1L2_D1D1FM2(i3, i2, i1) -= val2;
		    D1D2D1L2_D1D2FM2(i2, i3, i1) -= val2;		
		    D1D2D1L2_D1D2FM2(i3, i2, i1) -= val2;		
		    _D2D2D1L2_D2D2FM2(i1, i2, i3) -= val2;
		    _D2D2D1L2_D2D2FM2(i1, i3, i2) -= val2;
		    D1D1D2L2(i2, i3, i1) += val2;
		    D1D1D2L2(i3, i2, i1) += val2;
		    D1D2D2L2(i2, i3, i1) += val2;
		    D1D2D2L2(i3, i2, i1) += val2;
		    _D2D2D2L2(i1, i2, i3) += val2;
		    _D2D2D2L2(i1, i3, i2) += val2;
		}
		if(i2 < nd) {
		    D1D1D1L2_D1D1FM2(i1, i3, i2) -= val2;
		    D1D1D1L2_D1D1FM2(i3, i1, i2) -= val2;
		    D1D2D1L2_D1D2FM2(i1, i3, i2) -= val2;		
		    D1D2D1L2_D1D2FM2(i3, i1, i2) += val2;		
		    _D2D2D1L2_D2D2FM2(i2, i1, i3) += val2;
		    _D2D2D1L2_D2D2FM2(i2, i3, i1) += val2;
		    D1D1D2L2(i1, i3, i2) -= val2;
		    D1D1D2L2(i3, i1, i2) -= val2;
		    D1D2D2L2(i1, i3, i2) -= val2;
		    D1D2D2L2(i3, i1, i2) += val2;
		    _D2D2D2L2(i2, i1, i3) += val2;
		    _D2D2D2L2(i2, i3, i1) += val2;
		}
		if(i3 < nd) {
		    D1D1D1L2_D1D1FM2(i1, i2, i3) -= val2;
		    D1D1D1L2_D1D1FM2(i2, i1, i3) -= val2;
		    D1D2D1L2_D1D2FM2(i1, i2, i3) -= val2;		
		    D1D2D1L2_D1D2FM2(i2, i1, i3) += val2;		
		    _D2D2D1L2_D2D2FM2(i3, i1, i2) += val2;
		    _D2D2D1L2_D2D2FM2(i3, i2, i1) += val2;
		    D1D1D2L2(i1, i2, i3) -= val2;
		    D1D1D2L2(i2, i1, i3) -= val2;
		    D1D2D2L2(i1, i2, i3) -= val2;
		    D1D2D2L2(i2, i1, i3) += val2;
		    _D2D2D2L2(i3, i1, i2) += val2;
		    _D2D2D2L2(i3, i2, i1) += val2;
		}
	    }
	}
    }
}

static void calc_deriv2_cache_ddqddqdq(MidpointVI *mvi)
{
    int i1, i2, i3;
    Config *q1, *q2, *q3;
    double val3;
    double dt = mvi->t2 - mvi->t1;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nk = System_KIN_CONFIGS(mvi->system);
    DECLARE_F_IDX3(mvi->D1D1D1L2_D1D1fm2, D1D1D1L2_D1D1fm2);
    DECLARE_F_IDX3(mvi->D1D2D1L2_D1D2fm2, D1D2D1L2_D1D2fm2);          
    DECLARE_F_IDX3(mvi->_D2D2D1L2_D2D2fm2, _D2D2D1L2_D2D2fm2);
    DECLARE_F_IDX3(mvi->D1D1D2L2, D1D1D2L2);
    DECLARE_F_IDX3(mvi->D1D2D2L2, D1D2D2L2);
    DECLARE_F_IDX3(mvi->_D2D2D2L2, _D2D2D2L2);

    for(i1 = 0; i1 < nd+nk; i1++) { 
	q1 = System_CONFIG(mvi->system, i1);
	
    	for(i2 = 0; i2 < nd+nk; i2++) {
    	    q2 = System_CONFIG(mvi->system, i2);

	    val3 = 0.5/dt*System_L_ddqddqdq(mvi->system, q2, q2, q1);

	    if(i1 < nd) {
		D1D1D1L2_D1D1FM2(i2, i2, i1) += val3;
		D1D2D1L2_D1D2FM2(i2, i2, i1) -= val3;
		_D2D2D1L2_D2D2FM2(i1, i2, i2) += val3;
		D1D1D2L2(i2, i2, i1) += val3;
		D1D2D2L2(i2, i2, i1) -= val3;
		_D2D2D2L2(i1, i2, i2) += val3;
	    }
	    if(i2 < nd) {
		D1D1D1L2_D1D1FM2(i2, i1, i2) += val3;
		D1D1D1L2_D1D1FM2(i1, i2, i2) += val3;
		D1D2D1L2_D1D2FM2(i1, i2, i2) -= val3;
		D1D2D1L2_D1D2FM2(i2, i1, i2) += val3;
		_D2D2D1L2_D2D2FM2(i2, i1, i2) -= val3;
		_D2D2D1L2_D2D2FM2(i2, i2, i1) -= val3;
		D1D1D2L2(i1, i2, i2) -= val3;
		D1D1D2L2(i2, i1, i2) -= val3;
		D1D2D2L2(i1, i2, i2) += val3;
		D1D2D2L2(i2, i1, i2) -= val3;
		_D2D2D2L2(i2, i1, i2) += val3;
		_D2D2D2L2(i2, i2, i1) += val3;
	    }
	    
	    for(i3 = 0; i3 < i2; i3++) { 
		q3 = System_CONFIG(mvi->system, i3);

		if(i1 >= nd && i2 >= nd && i3 >= nd)
		    continue;
		
		val3 = 0.5/dt*System_L_ddqddqdq(mvi->system, q3, q2, q1);

		if(i1 < nd) {
		    D1D1D1L2_D1D1FM2(i2, i3, i1) += val3;
		    D1D1D1L2_D1D1FM2(i3, i2, i1) += val3;
		    D1D2D1L2_D1D2FM2(i2, i3, i1) -= val3;
		    D1D2D1L2_D1D2FM2(i3, i2, i1) -= val3;
		    _D2D2D1L2_D2D2FM2(i1, i2, i3) += val3;
		    _D2D2D1L2_D2D2FM2(i1, i3, i2) += val3;
		    D1D1D2L2(i2, i3, i1) += val3;
		    D1D1D2L2(i3, i2, i1) += val3;
		    D1D2D2L2(i2, i3, i1) -= val3;
		    D1D2D2L2(i3, i2, i1) -= val3;
		    _D2D2D2L2(i1, i2, i3) += val3;	
		    _D2D2D2L2(i1, i3, i2) += val3;
		}
		if(i2 < nd) {
		    D1D1D1L2_D1D1FM2(i1, i3, i2) += val3;
		    D1D1D1L2_D1D1FM2(i3, i1, i2) += val3;
		    D1D2D1L2_D1D2FM2(i1, i3, i2) -= val3;
		    D1D2D1L2_D1D2FM2(i3, i1, i2) += val3;
		    _D2D2D1L2_D2D2FM2(i2, i1, i3) -= val3;
		    _D2D2D1L2_D2D2FM2(i2, i3, i1) -= val3;
		    D1D1D2L2(i1, i3, i2) -= val3;
		    D1D1D2L2(i3, i1, i2) -= val3;
		    D1D2D2L2(i1, i3, i2) += val3;
		    D1D2D2L2(i3, i1, i2) -= val3;
		    _D2D2D2L2(i2, i1, i3) += val3;
		    _D2D2D2L2(i2, i3, i1) += val3;
		}
		if(i3 < nd) {
		    D1D1D1L2_D1D1FM2(i1, i2, i3) += val3;
		    D1D1D1L2_D1D1FM2(i2, i1, i3) += val3;
		    D1D2D1L2_D1D2FM2(i1, i2, i3) -= val3;
		    D1D2D1L2_D1D2FM2(i2, i1, i3) += val3;
		    _D2D2D1L2_D2D2FM2(i3, i1, i2) -= val3;
		    _D2D2D1L2_D2D2FM2(i3, i2, i1) -= val3;
		    D1D1D2L2(i1, i2, i3) -= val3;
		    D1D1D2L2(i2, i1, i3) -= val3;
		    D1D2D2L2(i1, i2, i3) += val3;
		    D1D2D2L2(i2, i1, i3) -= val3;
		    _D2D2D2L2(i3, i1, i2) += val3;
		    _D2D2D2L2(i3, i2, i1) += val3;
		}
	    }
	}
    }
}

static int calc_deriv2_cache_f(MidpointVI *mvi)
{
    int i1, i2, i3;
    Config *q1, *q2, *q3;
    Input *u1, *u2;
    double dt = mvi->t2 - mvi->t1;
    double val1, val2, val3, val4;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nk = System_KIN_CONFIGS(mvi->system);
    int nu = System_INPUTS(mvi->system);
    DECLARE_F_IDX3(mvi->D1D1D1L2_D1D1fm2, D1D1D1L2_D1D1fm2);
    DECLARE_F_IDX3(mvi->D1D2D1L2_D1D2fm2, D1D2D1L2_D1D2fm2);          
    DECLARE_F_IDX3(mvi->_D2D2D1L2_D2D2fm2, _D2D2D1L2_D2D2fm2);
    DECLARE_F_IDX3(mvi->D1D3fm2, D1D3fm2);
    DECLARE_F_IDX3(mvi->D2D3fm2, D2D3fm2);
    DECLARE_F_IDX3(mvi->D3D3fm2, D3D3fm2);

    for(i3 = 0; i3 < nd; i3++) {
	q3 = System_CONFIG(mvi->system, i3);
	
	for(i1 = 0; i1 < nd+nk; i1++) { 
	    q1 = System_CONFIG(mvi->system, i1);

	    val1 = 0.25 * dt * System_F_dqdq(mvi->system, q3, q1, q1);
	    val2 = 1.0 / dt * System_F_ddqddq(mvi->system, q3, q1, q1);
	    val3 = 0.5 * System_F_ddqdq(mvi->system, q3, q1, q1);
	    D1D1D1L2_D1D1FM2(i1, i1, i3) += val1 + val2 - val3 - val3;
	    D1D2D1L2_D1D2FM2(i1, i1, i3) += val1 - val2;
	    _D2D2D1L2_D2D2FM2(i3, i1, i1) += val1 + val2 + val3 + val3;
	    
	    for(i2 = 0; i2 < i1; i2++) {
		q2 = System_CONFIG(mvi->system, i2);

		val1 = 0.25 * dt * System_F_dqdq(mvi->system, q3, q1, q2);
		val2 = 1.0 / dt * System_F_ddqddq(mvi->system, q3, q1, q2);
		val3 = 0.5 * System_F_ddqdq(mvi->system, q3, q1, q2);
		val4 = 0.5 * System_F_ddqdq(mvi->system, q3, q2, q1);
		D1D1D1L2_D1D1FM2(i1, i2, i3) += val1 + val2 - val3 - val4;
		D1D1D1L2_D1D1FM2(i2, i1, i3) += val1 + val2 - val3 - val4;
		D1D2D1L2_D1D2FM2(i1, i2, i3) += val1 - val2 - val3 + val4;
		D1D2D1L2_D1D2FM2(i2, i1, i3) += val1 - val2 + val3 - val4;
		_D2D2D1L2_D2D2FM2(i3, i1, i2) += val1 + val2 + val3 + val4;
		_D2D2D1L2_D2D2FM2(i3, i2, i1) += val1 + val2 + val3 + val4;
	    }
	}
    }

    for(i1 = 0; i1 < nd+nk; i1++) {
	q1 = System_CONFIG(mvi->system, i1);
	for(i2 = 0; i2 < nu; i2++) {
	    u2 = System_INPUT(mvi->system, i2);
	    for(i3 = 0; i3 < nd; i3++) {
		q3 = System_CONFIG(mvi->system, i3);
		val1 = 0.5 * dt * System_F_dudq(mvi->system, q3, u2, q1);
		val2 = 1.0 * System_F_duddq(mvi->system, q3, u2, q1);	    
		D1D3FM2(i1, i2, i3) = val1 - val2;
		D2D3FM2(i1, i2, i3) = val1 + val2;
	    }
	}
    }

    for(i1 = 0; i1 < nu; i1++) {
	u1 = System_INPUT(mvi->system, i1);
	for(i2 = 0; i2 <= i1; i2++) {
	    u2 = System_INPUT(mvi->system, i2);
	    for(i3 = 0; i3 < nd; i3++) {
		q3 = System_CONFIG(mvi->system, i3);
		val1 = dt * System_F_dudu(mvi->system, q3, u1, u2);
		D3D3FM2(i1, i2, i3) = val1;
		D3D3FM2(i2, i1, i3) = val1;
	    }
	}
    }

    if(PyErr_Occurred())
	return -1;
    return 0;
}

static int calc_deriv2_cache(MidpointVI *mvi)
{
    /* Build the cache tables for the values we're going to need.
     * This just makes profiling easier since it becomes easier to
     * tell how much time is spent in the tree cache vs how much time
     * is spent in the functions themselves.  They are not necessary
     * as they they would be called automatically by the functions
     * below.
     */
    build_g_dqdqdq_cache(mvi->system);
    build_vb_dqdqdq_cache(mvi->system);
    build_vb_ddqdqdq_cache(mvi->system);
    /* calc_deriv2_cache_dqdqdq must be called first because it sets
     * the tables as opposed to adding to them! */
    calc_deriv2_cache_dqdqdq(mvi);
    calc_deriv2_cache_ddqdqdq(mvi);
    calc_deriv2_cache_ddqddqdq(mvi);
    if(calc_deriv2_cache_f(mvi))
	return -1;
    
    if(PyErr_Occurred())
	return -1;
    return 0;
}

static int calc_h1_deriv2(MidpointVI *mvi)
{
    int i1, i2, i3, i4;
    Config *q1;
    Config *q2;
    Config *q3;
    double val;
    Constraint *constraint;
    DECLARE_F_IDX4(mvi->DDDh1T, DDDh1T);
    
    int nd = System_DYN_CONFIGS(mvi->system);
    int nk = System_KIN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    
    for(i4 = 0; i4 < nc; i4++) {
	constraint = System_CONSTRAINT(mvi->system, i4);
	for(i1 = 0; i1 < nd+nk; i1++) {
	    q1 = System_CONFIG(mvi->system, i1);
	    for(i2 = i1; i2 < nd+nk; i2++) {
		q2 = System_CONFIG(mvi->system, i2);
		for(i3 = i2; i3 < nd+nk; i3++) {
		    q3 = System_CONFIG(mvi->system, i3);
		    val = constraint->h_dqdqdq(constraint, q1, q2, q3);
		    DDDH1T(i1, i2, i3, i4) = val;
		    DDDH1T(i1, i3, i2, i4) = val;
		    DDDH1T(i2, i1, i3, i4) = val;
		    DDDH1T(i2, i3, i1, i4) = val;
		    DDDH1T(i3, i1, i2, i4) = val;
		    DDDH1T(i3, i2, i1, i4) = val;
		}
	    }
	}
    }
    if(PyErr_Occurred())
	return -1;
    return 0;    
}

static int calc_h2_deriv2(MidpointVI *mvi)
{
    int i1, i2, i3;
    Config *q1;
    Config *q2;
    Constraint *constraint;
    DECLARE_F_IDX3(mvi->DDh2, DDh2);

    int nd = System_DYN_CONFIGS(mvi->system);
    int nk = System_KIN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);

    for(i1 = 0; i1 < nd+nk; i1++) {
	q1 = System_CONFIG(mvi->system, i1);
	for(i2 = 0; i2 < nd+nk; i2++) {
	    q2 = System_CONFIG(mvi->system, i2);
	    for(i3 = 0; i3 < nc; i3++) {
		constraint = System_CONSTRAINT(mvi->system, i3);
		DDH2(i3, i1, i2) = constraint->h_dqdq(constraint, q1, q2);
	    }
	}
    }
    if(PyErr_Occurred())
	return -1;
    return 0;    
}

// Called by calc_deriv2_dq1 to calculate q2_dq1dq1 and
// lambda1_dq1dq1.  
static void calc_deriv2_dq1_dq1(MidpointVI *mvi, int i, int j)
{
    int i1, i2;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    double *C_dq1 = (double*)malloc(sizeof(double)*nd);
    DECLARE_F_IDX2(mvi->q2_dq1, q2_dq1);            
    DECLARE_F_IDX2(mvi->l1_dq1, l1_dq1);
    DECLARE_F_IDX3(mvi->dq2_dq1_op, dq2_dq1_op);
    DECLARE_F_IDX3(mvi->dl1_dq1_op, dl1_dq1_op);
    DECLARE_F_IDX3(mvi->dp2_dq1_op, dp2_dq1_op);
    DECLARE_F_IDX3(mvi->q2_dq1dq1, q2_dq1dq1);        
    DECLARE_F_IDX3(mvi->l1_dq1dq1, l1_dq1dq1);        
    DECLARE_F_IDX3(mvi->p2_dq1dq1, p2_dq1dq1);        
    DECLARE_F_IDX3(mvi->DDh1T, DDh1T);
    DECLARE_F_IDX4(mvi->DDDh1T, DDDh1T);             
    DECLARE_F_IDX3(mvi->D1D1D1L2_D1D1fm2, D1D1D1L2_D1D1fm2);
    DECLARE_F_IDX3(mvi->D1D2D1L2_D1D2fm2, D1D2D1L2_D1D2fm2);          
    DECLARE_F_IDX3(mvi->D1D1D2L2, D1D1D2L2);
    DECLARE_F_IDX3(mvi->D1D2D2L2, D1D2D2L2);
    DECLARE_F_IDX2(mvi->D2D2L2, D2D2L2);
    DECLARE_F_IDX2(mvi->Dh1T, Dh1T);
    DECLARE_F_IDX1(mvi->lambda1, __lambda1);
     
    for(i1 = 0; i1 < nd; i1++) {
	C_dq1[i1] =  - D1D1D1L2_D1D1FM2(j, i, i1);
	for(i2 = 0; i2 < nc; i2++) {
	    C_dq1[i1] += DDH1T(j, i1, i2)*L1_DQ1(i, i2);
	    C_dq1[i1] += DDH1T(i, i1, i2)*L1_DQ1(j, i2);
	    C_dq1[i1] += DDDH1T(j, i, i1, i2)*LAMBDA1(i2);
	}
	for(i2 = 0; i2 < nd; i2++) {
	    C_dq1[i1] -= D1D2D1L2_D1D2FM2(j, i2, i1)*Q2_DQ1(i, i2);
	    C_dq1[i1] -= D1D2D1L2_D1D2FM2(i, i2, i1)*Q2_DQ1(j, i2);
	    C_dq1[i1] -= DQ2_DQ1_OP(i1, j, i2)*Q2_DQ1(i, i2);
	}
    }

    // Save C_dq1 to q2_dq1dq1
    copy_vector(&Q2_DQ1DQ1(j, i, 0), C_dq1, nd);	
    // Solve for lambda_dq1dq1
    if(nc > 0) {
	LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, C_dq1);
	mul_matvec_c_np_c(&L1_DQ1DQ1(j, i, 0), nc, mvi->Dh2, C_dq1, nd);
	for(i1 = 0; i1 < nc; i1++)
	    for(i2 = 0; i2 < nd; i2++)
		L1_DQ1DQ1(j, i, i1) += DL1_DQ1_OP(i1, j, i2)*Q2_DQ1(i, i2);    
	LU_solve_vec(mvi->proj_lu, nc, mvi->proj_lu_index, &L1_DQ1DQ1(j, i, 0));
        // Solve for q2_dq1dq1
	for(i1 = 0; i1 < nd; i1++) {
	    for(i2 = 0; i2 < nc; i2++)
		Q2_DQ1DQ1(j, i, i1) += DH1T(i1, i2)*L1_DQ1DQ1(j, i, i2);
	}
    }
    LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, &Q2_DQ1DQ1(j, i, 0));

    // Solve for p2_dq1dq1
    for(i1 = 0; i1 < nd; i1++) {
	P2_DQ1DQ1(j, i, i1) =  D1D1D2L2(j, i, i1);
	for(i2 = 0; i2 < nd; i2++) {
	    P2_DQ1DQ1(j, i, i1) += D1D2D2L2(j, i2, i1)*Q2_DQ1(i, i2);   // Might be worth swapping indices
	    P2_DQ1DQ1(j, i, i1) += D1D2D2L2(i, i2, i1)*Q2_DQ1(j, i2);   // to put i2 at the end?
	    P2_DQ1DQ1(j, i, i1) += D2D2L2(i2, i1)*Q2_DQ1DQ1(j, i, i2);    // here too?
	    P2_DQ1DQ1(j, i, i1) += DP2_DQ1_OP(i1, j, i2)*Q2_DQ1(i, i2);
	}
    }

    if(i != j) {
	// Take care of symmetric terms
	if(nc > 0)
	    copy_vector(&L1_DQ1DQ1(i, j, 0), &L1_DQ1DQ1(j, i, 0), nc);
	copy_vector(&Q2_DQ1DQ1(i, j, 0), &Q2_DQ1DQ1(j, i, 0), nd);
	copy_vector(&P2_DQ1DQ1(i, j, 0), &P2_DQ1DQ1(j, i, 0), nd);
    }
    free(C_dq1);
}

// Called by calc_deriv2_dq1 to calculate q2_dq1dp1 and
// lambda1_dq1dp1.  
static void calc_deriv2_dq1_dp1(MidpointVI *mvi, int i, int j)
{
    int i1, i2;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    double *C_dq1 = (double*)malloc(sizeof(double)*nd);
    DECLARE_F_IDX2(mvi->q2_dp1, q2_dp1);             
    DECLARE_F_IDX2(mvi->l1_dp1, l1_dp1);             
    DECLARE_F_IDX3(mvi->dq2_dq1_op, dq2_dq1_op);
    DECLARE_F_IDX3(mvi->dl1_dq1_op, dl1_dq1_op);
    DECLARE_F_IDX3(mvi->dp2_dq1_op, dp2_dq1_op);
    DECLARE_F_IDX3(mvi->q2_dq1dp1, q2_dq1dp1);        
    DECLARE_F_IDX3(mvi->l1_dq1dp1, l1_dq1dp1);        
    DECLARE_F_IDX3(mvi->p2_dq1dp1, p2_dq1dp1);        
    DECLARE_F_IDX3(mvi->DDh1T, DDh1T);
    DECLARE_F_IDX3(mvi->D1D2D1L2_D1D2fm2, D1D2D1L2_D1D2fm2);          
    DECLARE_F_IDX3(mvi->D1D2D2L2, D1D2D2L2);
    DECLARE_F_IDX2(mvi->D2D2L2, D2D2L2);
    DECLARE_F_IDX2(mvi->Dh1T, Dh1T);
    
    for(i1 = 0; i1 < nd; i1++) {
	C_dq1[i1] =  0.0;
	for(i2 = 0; i2 < nc; i2++) {
	    C_dq1[i1] += DDH1T(j, i1, i2)*L1_DP1(i, i2);
	}
	for(i2 = 0; i2 < nd; i2++) {
	    C_dq1[i1] -= D1D2D1L2_D1D2FM2(j, i2, i1)*Q2_DP1(i, i2);
	    C_dq1[i1] -= DQ2_DQ1_OP(i1, j, i2)*Q2_DP1(i, i2);
	}
    }
    // Save c_dq1dp1 to q2_dq1dp1
    copy_vector(&Q2_DQ1DP1(j, i, 0), C_dq1, nd);
    if(nc > 0) {
	// Solve for lambda_dq1dp1
	LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, C_dq1);
	mul_matvec_c_np_c(&L1_DQ1DP1(j, i, 0), nc, mvi->Dh2, C_dq1, nd);
	for(i1 = 0; i1 < nc; i1++)
	    for(i2 = 0; i2 < nd; i2++)
		L1_DQ1DP1(j, i, i1) += DL1_DQ1_OP(i1, j, i2)*Q2_DP1(i, i2);    
	LU_solve_vec(mvi->proj_lu, nc, mvi->proj_lu_index, &L1_DQ1DP1(j, i, 0));
        // Solve for q2_dq1dp1
	for(i1 = 0; i1 < nd; i1++) {
	    for(i2 = 0; i2 < nc; i2++)
		Q2_DQ1DP1(j, i, i1) += DH1T(i1, i2)*L1_DQ1DP1(j, i, i2);
	}
    }
    LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, &Q2_DQ1DP1(j, i, 0));

    // Solve for p2_dq1dp1
    for(i1 = 0; i1 < nd; i1++) {
	P2_DQ1DP1(j, i, i1) =  0.0;
	for(i2 = 0; i2 < nd; i2++) {
	    P2_DQ1DP1(j, i, i1) += D1D2D2L2(j, i2, i1)*Q2_DP1(i, i2);
	    P2_DQ1DP1(j, i, i1) += D2D2L2(i2, i1)*Q2_DQ1DP1(j, i, i2);
	    P2_DQ1DP1(j, i, i1) += DP2_DQ1_OP(i1, j, i2)*Q2_DP1(i, i2);
	}
    }	   

    free(C_dq1);
}

// Called by calc_deriv2_dq1 to calculate q2_dq1du1 and
// l1_dq1du1.  
static void calc_deriv2_dq1_du1(MidpointVI *mvi, int i, int j)
{
    int i1, i2;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    double *C_dq1 = (double*)malloc(sizeof(double)*nd);
    DECLARE_F_IDX2(mvi->q2_dq1, q2_dq1);            
    DECLARE_F_IDX2(mvi->q2_du1, q2_du1);             
    DECLARE_F_IDX2(mvi->l1_du1, l1_du1);             
    DECLARE_F_IDX3(mvi->dq2_dq1_op, dq2_dq1_op);
    DECLARE_F_IDX3(mvi->dl1_dq1_op, dl1_dq1_op);
    DECLARE_F_IDX3(mvi->dp2_dq1_op, dp2_dq1_op);
    DECLARE_F_IDX3(mvi->q2_dq1du1, q2_dq1du1);        
    DECLARE_F_IDX3(mvi->l1_dq1du1, l1_dq1du1);        
    DECLARE_F_IDX3(mvi->p2_dq1du1, p2_dq1du1);        
    DECLARE_F_IDX3(mvi->DDh1T, DDh1T);
    DECLARE_F_IDX3(mvi->D1D2D1L2_D1D2fm2, D1D2D1L2_D1D2fm2);          
    DECLARE_F_IDX3(mvi->D1D2D2L2, D1D2D2L2);
    DECLARE_F_IDX3(mvi->D1D3fm2, D1D3fm2);
    DECLARE_F_IDX3(mvi->D2D3fm2, D2D3fm2);
    DECLARE_F_IDX2(mvi->D2D2L2, D2D2L2);
    DECLARE_F_IDX2(mvi->Dh1T, Dh1T);
    
    for(i1 = 0; i1 < nd; i1++) {
	C_dq1[i1] = - D1D3FM2(j, i, i1);
	for(i2 = 0; i2 < nc; i2++) 
	    C_dq1[i1] += DDH1T(j, i1, i2)*L1_DU1(i, i2);
	for(i2 = 0; i2 < nd; i2++) {
	    C_dq1[i1] -= D2D3FM2(i2, i, i1)*Q2_DQ1(j, i2);
	    C_dq1[i1] -= D1D2D1L2_D1D2FM2(j, i2, i1)*Q2_DU1(i, i2);
	    C_dq1[i1] -= DQ2_DQ1_OP(i1, j, i2)*Q2_DU1(i, i2);
	}
    }
    // Save c_dq1du1 to q2_dq1du1
    copy_vector(&Q2_DQ1DU1(j, i, 0), C_dq1, nd);	
    // Solve for lambda_dq1du1
    if(nc > 0) {
	LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, C_dq1);
	mul_matvec_c_np_c(&L1_DQ1DU1(j, i, 0), nc, mvi->Dh2, C_dq1, nd);
	for(i1 = 0; i1 < nc; i1++)
	    for(i2 = 0; i2 < nd; i2++)
		L1_DQ1DU1(j, i, i1) += DL1_DQ1_OP(i1, j, i2)*Q2_DU1(i, i2);    
	LU_solve_vec(mvi->proj_lu, nc, mvi->proj_lu_index, &L1_DQ1DU1(j, i, 0));
        // Solve for q2_dq1du1
	for(i1 = 0; i1 < nd; i1++) {
	    for(i2 = 0; i2 < nc; i2++)
		Q2_DQ1DU1(j, i, i1) += DH1T(i1, i2)*L1_DQ1DU1(j, i, i2);
	}
    }
    LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, &Q2_DQ1DU1(j, i, 0));

    // Solve for p2_dq1du1
    for(i1 = 0; i1 < nd; i1++) {
	P2_DQ1DU1(j, i, i1) = 0.0;
	for(i2 = 0; i2 < nd; i2++) {
	    P2_DQ1DU1(j, i, i1) += D1D2D2L2(j, i2, i1)*Q2_DU1(i, i2);
	    P2_DQ1DU1(j, i, i1) += D2D2L2(i2, i1)*Q2_DQ1DU1(j, i, i2);
	    P2_DQ1DU1(j, i, i1) += DP2_DQ1_OP(i1, j, i2)*Q2_DU1(i, i2);
	}
    }

    free(C_dq1);
}

// Called by calc_deriv2_dq1 to calculate q2_dq1dk2 and
// lambda1_dq1dk2.  
static void calc_deriv2_dq1_dk2(MidpointVI *mvi, int i, int j)
{
    int i1, i2;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    double *C_dq1 = (double*)malloc(sizeof(double)*nd);
    DECLARE_F_IDX2(mvi->q2_dq1, q2_dq1);            
    DECLARE_F_IDX2(mvi->q2_dk2, q2_dk2);              
    DECLARE_F_IDX2(mvi->l1_dk2, l1_dk2);              
    DECLARE_F_IDX3(mvi->dq2_dq1_op, dq2_dq1_op);
    DECLARE_F_IDX3(mvi->dl1_dq1_op, dl1_dq1_op);
    DECLARE_F_IDX3(mvi->dp2_dq1_op, dp2_dq1_op);
    DECLARE_F_IDX3(mvi->q2_dq1dk2, q2_dq1dk2);        
    DECLARE_F_IDX3(mvi->l1_dq1dk2, l1_dq1dk2);        
    DECLARE_F_IDX3(mvi->p2_dq1dk2, p2_dq1dk2);        
    DECLARE_F_IDX3(mvi->DDh2, DDh2);
    DECLARE_F_IDX3(mvi->DDh1T, DDh1T);
    DECLARE_F_IDX3(mvi->D1D2D1L2_D1D2fm2, D1D2D1L2_D1D2fm2);          
    DECLARE_F_IDX3(mvi->_D2D2D1L2_D2D2fm2, _D2D2D1L2_D2D2fm2);
    DECLARE_F_IDX3(mvi->D1D2D2L2, D1D2D2L2);
    DECLARE_F_IDX3(mvi->_D2D2D2L2, _D2D2D2L2);
    DECLARE_F_IDX2(mvi->D2D2L2, D2D2L2);
    DECLARE_F_IDX2(mvi->Dh1T, Dh1T);

    for(i1 = 0; i1 < nd; i1++) {
	C_dq1[i1] =  - D1D2D1L2_D1D2FM2(j, nd+i, i1);
	for(i2 = 0; i2 < nc; i2++)
	    C_dq1[i1] += DDH1T(j, i1, i2)*L1_DK2(i, i2);
	for(i2 = 0; i2 < nd; i2++) {
	    C_dq1[i1] -= _D2D2D1L2_D2D2FM2(i1, nd+i, i2)*Q2_DQ1(j, i2);
	    C_dq1[i1] -= D1D2D1L2_D1D2FM2(j, i2, i1)*Q2_DK2(i, i2);
	    C_dq1[i1] -= DQ2_DQ1_OP(i1, j, i2)*Q2_DK2(i, i2);
	}
    }
    // Save c_dq1dk2 to q2_dq1dk2
    copy_vector(&Q2_DQ1DK2(j, i, 0), C_dq1, nd);	
    // Solve for lambda_dq1dk2
    if(nc > 0) {
	LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, C_dq1);
	mul_matvec_c_np_c(&L1_DQ1DK2(j, i, 0), nc, mvi->Dh2, C_dq1, nd);
	for(i1 = 0; i1 < nc; i1++)
	    for(i2 = 0; i2 < nd; i2++)
		L1_DQ1DK2(j, i, i1) += DL1_DQ1_OP(i1, j, i2)*Q2_DK2(i, i2);    
	for(i1 = 0; i1 < nd; i1++)
	    for(i2 = 0; i2 < nc; i2++) 
		L1_DQ1DK2(j, i, i2) += DDH2(i2, i1, nd+i)*Q2_DQ1(j, i1);
			
	LU_solve_vec(mvi->proj_lu, nc, mvi->proj_lu_index, &L1_DQ1DK2(j, i, 0));
	    
	// Solve for q2_dq1dk2
	for(i1 = 0; i1 < nd; i1++) {
	    for(i2 = 0; i2 < nc; i2++)
		Q2_DQ1DK2(j, i, i1) += DH1T(i1, i2)*L1_DQ1DK2(j, i, i2);
	}
    }
    LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, &Q2_DQ1DK2(j, i, 0));

    // Solve for p2_dq1dk2
    for(i1 = 0; i1 < nd; i1++) {
	P2_DQ1DK2(j, i, i1) =  D1D2D2L2(j, nd+i, i1);
	for(i2 = 0; i2 < nd; i2++) {
	    P2_DQ1DK2(j, i, i1) += _D2D2D2L2(i1, nd+i, i2)*Q2_DQ1(j, i2);
	    P2_DQ1DK2(j, i, i1) += D1D2D2L2(j, i2, i1)*Q2_DK2(i, i2);
	    P2_DQ1DK2(j, i, i1) += D2D2L2(i2, i1)*Q2_DQ1DK2(j, i, i2);
	    P2_DQ1DK2(j, i, i1) += DP2_DQ1_OP(i1, j, i2)*Q2_DK2(i, i2);
	}
    }

    free(C_dq1);
}

static void calc_d_dq1_op(MidpointVI *mvi, int j)
{
    int i1, i2, i3;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    DECLARE_F_IDX2(mvi->q2_dq1, q2_dq1);            
    DECLARE_F_IDX3(mvi->dq2_dq1_op, dq2_dq1_op);
    DECLARE_F_IDX3(mvi->dl1_dq1_op, dl1_dq1_op);
    DECLARE_F_IDX3(mvi->dp2_dq1_op, dp2_dq1_op);
    DECLARE_F_IDX3(mvi->DDh2, DDh2);
    DECLARE_F_IDX3(mvi->_D2D2D1L2_D2D2fm2, _D2D2D1L2_D2D2fm2);
    DECLARE_F_IDX3(mvi->_D2D2D2L2, _D2D2D2L2);

    // Calculate [D2D2D1L2_D2D2fm2]*q2_dq1[j]
    // Calculate [D2D2D2L2]*q2_dq1[j]
    for(i1 = 0; i1 < nd; i1++) {
	for(i2 = 0; i2 < nd; i2++) {
	    DQ2_DQ1_OP(i1, j, i2) = 0.0; 
	    DP2_DQ1_OP(i1, j, i2) = 0.0; 
	    for(i3 = 0; i3 < nd; i3++) {
		DQ2_DQ1_OP(i1, j, i2) += _D2D2D1L2_D2D2FM2(i1, i2, i3)*Q2_DQ1(j, i3);
		DP2_DQ1_OP(i1, j, i2) += _D2D2D2L2(i1, i2, i3)*Q2_DQ1(j, i3);
	    }
	}
    }

    // Calculate DDh_2 * q2_dq1[j]
    for(i1 = 0; i1 < nc; i1++) {
	for(i2 = 0; i2 < nd; i2++) {
	    DL1_DQ1_OP(i1, j, i2) = 0.0; 
	    for(i3 = 0; i3 < nd; i3++) {
		DL1_DQ1_OP(i1, j, i2) += DDH2(i1, i2, i3)*Q2_DQ1(j, i3);
	    }
	} 
    }	

}

static int calc_deriv2_dq1_row(MidpointVI *mvi, int j)
{
    int i;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nk = System_KIN_CONFIGS(mvi->system);
    int nu = System_INPUTS(mvi->system);

    calc_d_dq1_op(mvi, j);

    for(i = j; i < nd+nk; i++)
	calc_deriv2_dq1_dq1(mvi, i, j);
    for(i = 0; i < nd; i++) 
	calc_deriv2_dq1_dp1(mvi, i, j);
    for(i = 0; i < nu; i++) 
	calc_deriv2_dq1_du1(mvi, i, j);
    for(i = 0; i < nk; i++) 
	calc_deriv2_dq1_dk2(mvi, i, j);

    return 0;
}


// Called by calc_deriv2_dp1 to calculate q2_dp1dp1 and
// lambda1_dp1dp1.  
static void calc_deriv2_dp1_dp1(MidpointVI *mvi, int i, int j)
{
    int i1, i2;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    double *C_dp1 = (double*)malloc(sizeof(double)*nd);
    DECLARE_F_IDX2(mvi->q2_dp1, q2_dp1);
    DECLARE_F_IDX3(mvi->dq2_dp1_op, dq2_dp1_op);
    DECLARE_F_IDX3(mvi->dl1_dp1_op, dl1_dp1_op);
    DECLARE_F_IDX3(mvi->dp2_dp1_op, dp2_dp1_op);
    DECLARE_F_IDX3(mvi->q2_dp1dp1, q2_dp1dp1);        
    DECLARE_F_IDX3(mvi->l1_dp1dp1, l1_dp1dp1);        
    DECLARE_F_IDX3(mvi->p2_dp1dp1, p2_dp1dp1);        
    DECLARE_F_IDX2(mvi->D2D2L2, D2D2L2);
    DECLARE_F_IDX2(mvi->Dh1T, Dh1T);

    for(i1 = 0; i1 < nd; i1++) {
	C_dp1[i1] =  0.0;
	for(i2 = 0; i2 < nd; i2++) {
	    C_dp1[i1] -= DQ2_DP1_OP(i1, j, i2)*Q2_DP1(i, i2);
	}
    }
    // Save c_dp1dp1 to q2_dp1dp1
    copy_vector(&Q2_DP1DP1(j, i, 0), C_dp1, nd);	
    // Solve for lambda_dp1dp1
    if(nc > 0) {
	LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, C_dp1);
	mul_matvec_c_np_c(&L1_DP1DP1(j, i, 0), nc, mvi->Dh2, C_dp1, nd);
	for(i1 = 0; i1 < nc; i1++)
	    for(i2 = 0; i2 < nd; i2++)
		L1_DP1DP1(j, i, i1) += DL1_DP1_OP(i1, j, i2)*Q2_DP1(i, i2);    
	LU_solve_vec(mvi->proj_lu, nc, mvi->proj_lu_index, &L1_DP1DP1(j, i, 0));
	// Solve for q2_dp1dp1
	for(i1 = 0; i1 < nd; i1++) {
	    for(i2 = 0; i2 < nc; i2++)
		Q2_DP1DP1(j, i, i1) += DH1T(i1, i2)*L1_DP1DP1(j, i, i2);
	}
    }
    LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, &Q2_DP1DP1(j, i, 0));

    // Solve for p2_dp2dp1
    for(i1 = 0; i1 < nd; i1++) {
	P2_DP1DP1(j, i, i1) =  0.0;
	for(i2 = 0; i2 < nd; i2++) {
	    P2_DP1DP1(j, i, i1) += D2D2L2(i2, i1)*Q2_DP1DP1(j, i, i2);
	    P2_DP1DP1(j, i, i1) += DP2_DP1_OP(i1, j, i2)*Q2_DP1(i, i2);
	}
    }	   

    if(j != i) {
	if(nc > 0)
	    copy_vector(&L1_DP1DP1(i, j, 0), &L1_DP1DP1(j, i, 0), nc);
	copy_vector(&Q2_DP1DP1(i, j, 0), &Q2_DP1DP1(j, i, 0), nd);
	copy_vector(&P2_DP1DP1(i, j, 0), &P2_DP1DP1(j, i, 0), nd);
    }
    
    free(C_dp1);
}

// Called by calc_deriv2_dp1 to calculate q2_dp1du1 and
// lambda1_dp1du1.  
static void calc_deriv2_dp1_du1(MidpointVI *mvi, int i, int j)
{
    int i1, i2;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    double *C_dp1 = (double*)malloc(sizeof(double)*nd);
    DECLARE_F_IDX2(mvi->q2_dp1, q2_dp1);
    DECLARE_F_IDX2(mvi->q2_du1, q2_du1);             
    DECLARE_F_IDX3(mvi->dq2_dp1_op, dq2_dp1_op);
    DECLARE_F_IDX3(mvi->dl1_dp1_op, dl1_dp1_op);
    DECLARE_F_IDX3(mvi->dp2_dp1_op, dp2_dp1_op);
    DECLARE_F_IDX3(mvi->q2_dp1du1, q2_dp1du1);        
    DECLARE_F_IDX3(mvi->l1_dp1du1, l1_dp1du1);        
    DECLARE_F_IDX3(mvi->p2_dp1du1, p2_dp1du1);        
    DECLARE_F_IDX3(mvi->D2D3fm2, D2D3fm2);
    DECLARE_F_IDX2(mvi->D2D2L2, D2D2L2);
    DECLARE_F_IDX2(mvi->Dh1T, Dh1T);

    for(i1 = 0; i1 < nd; i1++) {
	C_dp1[i1] =  0.0;
	for(i2 = 0; i2 < nd; i2++) {
	    C_dp1[i1] -= D2D3FM2(i2, i, i1)*Q2_DP1(j, i2);
	    C_dp1[i1] -= DQ2_DP1_OP(i1, j, i2)*Q2_DU1(i, i2);
	}
    }
    // Save c_dp1du1 to q2_dp1du1
    copy_vector(&Q2_DP1DU1(j, i, 0), C_dp1, nd);
    if(nc > 0) {
	// Solve for lambda_dp1du1
	LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, C_dp1);
	mul_matvec_c_np_c(&L1_DP1DU1(j, i, 0), nc, mvi->Dh2, C_dp1, nd);
	for(i1 = 0; i1 < nc; i1++)
	    for(i2 = 0; i2 < nd; i2++)
		L1_DP1DU1(j, i, i1) += DL1_DP1_OP(i1, j, i2)*Q2_DU1(i, i2);    
	LU_solve_vec(mvi->proj_lu, nc, mvi->proj_lu_index, &L1_DP1DU1(j, i, 0));
        // Solve for q2_dp1du1
	for(i1 = 0; i1 < nd; i1++) {
	    for(i2 = 0; i2 < nc; i2++)
		Q2_DP1DU1(j, i, i1) += DH1T(i1, i2)*L1_DP1DU1(j, i, i2);
	}
    }
    LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, &Q2_DP1DU1(j, i, 0));

    // Solve for p2_dp1du1
    for(i1 = 0; i1 < nd; i1++) {
	P2_DP1DU1(j, i, i1) =  0.0;
	for(i2 = 0; i2 < nd; i2++) {
	    P2_DP1DU1(j, i, i1) += D2D2L2(i2, i1)*Q2_DP1DU1(j, i, i2);
	    P2_DP1DU1(j, i, i1) += DP2_DP1_OP(i1, j, i2)*Q2_DU1(i, i2);
	}
    }
    
    free(C_dp1);
}

// Called by calc_deriv2_dp1 to calculate q2_dp1dk2 and
// lambda1_dp1dk2.  
static void calc_deriv2_dp1_dk2(MidpointVI *mvi, int i, int j)
{
    int i1, i2;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    double *C_dp1 = (double*)malloc(sizeof(double)*nd);
    DECLARE_F_IDX2(mvi->q2_dp1, q2_dp1);
    DECLARE_F_IDX2(mvi->q2_dk2, q2_dk2);              
    DECLARE_F_IDX3(mvi->q2_dp1dk2, q2_dp1dk2);        
    DECLARE_F_IDX3(mvi->l1_dp1dk2, l1_dp1dk2);        
    DECLARE_F_IDX3(mvi->p2_dp1dk2, p2_dp1dk2);        
    DECLARE_F_IDX3(mvi->dq2_dp1_op, dq2_dp1_op);
    DECLARE_F_IDX3(mvi->dl1_dp1_op, dl1_dp1_op);
    DECLARE_F_IDX3(mvi->dp2_dp1_op, dp2_dp1_op);
    DECLARE_F_IDX3(mvi->DDh2, DDh2);
    DECLARE_F_IDX3(mvi->_D2D2D1L2_D2D2fm2, _D2D2D1L2_D2D2fm2);
    DECLARE_F_IDX3(mvi->_D2D2D2L2, _D2D2D2L2);
    DECLARE_F_IDX2(mvi->D2D2L2, D2D2L2);
    DECLARE_F_IDX2(mvi->Dh1T, Dh1T);

    for(i1 = 0; i1 < nd; i1++) {
	C_dp1[i1] =  0.0;
	for(i2 = 0; i2 < nd; i2++) {
	    C_dp1[i1] -= _D2D2D1L2_D2D2FM2(i1, nd+i, i2)*Q2_DP1(j, i2);
	    C_dp1[i1] -= DQ2_DP1_OP(i1, j, i2)*Q2_DK2(i, i2);
	}
    }

    // Save c_dp1dk2 to q2_dp1dk2
    copy_vector(&Q2_DP1DK2(j, i, 0), C_dp1, nd);
    if(nc > 0) {
	// Solve for lambda_dp1dk2
	LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, C_dp1);
	mul_matvec_c_np_c(&L1_DP1DK2(j, i, 0), nc, mvi->Dh2, C_dp1, nd);
	for(i1 = 0; i1 < nc; i1++)
	    for(i2 = 0; i2 < nd; i2++)
		L1_DP1DK2(j, i, i1) += DL1_DP1_OP(i1, j, i2)*Q2_DK2(i, i2);    
	for(i1 = 0; i1 < nd; i1++)
	    for(i2 = 0; i2 < nc; i2++) 
		L1_DP1DK2(j, i, i2) += DDH2(i2, i1, nd+i)*Q2_DP1(j, i1);
	
	LU_solve_vec(mvi->proj_lu, nc, mvi->proj_lu_index, &L1_DP1DK2(j, i, 0));
	    
        // Solve for q2_dp1dk2
	for(i1 = 0; i1 < nd; i1++) {
	    for(i2 = 0; i2 < nc; i2++)
		Q2_DP1DK2(j, i, i1) += DH1T(i1, i2)*L1_DP1DK2(j, i, i2);
	}
    }
    LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, &Q2_DP1DK2(j, i, 0));

    // Solve for p2_dp1dk2
    for(i1 = 0; i1 < nd; i1++) {
	P2_DP1DK2(j, i, i1) =  0.0;
	for(i2 = 0; i2 < nd; i2++) {
	    P2_DP1DK2(j, i, i1) += _D2D2D2L2(i1, nd+i, i2)*Q2_DP1(j, i2);
	    P2_DP1DK2(j, i, i1) += D2D2L2(i2, i1)*Q2_DP1DK2(j, i, i2);
	    P2_DP1DK2(j, i, i1) += DP2_DP1_OP(i1, j, i2)*Q2_DK2(i, i2);
	}
    }

    free(C_dp1);
}

static void calc_d_dp1_op(MidpointVI *mvi, int j)
{
    int i1, i2, i3;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    DECLARE_F_IDX2(mvi->q2_dp1, q2_dp1);
    DECLARE_F_IDX3(mvi->dq2_dp1_op, dq2_dp1_op);
    DECLARE_F_IDX3(mvi->dl1_dp1_op, dl1_dp1_op);
    DECLARE_F_IDX3(mvi->dp2_dp1_op, dp2_dp1_op);
    DECLARE_F_IDX3(mvi->DDh2, DDh2);
    DECLARE_F_IDX3(mvi->_D2D2D1L2_D2D2fm2, _D2D2D1L2_D2D2fm2);
    DECLARE_F_IDX3(mvi->_D2D2D2L2, _D2D2D2L2);
    
    // Calculate [D2D2D1L2_D2D2fm2]*q2_dp1[j]
    // Calculate [D2D2D2L2]*q2_dp1[j]
    for(i1 = 0; i1 < nd; i1++) {
	for(i2 = 0; i2 < nd; i2++) {
	    DQ2_DP1_OP(i1, j, i2) = 0.0; 
	    DP2_DP1_OP(i1, j, i2) = 0.0; 
	    for(i3 = 0; i3 < nd; i3++) {
		DQ2_DP1_OP(i1, j, i2) += _D2D2D1L2_D2D2FM2(i1, i2, i3)*Q2_DP1(j, i3);
		DP2_DP1_OP(i1, j, i2) += _D2D2D2L2(i1, i2, i3)*Q2_DP1(j, i3);
	    }
	} 
    }

    // Calculate DDh_2 * q2_dp1[j]
    for(i1 = 0; i1 < nc; i1++) {
	for(i2 = 0; i2 < nd; i2++) {
	    DL1_DP1_OP(i1, j, i2) = 0.0; 
	    for(i3 = 0; i3 < nd; i3++) {
		DL1_DP1_OP(i1, j, i2) += DDH2(i1, i2, i3)*Q2_DP1(j, i3);
	    }
	} 
    }
}    

static int calc_deriv2_dp1_row(MidpointVI *mvi, int j)
{
    int i;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nu = System_INPUTS(mvi->system);
    int nk = System_KIN_CONFIGS(mvi->system);
    
    calc_d_dp1_op(mvi, j);
    
    for(i = j; i < nd; i++)
	calc_deriv2_dp1_dp1(mvi, i, j);
    for(i = 0; i < nu; i++) 
	calc_deriv2_dp1_du1(mvi, i, j);
    for(i = 0; i < nk; i++) 
	calc_deriv2_dp1_dk2(mvi, i, j);

    return 0;    
}

// Called by calc_deriv2_du1 to calculate q2_du1du1 and
// lambda1_du1du1.  
static void calc_deriv2_du1_du1(MidpointVI *mvi, int i, int j)
{
    int i1, i2;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    double *C_du1 = (double*)malloc(sizeof(double)*nd);
    DECLARE_F_IDX2(mvi->q2_du1, q2_du1);             
    DECLARE_F_IDX3(mvi->q2_du1du1, q2_du1du1);        
    DECLARE_F_IDX3(mvi->l1_du1du1, l1_du1du1);        
    DECLARE_F_IDX3(mvi->p2_du1du1, p2_du1du1);        
    DECLARE_F_IDX3(mvi->dq2_du1_op, dq2_du1_op);     
    DECLARE_F_IDX3(mvi->dl1_du1_op, dl1_du1_op);     
    DECLARE_F_IDX3(mvi->dp2_du1_op, dp2_du1_op);     
    DECLARE_F_IDX3(mvi->D2D3fm2, D2D3fm2);
    DECLARE_F_IDX3(mvi->D3D3fm2, D3D3fm2);
    DECLARE_F_IDX2(mvi->D2D2L2, D2D2L2);
    DECLARE_F_IDX2(mvi->Dh1T, Dh1T);

    for(i1 = 0; i1 < nd; i1++) {
	C_du1[i1] = - D3D3FM2(j, i, i1);
	for(i2 = 0; i2 < nd; i2++) {
	    C_du1[i1] -= D2D3FM2(i2, i, i1)*Q2_DU1(j, i2);
	    C_du1[i1] -= D2D3FM2(i2, j, i1)*Q2_DU1(i, i2);
	    C_du1[i1] -= DQ2_DU1_OP(i1, j, i2)*Q2_DU1(i, i2);
	}
    }
    // Save c_dq1du1 to q2_du1du1
    copy_vector(&Q2_DU1DU1(j, i, 0), C_du1, nd);
    if(nc > 0) {
	// Solve for lambda_du1du1
	LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, C_du1);
	mul_matvec_c_np_c(&L1_DU1DU1(j, i, 0), nc, mvi->Dh2, C_du1, nd);
	for(i1 = 0; i1 < nc; i1++)
	    for(i2 = 0; i2 < nd; i2++)
		L1_DU1DU1(j, i, i1) += DL1_DU1_OP(i1, j, i2)*Q2_DU1(i, i2);    
	LU_solve_vec(mvi->proj_lu, nc, mvi->proj_lu_index, &L1_DU1DU1(j, i, 0));
        // Solve for q2_du1du1
	for(i1 = 0; i1 < nd; i1++) {
	    for(i2 = 0; i2 < nc; i2++)
		Q2_DU1DU1(j, i, i1) += DH1T(i1, i2)*L1_DU1DU1(j, i, i2);
	}
    }
    LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, &Q2_DU1DU1(j, i, 0));

    for(i1 = 0; i1 < nd; i1++) {
	P2_DU1DU1(j, i, i1) =  0.0;
	for(i2 = 0; i2 < nd; i2++) {
	    P2_DU1DU1(j, i, i1) += D2D2L2(i2, i1)*Q2_DU1DU1(j, i, i2);
	    P2_DU1DU1(j, i, i1) += DP2_DU1_OP(i1, j, i2)*Q2_DU1(i, i2);
	}
    }

    if(i != j) {
	if(nc > 0)
	    copy_vector(&L1_DU1DU1(i, j, 0), &L1_DU1DU1(j, i, 0), nc);
	copy_vector(&Q2_DU1DU1(i, j, 0), &Q2_DU1DU1(j, i, 0), nd);
	copy_vector(&P2_DU1DU1(i, j, 0), &P2_DU1DU1(j, i, 0), nd);
    }
    free(C_du1);
}

// Called by calc_deriv2_du1 to calculate q2_du1dk2 and
// lambda1_du1dk2.  
static void calc_deriv2_du1_dk2(MidpointVI *mvi, int i, int j)
{
    int i1, i2;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    double *C_du1 = (double*)malloc(sizeof(double)*nd);
    DECLARE_F_IDX3(mvi->q2_du1dk2, q2_du1dk2);        
    DECLARE_F_IDX3(mvi->l1_du1dk2, l1_du1dk2);        
    DECLARE_F_IDX3(mvi->p2_du1dk2, p2_du1dk2);        
    DECLARE_F_IDX3(mvi->dq2_du1_op, dq2_du1_op);      
    DECLARE_F_IDX3(mvi->dl1_du1_op, dl1_du1_op);      
    DECLARE_F_IDX3(mvi->dp2_du1_op, dp2_du1_op);      
    DECLARE_F_IDX3(mvi->_D2D2D1L2_D2D2fm2, _D2D2D1L2_D2D2fm2);        
    DECLARE_F_IDX3(mvi->_D2D2D2L2, _D2D2D2L2);        
    DECLARE_F_IDX2(mvi->D2D2L2, D2D2L2);              
    DECLARE_F_IDX3(mvi->D2D3fm2, D2D3fm2);            
    DECLARE_F_IDX3(mvi->DDh2, DDh2);                  
    DECLARE_F_IDX2(mvi->q2_dk2, q2_dk2);              
    DECLARE_F_IDX2(mvi->q2_du1, q2_du1);              
    DECLARE_F_IDX2(mvi->Dh1T, Dh1T);                  

    for(i1 = 0; i1 < nd; i1++) {
	C_du1[i1] = - D2D3FM2(nd+i, j, i1);
	for(i2 = 0; i2 < nd; i2++) {
	    C_du1[i1] -= _D2D2D1L2_D2D2FM2(i1, nd+i, i2)*Q2_DU1(j, i2);
	    C_du1[i1] -= D2D3FM2(i2, j, i1)*Q2_DK2(i, i2);
	    C_du1[i1] -= DQ2_DU1_OP(i1, j, i2)*Q2_DK2(i, i2);
	}
    }
	    
    // Save c_du1dk2 to q2_du1dk2
    copy_vector(&Q2_DU1DK2(j, i, 0), C_du1, nd);
    if(nc > 0) {
	// Solve for lambda_du1dk2
	LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, C_du1);
	mul_matvec_c_np_c(&L1_DU1DK2(j, i, 0), nc, mvi->Dh2, C_du1, nd);
	for(i1 = 0; i1 < nc; i1++)
	    for(i2 = 0; i2 < nd; i2++)
		L1_DU1DK2(j, i, i1) += DL1_DU1_OP(i1, j, i2)*Q2_DK2(i, i2);    
	for(i1 = 0; i1 < nd; i1++)
	    for(i2 = 0; i2 < nc; i2++) 
		L1_DU1DK2(j, i, i2) += DDH2(i2, i1, nd+i)*Q2_DU1(j, i1);
	
	LU_solve_vec(mvi->proj_lu, nc, mvi->proj_lu_index, &L1_DU1DK2(j, i, 0));
	
	// Solve for q2_du1dk2
	for(i1 = 0; i1 < nd; i1++) {
	    for(i2 = 0; i2 < nc; i2++)
		Q2_DU1DK2(j, i, i1) += DH1T(i1, i2)*L1_DU1DK2(j, i, i2);
	}
    }
    LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, &Q2_DU1DK2(j, i, 0));

    for(i1 = 0; i1 < nd; i1++) {
	P2_DU1DK2(j, i, i1) = 0.0;
	for(i2 = 0; i2 < nd; i2++) {
	    P2_DU1DK2(j, i, i1) += _D2D2D2L2(i1, nd+i, i2)*Q2_DU1(j, i2);
	    P2_DU1DK2(j, i, i1) += D2D2L2(i2, i1)*Q2_DU1DK2(j, i, i2);
	    P2_DU1DK2(j, i, i1) += DP2_DU1_OP(i1, j, i2)*Q2_DK2(i, i2);
	}
    }

    free(C_du1);
}

static void calc_d_du1_op(MidpointVI *mvi, int j)
{
    int i1, i2, i3;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    DECLARE_F_IDX3(mvi->dq2_du1_op, dq2_du1_op);      
    DECLARE_F_IDX3(mvi->dl1_du1_op, dl1_du1_op);      
    DECLARE_F_IDX3(mvi->dp2_du1_op, dp2_du1_op);      
    DECLARE_F_IDX3(mvi->_D2D2D1L2_D2D2fm2, _D2D2D1L2_D2D2fm2);        
    DECLARE_F_IDX3(mvi->_D2D2D2L2, _D2D2D2L2);        
    DECLARE_F_IDX3(mvi->DDh2, DDh2);                  
    DECLARE_F_IDX2(mvi->q2_du1, q2_du1);              

    // Calculate [D2D2D1L2_D2D2fm2]*q2_u1[j]
    // Calculate [D2D2D2L2]*q2_du1[j]
    for(i1 = 0; i1 < nd; i1++) {
	for(i2 = 0; i2 < nd; i2++) {
	    DQ2_DU1_OP(i1, j, i2) = 0.0; 
	    DP2_DU1_OP(i1, j, i2) = 0.0; 
	    for(i3 = 0; i3 < nd; i3++) {
		DQ2_DU1_OP(i1, j, i2) += _D2D2D1L2_D2D2FM2(i1, i2, i3)*Q2_DU1(j, i3);
		DP2_DU1_OP(i1, j, i2) += _D2D2D2L2(i1, i2, i3)*Q2_DU1(j, i3);
	    }
	}
    }

    // Calculate DDh_2 * q2_du1[j]
    for(i1 = 0; i1 < nc; i1++) {
	for(i2 = 0; i2 < nd; i2++) {
	    DL1_DU1_OP(i1, j, i2) = 0.0; 
	    for(i3 = 0; i3 < nd; i3++) {
		DL1_DU1_OP(i1, j, i2) += DDH2(i1, i2, i3)*Q2_DU1(j, i3);
	    }
	} 
    }
}

static int calc_deriv2_du1_row(MidpointVI *mvi, int j)
{
    int i;
    int nu = System_INPUTS(mvi->system);
    int nk = System_KIN_CONFIGS(mvi->system);
    
    calc_d_du1_op(mvi, j);
    
    for(i = j; i < nu; i++)
	calc_deriv2_du1_du1(mvi, i, j);
    for(i = 0; i < nk; i++) 
	calc_deriv2_du1_dk2(mvi, i, j);
    return 0;
}


// Called by calc_deriv2_dk2 to calculate q2_dk2dk2 and
// lambda1_dk2dk2.  
static void calc_deriv2_dk2_dk2(MidpointVI *mvi, int i, int j)
{
    int i1, i2;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    double *C_dk2 = (double*)malloc(sizeof(double)*nd);
    DECLARE_F_IDX3(mvi->_D2D2D1L2_D2D2fm2, _D2D2D1L2_D2D2fm2);        
    DECLARE_F_IDX3(mvi->_D2D2D2L2, _D2D2D2L2);        
    DECLARE_F_IDX2(mvi->D2D2L2, D2D2L2);              
    DECLARE_F_IDX3(mvi->dq2_dk2_op, dq2_dk2_op);      
    DECLARE_F_IDX3(mvi->dl1_dk2_op, dl1_dk2_op);      
    DECLARE_F_IDX3(mvi->dp2_dk2_op, dp2_dk2_op);      
    DECLARE_F_IDX3(mvi->q2_dk2dk2, q2_dk2dk2);        
    DECLARE_F_IDX3(mvi->l1_dk2dk2, l1_dk2dk2);        
    DECLARE_F_IDX3(mvi->p2_dk2dk2, p2_dk2dk2);        
    DECLARE_F_IDX3(mvi->DDh2, DDh2);                  
    DECLARE_F_IDX2(mvi->q2_dk2, q2_dk2);              
    DECLARE_F_IDX2(mvi->Dh1T, Dh1T);                  

    for(i1 = 0; i1 < nd; i1++) {
	C_dk2[i1] =  - _D2D2D1L2_D2D2FM2(i1, nd+i, nd+j);
	for(i2 = 0; i2 < nd; i2++) {
	    C_dk2[i1] -= _D2D2D1L2_D2D2FM2(i1, nd+i, i2)*Q2_DK2(j, i2);
	    C_dk2[i1] -= _D2D2D1L2_D2D2FM2(i1, nd+j, i2)*Q2_DK2(i, i2);
	    C_dk2[i1] -= DQ2_DK2_OP(i1, j, i2)*Q2_DK2(i, i2);
	}
    }
	    
    // Save c_dk2dk2 to q2_dk2dk2
    copy_vector(&Q2_DK2DK2(j, i, 0), C_dk2 , nd);
    if(nc > 0) { 
	// Solve for lambda_dk2dk2
	LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, C_dk2);
	mul_matvec_c_np_c(&L1_DK2DK2(j, i, 0), nc, mvi->Dh2, C_dk2, nd);
	for(i1 = 0; i1 < nc; i1++)
	    for(i2 = 0; i2 < nd; i2++)
		L1_DK2DK2(j, i, i1) += DL1_DK2_OP(i1, j, i2)*Q2_DK2(i, i2);    
	for(i1 = 0; i1 < nd; i1++) 
	    for(i2 = 0; i2 < nc; i2++) 
		L1_DK2DK2(j, i, i2) += DDH2(i2, i1, nd+i)*Q2_DK2(j, i1);
	for(i1 = 0; i1 < nd; i1++) 
	    for(i2 = 0; i2 < nc; i2++) 
		L1_DK2DK2(j, i, i2) += DDH2(i2, i1, nd+j)*Q2_DK2(i, i1);
	for(i1 = 0; i1 < nc; i1++) 
	    L1_DK2DK2(j, i, i1) += DDH2(i1, nd+i, nd+j);
	
	LU_solve_vec(mvi->proj_lu, nc, mvi->proj_lu_index, &L1_DK2DK2(j, i, 0));
	    
        // Solve for q2_dk2dk2
	for(i1 = 0; i1 < nd; i1++) {
	    for(i2 = 0; i2 < nc; i2++)
		Q2_DK2DK2(j, i, i1) += DH1T(i1, i2)*L1_DK2DK2(j, i, i2);
	}
    }
    LU_solve_vec(mvi->M2_lu, nd, mvi->M2_lu_index, &Q2_DK2DK2(j, i, 0));

    // Calculate p2_dk2dk2
    for(i1 = 0; i1 < nd; i1++) {
	P2_DK2DK2(j, i, i1) =  _D2D2D2L2(i1, nd+i, nd+j);
	for(i2 = 0; i2 < nd; i2++) {
	    P2_DK2DK2(j, i, i1) += _D2D2D2L2(i1, nd+i, i2)*Q2_DK2(j, i2);
	    P2_DK2DK2(j, i, i1) += _D2D2D2L2(i1, nd+j, i2)*Q2_DK2(i, i2);
	    P2_DK2DK2(j, i, i1) += D2D2L2(i2, i1)*Q2_DK2DK2(j, i, i2);
	    P2_DK2DK2(j, i, i1) += DP2_DK2_OP(i1, j, i2)*Q2_DK2(i, i2);
	}
    }

    if(i != j) {
	if(nc > 0)
	    copy_vector(&L1_DK2DK2(i, j, 0), &L1_DK2DK2(j, i, 0), nc);
	copy_vector(&Q2_DK2DK2(i, j, 0), &Q2_DK2DK2(j, i, 0), nd);
	copy_vector(&P2_DK2DK2(i, j, 0), &P2_DK2DK2(j, i, 0), nd);
    }
    
    free(C_dk2);
}

static void calc_d_dk2_op(MidpointVI *mvi, int j)
{
    int i1, i2, i3;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nc = System_CONSTRAINTS(mvi->system);
    DECLARE_F_IDX3(mvi->dq2_dk2_op, dq2_dk2_op);      
    DECLARE_F_IDX3(mvi->dl1_dk2_op, dl1_dk2_op);      
    DECLARE_F_IDX3(mvi->dp2_dk2_op, dp2_dk2_op);      
    DECLARE_F_IDX3(mvi->_D2D2D1L2_D2D2fm2, _D2D2D1L2_D2D2fm2);        
    DECLARE_F_IDX3(mvi->_D2D2D2L2, _D2D2D2L2);        
    DECLARE_F_IDX3(mvi->DDh2, DDh2);                  
    DECLARE_F_IDX2(mvi->q2_dk2, q2_dk2);              
        
    // Calculate [D2D2D1L2_D2D2fm2]*q2_dk2[j] 
    // Calculate [D2D2D2L2]*q2_dk2[j]
    for(i1 = 0; i1 < nd; i1++) {
	for(i2 = 0; i2 < nd; i2++) {
	    DQ2_DK2_OP(i1, j, i2) = 0.0; 
	    DP2_DK2_OP(i1, j, i2) = 0.0; 
	    for(i3 = 0; i3 < nd; i3++) {
		DQ2_DK2_OP(i1, j, i2) += _D2D2D1L2_D2D2FM2(i1, i2, i3)*Q2_DK2(j, i3);
		DP2_DK2_OP(i1, j, i2) += _D2D2D2L2(i1, i2, i3)*Q2_DK2(j, i3);
	    }
	}
    }
    
    // Calculate DDh_2 * q2_dk2[j]
    for(i1 = 0; i1 < nc; i1++) {
	for(i2 = 0; i2 < nd; i2++) {
	    DL1_DK2_OP(i1, j, i2) = 0.0; 
	    for(i3 = 0; i3 < nd; i3++) {
		DL1_DK2_OP(i1, j, i2) += DDH2(i1, i2, i3)*Q2_DK2(j, i3);
	    }
	} 
    }    
}

static int calc_deriv2_dk2_row(MidpointVI *mvi, int j)
{
    int i;
    int nk = System_KIN_CONFIGS(mvi->system);

    calc_d_dk2_op(mvi, j);
    
    for(i = j; i < nk; i++) 
	calc_deriv2_dk2_dk2(mvi, i, j);
    return 0;
}

static int MidpointVI_calc_deriv2(MidpointVI *mvi)
{
    int j;
    int nd = System_DYN_CONFIGS(mvi->system);
    int nk = System_KIN_CONFIGS(mvi->system);
    int nu = System_INPUTS(mvi->system);
    
    if(mvi->cache & MIDPOINTVI_CACHE_SOLUTION_DERIV2)
	return 0;
    if(!(mvi->cache & MIDPOINTVI_CACHE_SOLUTION_DERIV1))
	if(MidpointVI_calc_deriv1(mvi))
	    return -1;
    
    MidpointVI_set_state(mvi, 1);
    if(calc_h1_deriv2(mvi)) return -1;
    MidpointVI_set_state(mvi, 2);
    if(calc_h2_deriv2(mvi)) return -1;
    MidpointVI_set_midpoint(mvi);
    if(calc_deriv2_cache(mvi)) return -1;

    for(j = 0; j < nd+nk; j++) queue_job(mvi, calc_deriv2_dq1_row, j);
    wait_for_jobs_to_finish(mvi);  // dp1 derivatives depend on dq1 results.
    for(j = 0; j < nd; j++) queue_job(mvi, calc_deriv2_dp1_row, j);
    for(j = 0; j < nu; j++) queue_job(mvi, calc_deriv2_du1_row, j);
    for(j = 0; j < nk; j++) queue_job(mvi, calc_deriv2_dk2_row, j);
    wait_for_jobs_to_finish(mvi);
    
    mvi->cache |= MIDPOINTVI_CACHE_SOLUTION_DERIV2;
    return 0;
}


/***********************************************************************
 * Python API
 **********************************************************************/

static void dealloc(MidpointVI *mvi)
{
    Py_CLEAR(mvi->system);

    Py_CLEAR(mvi->q1);
    Py_CLEAR(mvi->q2);
    Py_CLEAR(mvi->p1);
    Py_CLEAR(mvi->p2);
    Py_CLEAR(mvi->u1);
    Py_CLEAR(mvi->lambda1);
    
    Py_CLEAR(mvi->Dh1T);
    Py_CLEAR(mvi->Dh2);
    Py_CLEAR(mvi->f); 
    Py_CLEAR(mvi->Df);
    Py_CLEAR(mvi->Df_index);
    
    Py_CLEAR(mvi->DDh1T);
    Py_CLEAR(mvi->M2_lu);
    Py_CLEAR(mvi->M2_lu_index);
    Py_CLEAR(mvi->proj_lu);
    Py_CLEAR(mvi->proj_lu_index);

    Py_CLEAR(mvi->q2_dq1);
    Py_CLEAR(mvi->q2_dp1);
    Py_CLEAR(mvi->q2_du1);
    Py_CLEAR(mvi->q2_dk2);
    Py_CLEAR(mvi->p2_dq1);
    Py_CLEAR(mvi->p2_dp1);
    Py_CLEAR(mvi->p2_du1);
    Py_CLEAR(mvi->p2_dk2);
    Py_CLEAR(mvi->l1_dq1);
    Py_CLEAR(mvi->l1_dp1);
    Py_CLEAR(mvi->l1_du1);
    Py_CLEAR(mvi->l1_dk2);

    Py_CLEAR(mvi->q2_dq1dq1);
    Py_CLEAR(mvi->q2_dq1dp1);
    Py_CLEAR(mvi->q2_dq1du1);
    Py_CLEAR(mvi->q2_dq1dk2);
    Py_CLEAR(mvi->q2_dp1dp1);
    Py_CLEAR(mvi->q2_dp1du1);
    Py_CLEAR(mvi->q2_dp1dk2);
    Py_CLEAR(mvi->q2_du1du1);
    Py_CLEAR(mvi->q2_du1dk2);
    Py_CLEAR(mvi->q2_dk2dk2);

    Py_CLEAR(mvi->p2_dq1dq1);
    Py_CLEAR(mvi->p2_dq1dp1);
    Py_CLEAR(mvi->p2_dq1du1);
    Py_CLEAR(mvi->p2_dq1dk2);
    Py_CLEAR(mvi->p2_dp1dp1);
    Py_CLEAR(mvi->p2_dp1du1);
    Py_CLEAR(mvi->p2_dp1dk2);
    Py_CLEAR(mvi->p2_du1du1);
    Py_CLEAR(mvi->p2_du1dk2);
    Py_CLEAR(mvi->p2_dk2dk2);

    Py_CLEAR(mvi->l1_dq1dq1);
    Py_CLEAR(mvi->l1_dq1dp1);
    Py_CLEAR(mvi->l1_dq1du1);
    Py_CLEAR(mvi->l1_dq1dk2);
    Py_CLEAR(mvi->l1_dp1dp1);
    Py_CLEAR(mvi->l1_dp1du1);
    Py_CLEAR(mvi->l1_dp1dk2);
    Py_CLEAR(mvi->l1_du1du1);
    Py_CLEAR(mvi->l1_du1dk2);
    Py_CLEAR(mvi->l1_dk2dk2);

    Py_CLEAR(mvi->DDDh1T);
    Py_CLEAR(mvi->DDh2); 
    
    Py_CLEAR(mvi->temp_ndnc); 

    Py_CLEAR(mvi->D1D1L2_D1fm2);
    Py_CLEAR(mvi->D2D1L2_D2fm2);
    Py_CLEAR(mvi->D1D2L2);
    Py_CLEAR(mvi->D2D2L2);
    Py_CLEAR(mvi->D3fm2);

    Py_CLEAR(mvi->D1D1D1L2_D1D1fm2);
    Py_CLEAR(mvi->D1D2D1L2_D1D2fm2);
    Py_CLEAR(mvi->_D2D2D1L2_D2D2fm2);
    Py_CLEAR(mvi->D1D1D2L2);
    Py_CLEAR(mvi->D1D2D2L2);
    Py_CLEAR(mvi->_D2D2D2L2);
    Py_CLEAR(mvi->D1D3fm2);
    Py_CLEAR(mvi->D2D3fm2);
    Py_CLEAR(mvi->D3D3fm2);

    Py_CLEAR(mvi->dq2_dq1_op);
    Py_CLEAR(mvi->dl1_dq1_op);
    Py_CLEAR(mvi->dp2_dq1_op);
    Py_CLEAR(mvi->dq2_dp1_op);
    Py_CLEAR(mvi->dl1_dp1_op);
    Py_CLEAR(mvi->dp2_dp1_op);
    Py_CLEAR(mvi->dq2_du1_op);
    Py_CLEAR(mvi->dl1_du1_op);
    Py_CLEAR(mvi->dp2_du1_op);
    Py_CLEAR(mvi->dq2_dk2_op);
    Py_CLEAR(mvi->dl1_dk2_op);
    Py_CLEAR(mvi->dp2_dk2_op);
    
    mvi_kill_threading(mvi);
    mvi->ob_type->tp_free((PyObject*)mvi);
}

static int init(MidpointVI *mvi, PyObject *args, PyObject *kwds)
{
    mvi->tolerance = 1.0e-10;
    mvi->cache = 0;
    return 0;
}

static PyObject* set_midpoint(MidpointVI *mvi, PyObject *args)
{
    MidpointVI_set_midpoint(mvi);
    Py_RETURN_NONE;
}

static PyObject* calc_f_wrap(MidpointVI *mvi)
{
    if(MidpointVI_calc_f(mvi))
	return NULL;
    Py_RETURN_NONE;
}

static PyObject* calc_deriv1_wrap(MidpointVI *mvi)
{
    if(MidpointVI_calc_deriv1(mvi))
	return NULL;
    Py_RETURN_NONE;
}

static PyObject* calc_deriv2_wrap(MidpointVI *mvi)
{
    if(MidpointVI_calc_deriv2(mvi))
	return NULL;
    Py_RETURN_NONE;
}

static PyObject* calc_p2(MidpointVI *mvi, PyObject *args)
{
    MidpointVI_set_midpoint(mvi);
    if(MidpointVI_calc_p2(mvi))
	return NULL;
    Py_RETURN_NONE;
}

static PyObject* discrete_fm2(MidpointVI *mvi, PyObject *args)
{
    npy_intp size[1];
    PyArrayObject *array = NULL;
    int i;

    size[0] = System_DYN_CONFIGS(mvi->system);
    array = (PyArrayObject*)PyArray_SimpleNew(1, size, NPY_DOUBLE);

    MidpointVI_set_midpoint(mvi);
    for(i = 0; i < System_DYN_CONFIGS(mvi->system); i++) 
        IDX1_DBL(array, i) = fm2(mvi, System_DYN_CONFIG(mvi->system, i));
    if(PyErr_Occurred()) {
        Py_DECREF(array);
        return NULL;
    }
    return (PyObject*)array;
}

static PyObject *solve_DEL(MidpointVI *mvi, PyObject *args)
{
    int steps;
    int max_iterations;

    if(!PyArg_ParseTuple(args, "i", &max_iterations))
        return NULL;
    steps = MidpointVI_solve_DEL(mvi, max_iterations);    
    if(steps == -1)
	return NULL;
    return PyInt_FromLong(steps);
}

static PyObject *set_num_threads(MidpointVI *mvi, PyObject *args)
{
    int num_threads;

    if(!PyArg_ParseTuple(args, "i", &num_threads))
        return NULL;

    mvi_kill_threading(mvi);
    mvi_init_threading(mvi, num_threads);
    Py_RETURN_NONE;
}

static PyMethodDef methods_list[] = {
    {"set_midpoint", (PyCFunction)set_midpoint, METH_NOARGS, trep_internal_doc},
    //{"Ld", (PyCFunction)Ld, METH_NOARGS, trep_internal_doc},
    //{"D1Ld", (PyCFunction)D1Ld, METH_VARARGS, trep_internal_doc},
    //{"D2Ld", (PyCFunction)D2Ld, METH_VARARGS, trep_internal_doc},
    //{"D2D1Ld", (PyCFunction)D2D1Ld, METH_VARARGS, trep_internal_doc},
    {"_calc_f", (PyCFunction)calc_f_wrap, METH_NOARGS, trep_internal_doc},
    {"calc_p2", (PyCFunction)calc_p2, METH_NOARGS, trep_internal_doc},
    //{"Df", (PyCFunction)Df, METH_NOARGS, trep_internal_doc},
    {"_solve_DEL", (PyCFunction)solve_DEL, METH_VARARGS, trep_internal_doc},
    {"discrete_fm2", (PyCFunction)discrete_fm2, METH_NOARGS, trep_internal_doc},
    {"_calc_deriv1", (PyCFunction)calc_deriv1_wrap, METH_NOARGS, trep_internal_doc},
    {"_calc_deriv2", (PyCFunction)calc_deriv2_wrap, METH_NOARGS, trep_internal_doc},
    {"_set_num_threads", (PyCFunction)set_num_threads, METH_VARARGS, trep_internal_doc},
    {"_get_num_threads", (PyCFunction)set_num_threads, METH_NOARGS, trep_internal_doc},
    
    {NULL}  /* Sentinel */
};

static PyMemberDef members_list[] = {
    {"_system", T_OBJECT_EX, offsetof(MidpointVI, system), 0, trep_internal_doc},
    {"tolerance", T_DOUBLE, offsetof(MidpointVI, tolerance), 0, trep_internal_doc},
    {"_t1", T_DOUBLE, offsetof(MidpointVI, t1), 0, trep_internal_doc},
    {"_t2", T_DOUBLE, offsetof(MidpointVI, t2), 0, trep_internal_doc},
    {"_cache", T_ULONG, offsetof(MidpointVI, cache), 0, trep_internal_doc},

    {"_q1", T_OBJECT_EX, offsetof(MidpointVI, q1), 0, trep_internal_doc},
    {"_q2", T_OBJECT_EX, offsetof(MidpointVI, q2), 0, trep_internal_doc},
    {"_p1", T_OBJECT_EX, offsetof(MidpointVI, p1), 0, trep_internal_doc},
    {"_p2", T_OBJECT_EX, offsetof(MidpointVI, p2), 0, trep_internal_doc},
    {"_u1", T_OBJECT_EX, offsetof(MidpointVI, u1), 0, trep_internal_doc},
    {"_lambda1", T_OBJECT_EX, offsetof(MidpointVI, lambda1), 0, trep_internal_doc},

    {"_f", T_OBJECT_EX, offsetof(MidpointVI, f), 0, trep_internal_doc},

    {"_Df", T_OBJECT_EX, offsetof(MidpointVI, Df), 0, trep_internal_doc},
    {"_Df_index", T_OBJECT_EX, offsetof(MidpointVI, Df_index), 0, trep_internal_doc},
    {"_M2_lu", T_OBJECT_EX, offsetof(MidpointVI, M2_lu), 0, trep_internal_doc},
    {"_M2_lu_index", T_OBJECT_EX, offsetof(MidpointVI, M2_lu_index), 0, trep_internal_doc},
    {"_proj_lu", T_OBJECT_EX, offsetof(MidpointVI, proj_lu), 0, trep_internal_doc},
    {"_proj_lu_index", T_OBJECT_EX, offsetof(MidpointVI, proj_lu_index), 0, trep_internal_doc},
    {"_temp_ndnc", T_OBJECT_EX, offsetof(MidpointVI, temp_ndnc), 0, trep_internal_doc},
   
    {"_q2_dq1", T_OBJECT_EX, offsetof(MidpointVI, q2_dq1), 0, trep_internal_doc},
    {"_q2_dp1", T_OBJECT_EX, offsetof(MidpointVI, q2_dp1), 0, trep_internal_doc},
    {"_q2_du1", T_OBJECT_EX, offsetof(MidpointVI, q2_du1), 0, trep_internal_doc},
    {"_q2_dk2", T_OBJECT_EX, offsetof(MidpointVI, q2_dk2), 0, trep_internal_doc},
    {"_p2_dq1", T_OBJECT_EX, offsetof(MidpointVI, p2_dq1), 0, trep_internal_doc},
    {"_p2_dp1", T_OBJECT_EX, offsetof(MidpointVI, p2_dp1), 0, trep_internal_doc},
    {"_p2_du1", T_OBJECT_EX, offsetof(MidpointVI, p2_du1), 0, trep_internal_doc},
    {"_p2_dk2", T_OBJECT_EX, offsetof(MidpointVI, p2_dk2), 0, trep_internal_doc},
    {"_l1_dq1", T_OBJECT_EX, offsetof(MidpointVI, l1_dq1), 0, trep_internal_doc},
    {"_l1_dp1", T_OBJECT_EX, offsetof(MidpointVI, l1_dp1), 0, trep_internal_doc},
    {"_l1_du1", T_OBJECT_EX, offsetof(MidpointVI, l1_du1), 0, trep_internal_doc},
    {"_l1_dk2", T_OBJECT_EX, offsetof(MidpointVI, l1_dk2), 0, trep_internal_doc},

    {"_q2_dq1dq1", T_OBJECT_EX, offsetof(MidpointVI, q2_dq1dq1), 0, trep_internal_doc},
    {"_p2_dq1dq1", T_OBJECT_EX, offsetof(MidpointVI, p2_dq1dq1), 0, trep_internal_doc},
    {"_l1_dq1dq1", T_OBJECT_EX, offsetof(MidpointVI, l1_dq1dq1), 0, trep_internal_doc},
    {"_q2_dq1dp1", T_OBJECT_EX, offsetof(MidpointVI, q2_dq1dp1), 0, trep_internal_doc},
    {"_p2_dq1dp1", T_OBJECT_EX, offsetof(MidpointVI, p2_dq1dp1), 0, trep_internal_doc},
    {"_l1_dq1dp1", T_OBJECT_EX, offsetof(MidpointVI, l1_dq1dp1), 0, trep_internal_doc},
    {"_q2_dq1du1", T_OBJECT_EX, offsetof(MidpointVI, q2_dq1du1), 0, trep_internal_doc},
    {"_p2_dq1du1", T_OBJECT_EX, offsetof(MidpointVI, p2_dq1du1), 0, trep_internal_doc},
    {"_l1_dq1du1", T_OBJECT_EX, offsetof(MidpointVI, l1_dq1du1), 0, trep_internal_doc},
    {"_q2_dq1dk2", T_OBJECT_EX, offsetof(MidpointVI, q2_dq1dk2), 0, trep_internal_doc},
    {"_p2_dq1dk2", T_OBJECT_EX, offsetof(MidpointVI, p2_dq1dk2), 0, trep_internal_doc},
    {"_l1_dq1dk2", T_OBJECT_EX, offsetof(MidpointVI, l1_dq1dk2), 0, trep_internal_doc},
    {"_q2_dp1dp1", T_OBJECT_EX, offsetof(MidpointVI, q2_dp1dp1), 0, trep_internal_doc},
    {"_p2_dp1dp1", T_OBJECT_EX, offsetof(MidpointVI, p2_dp1dp1), 0, trep_internal_doc},
    {"_l1_dp1dp1", T_OBJECT_EX, offsetof(MidpointVI, l1_dp1dp1), 0, trep_internal_doc},
    {"_q2_dp1du1", T_OBJECT_EX, offsetof(MidpointVI, q2_dp1du1), 0, trep_internal_doc},
    {"_p2_dp1du1", T_OBJECT_EX, offsetof(MidpointVI, p2_dp1du1), 0, trep_internal_doc},
    {"_l1_dp1du1", T_OBJECT_EX, offsetof(MidpointVI, l1_dp1du1), 0, trep_internal_doc},
    {"_q2_dp1dk2", T_OBJECT_EX, offsetof(MidpointVI, q2_dp1dk2), 0, trep_internal_doc},
    {"_p2_dp1dk2", T_OBJECT_EX, offsetof(MidpointVI, p2_dp1dk2), 0, trep_internal_doc},
    {"_l1_dp1dk2", T_OBJECT_EX, offsetof(MidpointVI, l1_dp1dk2), 0, trep_internal_doc},
    {"_q2_du1du1", T_OBJECT_EX, offsetof(MidpointVI, q2_du1du1), 0, trep_internal_doc},
    {"_p2_du1du1", T_OBJECT_EX, offsetof(MidpointVI, p2_du1du1), 0, trep_internal_doc},
    {"_l1_du1du1", T_OBJECT_EX, offsetof(MidpointVI, l1_du1du1), 0, trep_internal_doc},
    {"_q2_du1dk2", T_OBJECT_EX, offsetof(MidpointVI, q2_du1dk2), 0, trep_internal_doc},
    {"_p2_du1dk2", T_OBJECT_EX, offsetof(MidpointVI, p2_du1dk2), 0, trep_internal_doc},
    {"_l1_du1dk2", T_OBJECT_EX, offsetof(MidpointVI, l1_du1dk2), 0, trep_internal_doc},
    {"_q2_dk2dk2", T_OBJECT_EX, offsetof(MidpointVI, q2_dk2dk2), 0, trep_internal_doc},
    {"_p2_dk2dk2", T_OBJECT_EX, offsetof(MidpointVI, p2_dk2dk2), 0, trep_internal_doc},
    {"_l1_dk2dk2", T_OBJECT_EX, offsetof(MidpointVI, l1_dk2dk2), 0, trep_internal_doc},
 
    {"_Dh1T", T_OBJECT_EX, offsetof(MidpointVI, Dh1T), 0, trep_internal_doc}, 
    {"_Dh2", T_OBJECT_EX, offsetof(MidpointVI, Dh2), 0, trep_internal_doc}, 
    {"_DDh1T", T_OBJECT_EX, offsetof(MidpointVI, DDh1T), 0, trep_internal_doc}, 
    {"_DDh2", T_OBJECT_EX, offsetof(MidpointVI, DDh2), 0, trep_internal_doc}, 
    {"_DDDh1T", T_OBJECT_EX, offsetof(MidpointVI, DDDh1T), 0, trep_internal_doc}, 

    {"_D1D1L2_D1fm2", T_OBJECT_EX, offsetof(MidpointVI, D1D1L2_D1fm2), 0, trep_internal_doc},
    {"_D2D1L2_D2fm2", T_OBJECT_EX, offsetof(MidpointVI, D2D1L2_D2fm2), 0, trep_internal_doc},
    {"_D1D2L2", T_OBJECT_EX, offsetof(MidpointVI, D1D2L2), 0, trep_internal_doc},
    {"_D2D2L2", T_OBJECT_EX, offsetof(MidpointVI, D2D2L2), 0, trep_internal_doc},
    {"_D3fm2", T_OBJECT_EX, offsetof(MidpointVI, D3fm2), 0, trep_internal_doc},
    {"_D1D1D1L2_D1D1fm2", T_OBJECT_EX, offsetof(MidpointVI, D1D1D1L2_D1D1fm2), 0, trep_internal_doc},
    {"_D1D2D1L2_D1D2fm2", T_OBJECT_EX, offsetof(MidpointVI, D1D2D1L2_D1D2fm2), 0, trep_internal_doc},
    {"_D2D2D1L2_D2D2fm2", T_OBJECT_EX, offsetof(MidpointVI, _D2D2D1L2_D2D2fm2), 0, trep_internal_doc},
    {"_D1D1D2L2", T_OBJECT_EX, offsetof(MidpointVI, D1D1D2L2), 0, trep_internal_doc},
    {"_D1D2D2L2", T_OBJECT_EX, offsetof(MidpointVI, D1D2D2L2), 0, trep_internal_doc},
    {"_D2D2D2L2", T_OBJECT_EX, offsetof(MidpointVI, _D2D2D2L2), 0, trep_internal_doc},
    
    {"_D1D3fm2", T_OBJECT_EX, offsetof(MidpointVI, D1D3fm2), 0, trep_internal_doc},
    {"_D2D3fm2", T_OBJECT_EX, offsetof(MidpointVI, D2D3fm2), 0, trep_internal_doc},
    {"_D3D3fm2", T_OBJECT_EX, offsetof(MidpointVI, D3D3fm2), 0, trep_internal_doc},
    
    {"_dq2_dq1_op", T_OBJECT_EX, offsetof(MidpointVI, dq2_dq1_op), 0, trep_internal_doc},
    {"_dl1_dq1_op", T_OBJECT_EX, offsetof(MidpointVI, dl1_dq1_op), 0, trep_internal_doc},
    {"_dp2_dq1_op", T_OBJECT_EX, offsetof(MidpointVI, dp2_dq1_op), 0, trep_internal_doc},
    {"_dq2_dp1_op", T_OBJECT_EX, offsetof(MidpointVI, dq2_dp1_op), 0, trep_internal_doc},
    {"_dl1_dp1_op", T_OBJECT_EX, offsetof(MidpointVI, dl1_dp1_op), 0, trep_internal_doc},
    {"_dp2_dp1_op", T_OBJECT_EX, offsetof(MidpointVI, dp2_dp1_op), 0, trep_internal_doc},
    {"_dq2_du1_op", T_OBJECT_EX, offsetof(MidpointVI, dq2_du1_op), 0, trep_internal_doc},
    {"_dl1_du1_op", T_OBJECT_EX, offsetof(MidpointVI, dl1_du1_op), 0, trep_internal_doc},
    {"_dp2_du1_op", T_OBJECT_EX, offsetof(MidpointVI, dp2_du1_op), 0, trep_internal_doc},
    {"_dq2_dk2_op", T_OBJECT_EX, offsetof(MidpointVI, dq2_dk2_op), 0, trep_internal_doc},
    {"_dl1_dk2_op", T_OBJECT_EX, offsetof(MidpointVI, dl1_dk2_op), 0, trep_internal_doc},
    {"_dp2_dk2_op", T_OBJECT_EX, offsetof(MidpointVI, dp2_dk2_op), 0, trep_internal_doc},
    
    {NULL}  /* Sentinel */
};

PyTypeObject MidpointVIType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /* ob_size */
    "_trep._MidpointVI",      /* tp_name */
    sizeof(MidpointVI),       /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)dealloc,       /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_compare */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
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

