/***************************************************************************
 *cr
 *cr            (C) Copyright 1995-2009 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: VMDThreads.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.52 $       $Date: 2009/05/08 17:38:19 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  code for spawning threads on various platforms.
 ***************************************************************************/

#ifndef VMD_THREADS_INC
#define VMD_THREADS_INC 1

/* POSIX Threads */
#if defined(__hpux) || defined(__irix) || defined(__linux) || defined(_CRAY) || defined(__osf__) || defined(_AIX) || defined(__APPLE__) || defined(__sun)
#if !defined(USEPOSIXTHREADS)
#define USEPOSIXTHREADS
#endif
#endif

#ifdef VMDTHREADS
#ifdef USEPOSIXTHREADS
#include <pthread.h>

typedef pthread_t        vmd_thread_t;
typedef pthread_mutex_t   vmd_mutex_t;
typedef pthread_cond_t     vmd_cond_t;

typedef struct vmd_barrier_struct {
  int padding1[8]; /* Padding bytes to avoid false sharing and cache aliasing */
  pthread_mutex_t lock;   /**< Mutex lock for the structure */
  int n_clients;          /**< Number of threads to wait for at barrier */
  int n_waiting;          /**< Number of currently waiting threads */
  int phase;              /**< Flag to separate waiters from fast workers */
  int sum;                /**< Sum of arguments passed to barrier wait */
  int result;             /**< Answer to be returned by barrier wait */
  pthread_cond_t wait_cv; /**< Clients wait on condition variable to proceed */
  int padding2[8]; /* Padding bytes to avoid false sharing and cache aliasing */
} vmd_barrier_t;

#endif



#if defined(_MSC_VER) || defined(MINGW)
#include <windows.h>
typedef HANDLE vmd_thread_t;
typedef CRITICAL_SECTION vmd_mutex_t;

#if 0 && (NTDDI_VERSION >= NTDDI_WS08 || _WIN32_WINNT > 0x0600) 
/* Use native condition variables only with Windows Server 2008 and newer... */
#define VMDUSEWIN2008CONDVARS 1
typedef  CONDITION_VARIABLE vmd_cond_t;
#else
/* Every version of Windows prior to Vista/WS2008 must emulate */
/* variables using manually resettable events or other schemes */ 

/* For higher performance, use interlocked memory operations   */
/* rather than locking/unlocking mutexes when manipulating     */
/* internal state.                                             */
#if 1
#define VMDUSEINTERLOCKEDATOMICOPS 1
#endif 
#define VMD_COND_SIGNAL    0
#define VMD_COND_BROADCAST 1
typedef struct {
  LONG waiters;     /**< XXX this _MUST_ be 32-bit aligned for correct */
                    /**< operation with the InterlockedXXX() APIs      */
  CRITICAL_SECTION waiters_lock;
  HANDLE events[2]; /**< Signal and broadcast event HANDLEs. */
} vmd_cond_t;
#endif

typedef HANDLE vmd_barrier_t; /**< Not implemented for Windows */
#endif
#endif


#ifndef VMDTHREADS
typedef int vmd_thread_t;
typedef int vmd_mutex_t;
typedef int vmd_cond_t;
typedef int vmd_barrier_t;
#endif


typedef struct vmd_run_barrier_struct {
  int padding1[8]; /* Padding bytes to avoid false sharing and cache aliasing */
  vmd_mutex_t lock;       /**< Mutex lock for the structure */
  int n_clients;          /**< Number of threads to wait for at barrier */
  int n_waiting;          /**< Number of currently waiting threads */
  int phase;              /**< Flag to separate waiters from fast workers */
  void * (*fctn)(void *); /**< Fctn ptr to call, or NULL if done */
  void * parms;           /**< parms for fctn pointer */
  void * (*rslt)(void *); /**< Fctn ptr to return to barrier wait callers */
  void * rsltparms;       /**< parms to return to barrier wait callers */
  vmd_cond_t wait_cv;     /**< Clients wait on condition variable to proceed */
  int padding2[8]; /* Padding bytes to avoid false sharing and cache aliasing */
} vmd_run_barrier_t;


#ifdef __cplusplus
extern "C" {
#endif

/* number of physical processors available */
int vmd_thread_numphysprocessors(void);

/* number of processors available, subject to user override */
int vmd_thread_numprocessors(void);

/* query CPU affinity of the VMD process (if allowed by host system) */
int * vmd_cpu_affinitylist(int *cpuaffinitycount);

/* set the CPU affinity of the current thread (if allowed by host system) */
int vmd_thread_set_self_cpuaffinity(int cpu);

/* set the concurrency level and scheduling scope for threads */
int vmd_thread_setconcurrency(int);


/*
 * Thread management
 */

/* create a thread */
int vmd_thread_create(vmd_thread_t *, void * fctn(void *), void *);

/* join (wait for completion of, and merge with) a thread */
int vmd_thread_join(vmd_thread_t, void **);


/*
 * Mutexes
 */

/* initialize a mutex */
int vmd_mutex_init(vmd_mutex_t *);

/* lock a mutex */
int vmd_mutex_lock(vmd_mutex_t *);

/* unlock a mutex */
int vmd_mutex_unlock(vmd_mutex_t *);

/* destroy a mutex */
int vmd_mutex_destroy(vmd_mutex_t *);


/*
 * Condition variables
 */
/* initialize a condition variable */
int vmd_cond_init(vmd_cond_t *);

/* destroy a condition variable */
int vmd_cond_destroy(vmd_cond_t *);

/* wait on a condition variable */
int vmd_cond_wait(vmd_cond_t *, vmd_mutex_t *);

/* signal a condition variable, waking at least one thread */
int vmd_cond_signal(vmd_cond_t *);

/* signal a condition variable, waking all threads */
int vmd_cond_broadcast(vmd_cond_t *);


/* 
 * This is an implementation of a symmetric summing barrier, 
 * that causes participating threads to sleep while waiting 
 * for their peers to reach the barrier.
 */
/* initialize a thread barrier (for use within a single process */
int vmd_thread_barrier_init(vmd_barrier_t *, int n_clients);

/* 
 * When rendering in the CAVE we use a special synchronization
 * mode so that shared memory mutexes and condition variables 
 * will work correctly when accessed from multiple processes.
 * Inter-process synchronization involves the kernel to a greater
 * degree, so these barriers are substantially more costly to use 
 * than the ones designed for use within a single-process.
 */
int vmd_thread_barrier_init_proc_shared(vmd_barrier_t *, int n_clients);

/* destroy a thread barrier */
void vmd_thread_barrier_destroy(vmd_barrier_t *barrier);

/* 
 * Synchronize on a thread barrier, returning the sum of all 
 * of the "increment" parameters from participating threads 
 */
int vmd_thread_barrier(vmd_barrier_t *barrier, int increment);


/*
 * This is a symmetric barrier routine designed to be used
 * in implementing a sleepable thread pool.
 */
int vmd_thread_run_barrier_init(vmd_run_barrier_t *barrier, int n_clients);
void vmd_thread_run_barrier_destroy(vmd_run_barrier_t *barrier);

/* sleeping barrier synchronization */
void * (*vmd_thread_run_barrier(vmd_run_barrier_t *barrier,
                                void * fctn(void*),
                                void * parms,
                                void **rsltparms))(void *);

/* non-blocking poll to see if peers are already at the barrier */
int vmd_thread_run_barrier_poll(vmd_run_barrier_t *barrier);


/*
 * Shared flag/integer for use in error code propagation from
 * worker threads
 */
typedef struct vmd_shared_flag_struct {
  vmd_mutex_t mtx;     /**< mutex lock */
  int flag;            /**< shared flag/integer */
} vmd_shared_flag_t;

/* initialize the shared flag */
int vmd_shared_flag_init(vmd_shared_flag_t *flg);

/* destroy the shared flag */
int vmd_shared_flag_destroy(vmd_shared_flag_t *flg);

/* set the shared flag */
int vmd_shared_flag_set(vmd_shared_flag_t *flg, int value);

/* get the shared flag */
int vmd_shared_flag_get(vmd_shared_flag_t *flg, int *value);


/* Task tile struct for stack, iterator, and scheduler routines;  */
/* 'start' is inclusive, 'end' is exclusive.  This yields a       */
/* half-open interval that corresponds to a typical 'for' loop.   */
typedef struct vmd_tasktile_struct {
  int start;           /**< starting task ID (inclusive) */
  int end;             /**< ending task ID (exclusive) */
} vmd_tasktile_t;


/* 
 * tile stack 
 */
#define TILESTACK_EMPTY -1

typedef struct vmd_tasktile_stack_struct {
  vmd_mutex_t mtx;   /**< Mutex lock for the structure */
  int growthrate;    /**< stack growth chunk size */
  int size;          /**< current allocated stack size */
  int top;           /**< index of top stack element */
  vmd_tasktile_t *s; /**< stack of task tiles */
} vmd_tilestack_t;

int vmd_tilestack_init(vmd_tilestack_t *s, int size); 
void vmd_tilestack_destroy(vmd_tilestack_t *);
int vmd_tilestack_compact(vmd_tilestack_t *);
int vmd_tilestack_push(vmd_tilestack_t *, const vmd_tasktile_t *);
int vmd_tilestack_pop(vmd_tilestack_t *, vmd_tasktile_t *);
int vmd_tilestack_popall(vmd_tilestack_t *);
int vmd_tilestack_empty(vmd_tilestack_t *);


/*
 * Shared iterators intended for trivial CPU/GPU load balancing with no
 * exception handling capability (all work units must complete with 
 * no errors, or else the whole thing is canceled).
 */

/* work scheduling macros */
#define VMD_SCHED_DONE     -1
#define VMD_SCHED_CONTINUE  0

typedef struct vmd_shared_iterator_struct {
  vmd_mutex_t mtx;     /**< mutex lock */
  int start;           /**< starting value (inclusive) */
  int end;             /**< ending value (exlusive) */
  int current;         /**< current value */
  int fatalerror;      /**< cancel processing immediately for all threads */
} vmd_shared_iterator_t;

/* initialize a shared iterator */
int vmd_shared_iterator_init(vmd_shared_iterator_t *it);

/* destroy a shared iterator */
int vmd_shared_iterator_destroy(vmd_shared_iterator_t *it);

/* Set shared iterator state to half-open interval defined by tile */
int vmd_shared_iterator_set(vmd_shared_iterator_t *it, vmd_tasktile_t *tile);

/* iterate the shared iterator with a requested tile size,        */
/* returns the tile received, and a return code of -1 if no       */
/* iterations left or a fatal error has occured during processing,*/
/* canceling all worker threads.                                  */
int vmd_shared_iterator_next_tile(vmd_shared_iterator_t *it, int reqsize, 
                                  vmd_tasktile_t *tile);

/* worker thread calls this to indicate a fatal error */
int vmd_shared_iterator_setfatalerror(vmd_shared_iterator_t *it);

/* master thread calls this to query for fatal errors */
int vmd_shared_iterator_getfatalerror(vmd_shared_iterator_t *it);


/*
 * Thread pool.
 */

/* shortcut macro to tell the create routine we only want CPU cores */
#define VMD_THREADPOOL_DEVLIST_CPUSONLY NULL

/* symbolic constant macro to test if we have a GPU or not */
#define VMD_THREADPOOL_DEVID_CPU -1

/** thread-specific handle data for workers */
typedef struct vmd_threadpool_workerdata_struct {
  int padding1[8]; /* Padding bytes to avoid false sharing and cache aliasing */
  vmd_shared_iterator_t *iter;           /**< dynamic work scheduler */
  vmd_tilestack_t *errorstack;           /**< stack of tiles that failed */
  int threadid;                          /**< worker thread's id */
  int threadcount;                       /**< total number of worker threads */
  int devid;                             /**< worker CPU/GPU device ID */
  float devspeed;                        /**< speed scaling for this device */
  void *parms;                           /**< fctn parms for this worker */
  void *thrpool;                         /**< void ptr to thread pool struct */
  int padding2[8]; /* Padding bytes to avoid false sharing and cache aliasing */
} vmd_threadpool_workerdata_t;

typedef struct vmd_threadpool_struct {
  int workercount;                         /**< number of worker threads */
  int *devlist;                            /**< per-worker CPU/GPU device IDs */
  vmd_shared_iterator_t iter;              /**< dynamic work scheduler */
  vmd_tilestack_t errorstack;              /**< stack of tiles that failed */
  vmd_thread_t *threads;                   /**< worker threads */
  vmd_threadpool_workerdata_t *workerdata; /**< per-worker data */
  vmd_run_barrier_t runbar;                /**< execution barrier */
} vmd_threadpool_t;

/* create a thread pool with a specified number of worker threads */
vmd_threadpool_t * vmd_threadpool_create(int workercount, int *devlist);

/* launch threads onto a new function, with associated parms */
int vmd_threadpool_launch(vmd_threadpool_t *thrpool, 
                          void *fctn(void *), void *parms, int blocking);

/* wait for all worker threads to complete their work */
int vmd_threadpool_wait(vmd_threadpool_t *thrpool);

/* join all worker threads and free resources */
int vmd_threadpool_destroy(vmd_threadpool_t *thrpool);

/* worker thread can call this to get its ID and number of peers */
int vmd_threadpool_worker_getid(void *voiddata, int *threadid, int *threadcount);

/* worker thread can call this to get its CPU/GPU device ID */
int vmd_threadpool_worker_getdevid(void *voiddata, int *devid);

/* worker thread calls this to set relative speed of this device */
/* as determined by the SM/core count and clock rate             */
/* Note: this should only be called once, during the worker's    */
/* device initialization process                                 */
int vmd_threadpool_worker_setdevspeed(void *voiddata, float speed);

/* worker thread calls this to get relative speed of this device */
/* as determined by the SM/core count and clock rate             */
int vmd_threadpool_worker_getdevspeed(void *voiddata, float *speed);

/* worker thread calls this to scale max tile size by worker speed */
/* as determined by the SM/core count and clock rate             */
int vmd_threadpool_worker_devscaletile(void *voiddata, int *tilesize);

/* worker thread can call this to get its client data pointer */
int vmd_threadpool_worker_getdata(void *voiddata, void **clientdata);

/* Set dynamic scheduler state to half-open interval defined by tile */
int vmd_threadpool_sched_dynamic(vmd_threadpool_t *thrpool, vmd_tasktile_t *tile);

/* worker thread calls this to get its next work unit            */
/* iterate the shared iterator, returns -1 if no iterations left */
int vmd_threadpool_next_tile(void *thrpool, int reqsize, 
                             vmd_tasktile_t *tile);

/* worker thread calls this when it fails computing a tile after */
/* it has already taken it from the scheduler                    */
int vmd_threadpool_tile_failed(void *thrpool, vmd_tasktile_t *tile);

/* worker thread calls this to indicate that an unrecoverable error occured */
int vmd_threadpool_setfatalerror(void *thrparms);

/* master thread calls this to query for fatal errors */
int vmd_threadpool_getfatalerror(void *thrparms);



/*
 * Routines to generate a pool of threads which then grind through
 * a dynamically load balanced work queue implemented as a shared iterator.
 * No exception handling is possible, just a simple all-or-nothing attept.
 * Useful for simple calculations that take very little time.
 * An array of threads is generated, launched, and joined all with one call.
 */
typedef struct vmd_threadlaunch_struct {
  int padding1[8]; /* Padding bytes to avoid false sharing and cache aliasing */
  vmd_shared_iterator_t *iter;  /**< dynamic scheduler iterator */
  int threadid;                 /**< ID of worker thread */
  int threadcount;              /**< number of workers */
  void * clientdata;            /**< worker parameters */
  int padding2[8]; /* Padding bytes to avoid false sharing and cache aliasing */
} vmd_threadlaunch_t;

/* launch up to numprocs threads using shared iterator as a load balancer */
int vmd_threadlaunch(int numprocs, void *clientdata, void * fctn(void *),
                     vmd_tasktile_t *tile);

/* worker thread can call this to get its ID and number of peers */
int vmd_threadlaunch_getid(void *thrparms, int *threadid, int *threadcount);

/* worker thread can call this to get its client data pointer */
int vmd_threadlaunch_getdata(void *thrparms, void **clientdata);

/* worker thread calls this to get its next work unit            */
/* iterate the shared iterator, returns -1 if no iterations left */
int vmd_threadlaunch_next_tile(void *voidparms, int reqsize, 
                               vmd_tasktile_t *tile);

/* worker thread calls this to indicate that an unrecoverable error occured */
int vmd_threadlaunch_setfatalerror(void *thrparms);


#ifdef __cplusplus
}
#endif

#endif
