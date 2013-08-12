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
 *      $RCSfile: VMDThreads.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.84 $       $Date: 2009/05/08 18:54:18 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * VMDThreads.C - code for spawning threads on various platforms.
 *                Code donated by John Stone, john.stone@gmail.com 
 *                This code was originally written for the
 *                Tachyon Parallel/Multiprocessor Ray Tracer. 
 *                Improvements have been donated by Mr. Stone on an 
 *                ongoing basis. 
 *
 ***************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* If compiling on Linux, enable the GNU CPU affinity functions in both */
/* libc and the libpthreads                                             */
#if defined(__linux)
#define _GNU_SOURCE 1
#endif

#include "VMDThreads.h"

#if defined(_MSC_VER) || defined(MINGW)
#include <windows.h>
#include <winbase.h>
#endif

/* needed for call to sysconf() */
#if defined(__sun) || defined(__linux) || defined(__irix) || defined(_CRAY) || defined(__osf__) || defined(_AIX)
#include<unistd.h>
#endif

#if defined(__APPLE__) && defined(VMDTHREADS)
#include <Carbon/Carbon.h> /* Carbon APIs for Multiprocessing */
#endif

#if defined(__hpux)
#include <sys/mpctl.h>
#endif


#ifdef __cplusplus
extern "C" {
#endif

int vmd_thread_numphysprocessors(void) {
  int a=1;

#ifdef VMDTHREADS
#if defined(__APPLE__) 
  a = MPProcessorsScheduled(); /* Number of active/running CPUs */
#endif

#if defined(_MSC_VER) || defined(MINGW)
  struct _SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  a = sysinfo.dwNumberOfProcessors; /* total number of CPUs */
#endif /* _MSC_VER */

#if defined(_CRAY)
  a = sysconf(_SC_CRAY_NCPU);
#endif

#if defined(__sun) || defined(__linux) || defined(__osf__) || defined(_AIX)
  a = sysconf(_SC_NPROCESSORS_ONLN); /* number of active/running CPUs */
#endif /* SunOS */

#if defined(__irix)
  a = sysconf(_SC_NPROC_ONLN); /* number of active/running CPUs */
#endif /* IRIX */

#if defined(__hpux)
  a = mpctl(MPC_GETNUMSPUS, 0, 0); /* total number of CPUs */
#endif /* HPUX */
#endif /* VMDTHREADS */

  return a;
}

int vmd_thread_numprocessors(void) {
  int a=1;

#ifdef VMDTHREADS
  /* Allow the user to override the number of CPUs for use */
  /* in scalability testing, debugging, etc.               */
  char *forcecount = getenv("VMDFORCECPUCOUNT");
  if (forcecount != NULL) {
    if (sscanf(forcecount, "%d", &a) == 1) {
      return a; /* if we got a valid count, return it */
    } else {
      a=1;      /* otherwise use the real available hardware CPU count */
    }
  }

  /* otherwise return the number of physical processors currently available */
  a = vmd_thread_numphysprocessors();

  /* XXX we should add checking for the current CPU affinity masks here, */
  /* and return the min of the physical processor count and CPU affinity */
  /* mask enabled CPU count.                                             */
#endif /* VMDTHREADS */

  return a;
}


int * vmd_cpu_affinitylist(int *cpuaffinitycount) {
  int *affinitylist = NULL;
  *cpuaffinitycount = -1; /* return count -1 if unimplemented or err occurs */

/* Win32 process affinity mask query */
/* XXX untested, but based on the linux code, may work with a few tweaks */
#if 0 && (defined(_MSC_VER) || defined(MINGW))
  HANDLE myproc = GetCurrentProcess(); /* returns a psuedo-handle */
  DWORD affinitymask, sysaffinitymask;

  if (!GetProcessAffinityMask(myproc, &affinitymask, &sysaffinitymask)) {
    /* count length of affinity list */
    int affinitycount=0;
    int i;
    for (i=0; i<31; i++) {
      affinitycount += (affinitymask >> i) & 0x1;
    }
  
    /* build affinity list */
    if (affinitycount > 0) {
      affinitylist = (int *) malloc(affinitycount * sizeof(int));
      if (affinitylist == NULL)
        return NULL;

      int curcount = 0;
      for (i=0; i<CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &affinitymask)) {
          affinitylist[curcount] = i;
          curcount++;
        }
      }
    }

    *cpuaffinitycount = affinitycount; /* return final affinity list */
  }
#endif

/* Linux process affinity mask query */
#if defined(__linux)

/* protect ourselves from some older Linux distros */
#if defined(CPU_SETSIZE)
  int i;
  cpu_set_t affinitymask;
  int affinitycount=0;

  /* PID 0 refers to the current process */
  if (sched_getaffinity(0, sizeof(affinitymask), &affinitymask) < 0) {
    perror("vmd_cpu_affinitylist: sched_getaffinity");
    return NULL;
  }

  /* count length of affinity list */
  for (i=0; i<CPU_SETSIZE; i++) {
    affinitycount += CPU_ISSET(i, &affinitymask);
  }

  /* build affinity list */
  if (affinitycount > 0) {
    affinitylist = (int *) malloc(affinitycount * sizeof(int));
    if (affinitylist == NULL)
      return NULL;

    int curcount = 0;
    for (i=0; i<CPU_SETSIZE; i++) {
      if (CPU_ISSET(i, &affinitymask)) {
        affinitylist[curcount] = i;
        curcount++;
      }
    }
  }

  *cpuaffinitycount = affinitycount; /* return final affinity list */
#endif
#endif

  /* MacOS X 10.5.x has a CPU affinity query/set capability finally      */
  /* http://developer.apple.com/releasenotes/Performance/RN-AffinityAPI/ */

  /* Solaris and HP-UX use pset_bind() and related functions, and they   */
  /* don't use the single-level mask-based scheduling mechanism that     */
  /* the others, use.  Instead, they use a hierarchical tree of          */
  /* processor sets and processes float within those, or are tied to one */
  /* processor that's a member of a particular set.                      */

  return affinitylist;
}


int vmd_thread_set_self_cpuaffinity(int cpu) {
  int status=-1; /* unsupported by default */

#ifdef VMDTHREADS

#if defined(__linux)
#if 0
  /* XXX this code is too new even for RHEL4, though it runs on Fedora 7 */
  /* and other newer revs.                                               */
  /* NPTL systems can assign per-thread affinities this way              */
  cpu_set_t affinitymask;
  CPU_ZERO(&affinitymask); 
  CPU_SET(cpu, &affinitymask);
  status = pthread_setaffinity_np(pthread_self(), sizeof(affinitymask), &affinitymask);
#else
  /* non-NPTL systems based on the clone() API must use this method      */
  cpu_set_t affinitymask;
  CPU_ZERO(&affinitymask); 
  CPU_SET(cpu, &affinitymask);

  /* PID 0 refers to the current process */
  if ((status=sched_setaffinity(0, sizeof(affinitymask), &affinitymask)) < 0) {
    perror("vmd_thread_set_self_cpuaffinitylist: sched_setaffinity");
    return status;
  }
#endif

  /* call sched_yield() so new affinity mask takes effect immediately */
  sched_yield();
#endif /* linux */

  /* MacOS X 10.5.x has a CPU affinity query/set capability finally      */
  /* http://developer.apple.com/releasenotes/Performance/RN-AffinityAPI/ */

  /* Solaris and HP-UX use pset_bind() and related functions, and they   */
  /* don't use the single-level mask-based scheduling mechanism that     */
  /* the others, use.  Instead, they use a hierarchical tree of          */
  /* processor sets and processes float within those, or are tied to one */
  /* processor that's a member of a particular set.                      */
#endif

  return status;
}


int vmd_thread_setconcurrency(int nthr) {
  int status=0;

#ifdef VMDTHREADS
#if defined(__sun) 
#ifdef USEPOSIXTHREADS 
  status = pthread_setconcurrency(nthr);
#else
  status = thr_setconcurrency(nthr);
#endif
#endif /* SunOS */

#if defined(__irix) || defined(_AIX)
  status = pthread_setconcurrency(nthr);
#endif
#endif /* VMDTHREADS */

  return status;
}



/* Typedef to eliminate compiler warning caused by C/C++ linkage conflict. */
#ifdef __cplusplus
extern "C" {
#endif
  typedef void * (*VMDTHREAD_START_ROUTINE)(void *);
#ifdef __cplusplus
}
#endif

int vmd_thread_create(vmd_thread_t * thr, void * fctn(void *), void * arg) {
  int status=0;

#ifdef VMDTHREADS 
#if defined(_MSC_VER) || defined(MINGW)
  DWORD tid; /* thread id, msvc only */
  *thr = CreateThread(NULL, 8192, (LPTHREAD_START_ROUTINE) fctn, arg, 0, &tid);
  if (*thr == NULL) {
    status = -1;
  }
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS 
#if defined(_AIX)
  {
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
    status = pthread_create(thr, &attr, fctn, arg);
    pthread_attr_destroy(&attr);
  }
#else   
  status = pthread_create(thr, NULL, (VMDTHREAD_START_ROUTINE)fctn, arg);
#endif 
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */
 
  return status;
}


int vmd_thread_join(vmd_thread_t thr, void ** stat) {
  int status=0;  

#ifdef VMDTHREADS
#if defined(_MSC_VER) || defined(MINGW)
  DWORD wstatus = 0;
 
  wstatus = WAIT_TIMEOUT;
 
  while (wstatus != WAIT_OBJECT_0) {
    wstatus = WaitForSingleObject(thr, INFINITE);
  }
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_join(thr, stat);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}  


int vmd_mutex_init(vmd_mutex_t * mp) {
  int status=0;

#ifdef VMDTHREADS
#if defined(_MSC_VER) || defined(MINGW)
  InitializeCriticalSection(mp);
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_mutex_init(mp, 0);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}


int vmd_mutex_lock(vmd_mutex_t * mp) {
  int status=0;

#ifdef VMDTHREADS
#if defined(_MSC_VER) || defined(MINGW)
  EnterCriticalSection(mp);
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_mutex_lock(mp);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}


int vmd_mutex_unlock(vmd_mutex_t * mp) {
  int status=0;

#ifdef VMDTHREADS  
#if defined(_MSC_VER) || defined(MINGW)
  LeaveCriticalSection(mp);
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_mutex_unlock(mp);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}


int vmd_mutex_destroy(vmd_mutex_t * mp) {
  int status=0;

#ifdef VMDTHREADS
#if defined(_MSC_VER) || defined(MINGW)
  DeleteCriticalSection(mp);
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_mutex_destroy(mp);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}



int vmd_cond_init(vmd_cond_t * cvp) {
  int status=0;

#ifdef VMDTHREADS
#if defined(_MSC_VER) || defined(MINGW)
#if defined(VMDUSEWIN2008CONDVARS)
  InitializeConditionVariable(cvp);
#else
  /* XXX not implemented */
  cvp->waiters = 0;

  /* Create an auto-reset event. */
  cvp->events[VMD_COND_SIGNAL] = CreateEvent(NULL,  /* no security */
                                             FALSE, /* auto-reset event */
                                             FALSE, /* non-signaled initially */
                                             NULL); /* unnamed */

  // Create a manual-reset event.
  cvp->events[VMD_COND_BROADCAST] = CreateEvent(NULL,  /* no security */
                                                TRUE,  /* manual-reset */
                                                FALSE, /* non-signaled initially*/
                                                NULL); /* unnamed */
#endif
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_cond_init(cvp, NULL);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}

int vmd_cond_destroy(vmd_cond_t * cvp) {
  int status=0;

#ifdef VMDTHREADS
#if defined(_MSC_VER) || defined(MINGW)
#if defined(VMDUSEWIN2008CONDVARS)
  /* XXX not implemented */
#else
  CloseHandle(cvp->events[VMD_COND_SIGNAL]);
  CloseHandle(cvp->events[VMD_COND_BROADCAST]);
#endif
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_cond_destroy(cvp);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}

int vmd_cond_wait(vmd_cond_t * cvp, vmd_mutex_t * mp) {
  int status=0;

#ifdef VMDTHREADS
#if defined(_MSC_VER) || defined(MINGW)
#if defined(VMDUSEWIN2008CONDVARS)
  SleepConditionVariableCS(cvp, mp, INFINITE)
#else
#if !defined(VMDUSEINTERLOCKEDATOMICOPS)
  EnterCriticalSection(&cvp->waiters_lock);
  cvp->waiters++;
  LeaveCriticalSection(&cvp->waiters_lock);
#else
  InterlockedIncrement(&cvp->waiters);
#endif

  LeaveCriticalSection(mp); /* SetEvent() maintains state, avoids lost wakeup */

  /* Wait either a single or broadcast even to become signalled */
  int result = WaitForMultipleObjects(2, cvp->events, FALSE, INFINITE);

#if !defined(VMDUSEINTERLOCKEDATOMICOPS)
  EnterCriticalSection (&cvp->waiters_lock);
  cvp->waiters--;
  LONG last_waiter = 
    ((result == (WAIT_OBJECT_0 + VMD_COND_BROADCAST)) && cvp->waiters == 0);
  LeaveCriticalSection (&cvp->waiters_lock);
#else
  LONG my_waiter = InterlockedDecrement(&cvp->waiters);
  LONG last_waiter = 
    ((result == (WAIT_OBJECT_0 + VMD_COND_BROADCAST)) && my_waiter == 0);
#endif

  /* Some thread called cond_broadcast() */
  if (last_waiter)
    /* We're the last waiter to be notified or to stop waiting, so */
    /* reset the manual event.                                     */
    ResetEvent(cvp->events[VMD_COND_BROADCAST]); 

  EnterCriticalSection(mp);
#endif
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_cond_wait(cvp, mp);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}

int vmd_cond_signal(vmd_cond_t * cvp) {
  int status=0;

#ifdef VMDTHREADS
#if defined(_MSC_VER) || defined(MINGW)
#if defined(VMDUSEWIN2008CONDVARS)
  WakeConditionVariable(cvp);
#else
#if !defined(VMDUSEINTERLOCKEDATOMICOPS)
  EnterCriticalSection(&cvp->waiters_lock);
  int have_waiters = (cvp->waiters > 0);
  LeaveCriticalSection(&cvp->waiters_lock);
  if (have_waiters)
    SetEvent (cvp->events[VMD_COND_SIGNAL]);
#else
  if (InterlockedExchangeAdd(&cvp->waiters, 0) > 0)
    SetEvent(cvp->events[VMD_COND_SIGNAL]);
#endif
#endif
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_cond_signal(cvp);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}

int vmd_cond_broadcast(vmd_cond_t * cvp) {
  int status=0;

#ifdef VMDTHREADS
#if defined(_MSC_VER) || defined(MINGW)
#if defined(VMDUSEWIN2008CONDVARS)
  WakeAllConditionVariable(cvp);
#else
#if !defined(VMDUSEINTERLOCKEDATOMICOPS)
  EnterCriticalSection(&cvp->waiters_lock);
  int have_waiters = (cvp->waiters > 0);
  LeaveCriticalSection(&cvp->waiters_lock);
  if (have_waiters)
    SetEvent(cvp->events[VMD_COND_BROADCAST]);
#else
  if (InterlockedExchangeAdd(&cvp->waiters, 0) > 0)
    SetEvent(cvp->events[VMD_COND_BROADCAST]);
#endif

#endif
#endif /* _MSC_VER */

#ifdef USEPOSIXTHREADS
  status = pthread_cond_broadcast(cvp);
#endif /* USEPOSIXTHREADS */
#endif /* VMDTHREADS */

  return status;
}
  


#if !defined(VMDTHREADS)

int vmd_thread_barrier_init(vmd_barrier_t *barrier, int n_clients) {
  return 0;
}

int vmd_thread_barrier_init_proc_shared(vmd_barrier_t *barrier, int n_clients) {
  return 0;
}

void vmd_thread_barrier_destroy(vmd_barrier_t *barrier) {
}

int vmd_thread_barrier(vmd_barrier_t *barrier, int increment) {
  return 0;
}

#else 

#ifdef USEPOSIXTHREADS
/* symmetric summing barrier for use within a single process */
int vmd_thread_barrier_init(vmd_barrier_t *barrier, int n_clients) {
  if (barrier != NULL) {
    barrier->n_clients = n_clients;
    barrier->n_waiting = 0;
    barrier->phase = 0;
    barrier->sum = 0;

    pthread_mutex_init(&barrier->lock, NULL);
    pthread_cond_init(&barrier->wait_cv, NULL);
  }

  return 0;
}


/* When rendering in the CAVE we use a special synchronization    */
/* mode so that shared memory mutexes and condition variables     */
/* will work correctly when accessed from multiple processes.     */
/* Inter-process synchronization involves the kernel to a greater */
/* degree, so these barriers are substantially more costly to use */
/* than the ones designed for use within a single-process.        */
int vmd_thread_barrier_init_proc_shared(vmd_barrier_t *barrier, int n_clients) {
  if (barrier != NULL) {
    barrier->n_clients = n_clients;
    barrier->n_waiting = 0;
    barrier->phase = 0;
    barrier->sum = 0;

    pthread_mutexattr_t mattr;
    pthread_condattr_t  cattr;

    printf("Setting barriers to have system scope...\n");

    pthread_mutexattr_init(&mattr);
    if (pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED) != 0) {
      printf("WARNING: could not set mutex to process shared scope\n");
    }

    pthread_condattr_init(&cattr);
    if (pthread_condattr_setpshared(&cattr, PTHREAD_PROCESS_SHARED) != 0) {
      printf("WARNING: could not set mutex to process shared scope\n");
    }

    pthread_mutex_init(&barrier->lock, &mattr);
    pthread_cond_init(&barrier->wait_cv, &cattr);

    pthread_condattr_destroy(&cattr);
    pthread_mutexattr_destroy(&mattr);
  }

  return 0;
}

void vmd_thread_barrier_destroy(vmd_barrier_t *barrier) {
  pthread_mutex_destroy(&barrier->lock);
  pthread_cond_destroy(&barrier->wait_cv);
}

int vmd_thread_barrier(vmd_barrier_t *barrier, int increment) {
  int my_phase;
  int my_result;

  pthread_mutex_lock(&barrier->lock);
  my_phase = barrier->phase;
  barrier->sum += increment;
  barrier->n_waiting++;

  if (barrier->n_waiting == barrier->n_clients) {
    barrier->result = barrier->sum;
    barrier->sum = 0;
    barrier->n_waiting = 0;
    barrier->phase = 1 - my_phase;
    pthread_cond_broadcast(&barrier->wait_cv);
  }

  while (barrier->phase == my_phase) {
    pthread_cond_wait(&barrier->wait_cv, &barrier->lock);
  }

  my_result = barrier->result;

  pthread_mutex_unlock(&barrier->lock);

  return my_result;
}

#endif

#endif /* VMDTHREADS */


/* symmetric run barrier for use within a single process */
int vmd_thread_run_barrier_init(vmd_run_barrier_t *barrier, int n_clients) {
#ifdef VMDTHREADS
  if (barrier != NULL) {
    barrier->n_clients = n_clients;
    barrier->n_waiting = 0;
    barrier->phase = 0;
    barrier->fctn = NULL;

    vmd_mutex_init(&barrier->lock);
    vmd_cond_init(&barrier->wait_cv);
  }
#endif

  return 0;
}

void vmd_thread_run_barrier_destroy(vmd_run_barrier_t *barrier) {
#ifdef VMDTHREADS
  vmd_mutex_destroy(&barrier->lock);
  vmd_cond_destroy(&barrier->wait_cv);
#endif
}

/*
 * Wait until all threads reach barrier, and return the function 
 * pointer passed in by the master thread.
 */
void * (*vmd_thread_run_barrier(vmd_run_barrier_t *barrier, 
                                void * fctn(void*),
                                void * parms,
                                void **rsltparms))(void *) {
#if defined(VMDTHREADS)
  int my_phase;
  void * (*my_result)(void*);

  vmd_mutex_lock(&barrier->lock);
  my_phase = barrier->phase;
  if (fctn != NULL)
    barrier->fctn = fctn;
  if (parms != NULL)
    barrier->parms = parms;
  barrier->n_waiting++;

  if (barrier->n_waiting == barrier->n_clients) {
    barrier->rslt = barrier->fctn;
    barrier->rsltparms = barrier->parms;
    barrier->fctn = NULL;
    barrier->parms = NULL;
    barrier->n_waiting = 0;
    barrier->phase = 1 - my_phase;
    vmd_cond_broadcast(&barrier->wait_cv);
  }

  while (barrier->phase == my_phase) {
    vmd_cond_wait(&barrier->wait_cv, &barrier->lock);
  }

  my_result = barrier->rslt;
  if (rsltparms != NULL) 
    *rsltparms = barrier->rsltparms;

  vmd_mutex_unlock(&barrier->lock);
#else
  void * (*my_result)(void*) = fctn;
  if (rsltparms != NULL) 
    *rsltparms = parms;
#endif

  return my_result;
}

/* non-blocking poll to see if peers are already at the barrier */
int vmd_thread_run_barrier_poll(vmd_run_barrier_t *barrier) {
  int rc=0;
#if defined(VMDTHREADS)
  vmd_mutex_lock(&barrier->lock);
  if (barrier->n_waiting == (barrier->n_clients-1)) {
    rc=1;
  }
  vmd_mutex_unlock(&barrier->lock);
#endif
  return rc;
}



#if defined(VMDTHREADS)
/*
 * Thread pool.
 */
static void * vmd_threadpool_workerproc(void *voidparms) {
  void *(*fctn)(void*);
  vmd_threadpool_workerdata_t *workerdata = (vmd_threadpool_workerdata_t *) voidparms;
  vmd_threadpool_t *thrpool = (vmd_threadpool_t *) workerdata->thrpool;

  while ((fctn = vmd_thread_run_barrier(&thrpool->runbar, NULL, NULL, &workerdata->parms)) != NULL) {
    (*fctn)(workerdata);
  }

  return NULL; 
}
static void * vmd_threadpool_workersync(void *voidparms) {
  return NULL; 
}
#endif

vmd_threadpool_t * vmd_threadpool_create(int workercount, int *devlist) {
  int i;
  vmd_threadpool_t *thrpool = NULL;
  thrpool = (vmd_threadpool_t *) malloc(sizeof(vmd_threadpool_t));
  if (thrpool == NULL)
    return NULL;

  memset(thrpool, 0, sizeof(vmd_threadpool_t));

#if !defined(VMDTHREADS)
  workercount=1;
#endif

  /* if caller provides a device list, use it, otherwise we assume */
  /* all workers are CPU cores */
  thrpool->devlist = (int *) malloc(sizeof(int) * workercount);
  if (devlist == NULL) {
    for (i=0; i<workercount; i++)
      thrpool->devlist[i] = -1; /* mark as a CPU core */
  } else {
    memcpy(thrpool->devlist, devlist, sizeof(int) * workercount);
  }

  /* initialize shared iterator */
  vmd_shared_iterator_init(&thrpool->iter);

  /* initialize tile stack for error handling */
  vmd_tilestack_init(&thrpool->errorstack, 64);

  /* create a run barrier with N+1 threads: N workers, 1 master */
  thrpool->workercount = workercount;
  vmd_thread_run_barrier_init(&thrpool->runbar, workercount+1);

  /* allocate and initialize thread pool */
  thrpool->threads = (vmd_thread_t *) malloc(sizeof(vmd_thread_t) * workercount);
  thrpool->workerdata = (vmd_threadpool_workerdata_t *) malloc(sizeof(vmd_threadpool_workerdata_t) * workercount);
  memset(thrpool->workerdata, 0, sizeof(vmd_threadpool_workerdata_t) * workercount);

  /* setup per-worker data */
  for (i=0; i<workercount; i++) {
    thrpool->workerdata[i].iter=&thrpool->iter;
    thrpool->workerdata[i].errorstack=&thrpool->errorstack;
    thrpool->workerdata[i].threadid=i;
    thrpool->workerdata[i].threadcount=workercount;
    thrpool->workerdata[i].devid=thrpool->devlist[i];
    thrpool->workerdata[i].devspeed=1.0f; /* must be reset by dev setup code */
    thrpool->workerdata[i].thrpool=thrpool;
  }

#if defined(VMDTHREADS)
  /* launch thread pool */
  for (i=0; i<workercount; i++) {
    vmd_thread_create(&thrpool->threads[i], vmd_threadpool_workerproc, &thrpool->workerdata[i]);
  }
#endif

  return thrpool;
}


int vmd_threadpool_launch(vmd_threadpool_t *thrpool,
                          void *fctn(void *), void *parms, int blocking) {
  if (thrpool == NULL)
    return -1;

#if defined(VMDTHREADS)
  /* wake sleeping threads to run fctn(parms) */
  vmd_thread_run_barrier(&thrpool->runbar, fctn, parms, NULL);
  if (blocking)
    vmd_thread_run_barrier(&thrpool->runbar, vmd_threadpool_workersync, NULL, NULL);
#else
  thrpool->workerdata[0].parms = parms;
  (*fctn)(&thrpool->workerdata[0]);
#endif
  return 0;
}


int vmd_threadpool_wait(vmd_threadpool_t *thrpool) {
#if defined(VMDTHREADS)
  vmd_thread_run_barrier(&thrpool->runbar, vmd_threadpool_workersync, NULL, NULL);
#endif
  return 0;
}

 
int vmd_threadpool_poll(vmd_threadpool_t *thrpool) {
#if defined(VMDTHREADS)
  return vmd_thread_run_barrier_poll(&thrpool->runbar);
#else
  return 1;
#endif
}


int vmd_threadpool_destroy(vmd_threadpool_t *thrpool) {
#if defined(VMDTHREADS)
  int i;
#endif

  /* wake threads and tell them to shutdown */
  vmd_thread_run_barrier(&thrpool->runbar, NULL, NULL, NULL);

#if defined(VMDTHREADS)
  /* join the pool of worker threads */
  for (i=0; i<thrpool->workercount; i++) {
    vmd_thread_join(thrpool->threads[i], NULL);
  }
#endif

  /* destroy the thread barrier */
  vmd_thread_run_barrier_destroy(&thrpool->runbar);

  /* destroy the shared iterator */
  vmd_shared_iterator_destroy(&thrpool->iter);

  /* destroy tile stack for error handling */
  vmd_tilestack_destroy(&thrpool->errorstack);

  free(thrpool->devlist);
  free(thrpool->threads);
  free(thrpool->workerdata);
  free(thrpool);

  return 0;
}


/* worker thread can call this to get its ID and number of peers */
int vmd_threadpool_worker_getid(void *voiddata, int *threadid, int *threadcount) {
  vmd_threadpool_workerdata_t *worker = (vmd_threadpool_workerdata_t *) voiddata;
  if (threadid != NULL)
    *threadid = worker->threadid;

  if (threadcount != NULL)
    *threadcount = worker->threadcount;

  return 0;
}


/* worker thread can call this to get its CPU/GPU device ID */
int vmd_threadpool_worker_getdevid(void *voiddata, int *devid) {
  vmd_threadpool_workerdata_t *worker = (vmd_threadpool_workerdata_t *) voiddata;
  if (devid != NULL)
    *devid = worker->devid;

  return 0;
}


/* worker thread calls this to set relative speed of this device */
/* as determined by the SM/core count and clock rate             */
/* Note: this should only be called once, during the worker's    */
/* device initialization process                                 */
int vmd_threadpool_worker_setdevspeed(void *voiddata, float speed) {
  vmd_threadpool_workerdata_t *worker = (vmd_threadpool_workerdata_t *) voiddata;
  worker->devspeed = speed; 
  return 0;
}


/* worker thread calls this to get relative speed of this device */
/* as determined by the SM/core count and clock rate             */
int vmd_threadpool_worker_getdevspeed(void *voiddata, float *speed) {
  vmd_threadpool_workerdata_t *worker = (vmd_threadpool_workerdata_t *) voiddata;
  if (speed != NULL)
    *speed = worker->devspeed; 
  return 0;
}


/* worker thread calls this to scale max tile size by worker speed */
/* as determined by the SM/core count and clock rate               */
int vmd_threadpool_worker_devscaletile(void *voiddata, int *tilesize) {
  vmd_threadpool_workerdata_t *worker = (vmd_threadpool_workerdata_t *) voiddata;
  if (tilesize != NULL) {
    int scaledtilesize;
    scaledtilesize = (int) (worker->devspeed * ((float) (*tilesize))); 
    if (scaledtilesize < 1)
      scaledtilesize = 1; 

    *tilesize = scaledtilesize;
  }

  return 0;
}


/* worker thread can call this to get its client data pointer */
int vmd_threadpool_worker_getdata(void *voiddata, void **clientdata) {
  vmd_threadpool_workerdata_t *worker = (vmd_threadpool_workerdata_t *) voiddata;
  if (clientdata != NULL)
    *clientdata = worker->parms;

  return 0;
}


/* Set shared iterator state to half-open interval defined by tile */
int vmd_threadpool_sched_dynamic(vmd_threadpool_t *thrpool, vmd_tasktile_t *tile) {
  if (thrpool == NULL)
    return -1;
  return vmd_shared_iterator_set(&thrpool->iter, tile);
}


/* iterate the shared iterator over the requested half-open interval */
int vmd_threadpool_next_tile(void *voidparms, int reqsize,
                             vmd_tasktile_t *tile) {
  int rc;
  vmd_threadpool_workerdata_t *worker = (vmd_threadpool_workerdata_t *) voidparms;
  rc = vmd_shared_iterator_next_tile(worker->iter, reqsize, tile);
  if (rc == VMD_SCHED_DONE) {
    /* if the error stack is empty, then we're done, otherwise pop */
    /* a tile off of the error stack and retry it                  */
    if (vmd_tilestack_pop(worker->errorstack, tile) != TILESTACK_EMPTY)
      return VMD_SCHED_CONTINUE;
  }
   
  return rc;
}


/* worker thread calls this when a failure occurs on a tile it has */
/* already taken from the scheduler                                */
int vmd_threadpool_tile_failed(void *voidparms, vmd_tasktile_t *tile) {
  vmd_threadpool_workerdata_t *worker = (vmd_threadpool_workerdata_t *) voidparms;
  return vmd_tilestack_push(worker->errorstack, tile);
}


/* worker thread calls this to indicate that an unrecoverable error occured */
int vmd_threadpool_setfatalerror(void *voidparms) {
  vmd_threadpool_workerdata_t *worker = (vmd_threadpool_workerdata_t *) voidparms;
  vmd_shared_iterator_setfatalerror(worker->iter);
  return 0;
}


/* worker thread calls this to indicate that an unrecoverable error occured */
int vmd_threadpool_getfatalerror(void *voidparms) {
  vmd_threadpool_workerdata_t *worker = (vmd_threadpool_workerdata_t *) voidparms;
  /* query error status for return to caller */
  return vmd_shared_iterator_getfatalerror(worker->iter);
}



/*
 * shared flags
 */

/* initialize the shared flag */
int vmd_shared_flag_init(vmd_shared_flag_t *flg) {
#if defined(VMDTHREADS)
  vmd_mutex_init(&flg->mtx);
#endif
  return 0;
}

/* destroy the shared flag */
int vmd_shared_flag_destroy(vmd_shared_flag_t *flg) {
#if defined(VMDTHREADS)
  vmd_mutex_destroy(&flg->mtx);
#endif
  return 0;
}
  
/* set the shared flag */
int vmd_shared_flag_set(vmd_shared_flag_t *flg, int value) {
#if defined(VMDTHREADS)
  vmd_mutex_lock(&flg->mtx);
#endif
  flg->flag = value;
#if defined(VMDTHREADS)
  vmd_mutex_unlock(&flg->mtx);
#endif
  return 0;
}

/* get the shared flag */
int vmd_shared_flag_get(vmd_shared_flag_t *flg, int *value) {
  if (value != NULL) {
#if defined(VMDTHREADS)
    vmd_mutex_lock(&flg->mtx);
#endif
    *value = flg->flag;
#if defined(VMDTHREADS)
    vmd_mutex_unlock(&flg->mtx);
#endif
  }
  return 0;
}


/*
 * task tile stack
 */
int vmd_tilestack_init(vmd_tilestack_t *s, int size) {
  if (s == NULL)
    return -1;

#if defined(VMDTHREADS)
  vmd_mutex_init(&s->mtx);
#endif

  s->growthrate = 512;
  s->top = -1;

  if (size > 0) {
    s->size = size;
    s->s = (vmd_tasktile_t *) malloc(s->size * sizeof(vmd_tasktile_t));
  } else {
    s->size = 0;
    s->s = NULL;
  }

  return 0;
}


void vmd_tilestack_destroy(vmd_tilestack_t *s) {
#if defined(VMDTHREADS)
  vmd_mutex_destroy(&s->mtx);
#endif
  free(s->s);
  s->s = NULL; /* prevent access after free */
}


int vmd_tilestack_compact(vmd_tilestack_t *s) {
#if defined(VMDTHREADS)
  vmd_mutex_lock(&s->mtx);
#endif
  if (s->size > (s->top + 1)) {
    int newsize = s->top + 1;
    vmd_tasktile_t *tmp = (vmd_tasktile_t *) realloc(s->s, newsize * sizeof(vmd_tasktile_t));
    if (tmp == NULL) {
#if defined(VMDTHREADS)
      vmd_mutex_unlock(&s->mtx);
#endif
      return -1; /* out of space! */
    }
    s->s = tmp;
    s->size = newsize;
  }
#if defined(VMDTHREADS)
  vmd_mutex_unlock(&s->mtx);
#endif

  return 0;
}


int vmd_tilestack_push(vmd_tilestack_t *s, const vmd_tasktile_t *t) {
#if defined(VMDTHREADS)
  vmd_mutex_lock(&s->mtx);
#endif
  s->top++;
  if (s->top >= s->size) {
    int newsize = s->size + s->growthrate;
    vmd_tasktile_t *tmp = (vmd_tasktile_t *) realloc(s->s, newsize * sizeof(vmd_tasktile_t));
    if (tmp == NULL) {
      s->top--;
#if defined(VMDTHREADS)
      vmd_mutex_unlock(&s->mtx);
#endif
      return -1; /* out of space! */
    }
    s->s = tmp;
    s->size = newsize;
  }

  s->s[s->top] = *t; /* push onto the stack */

#if defined(VMDTHREADS)
  vmd_mutex_unlock(&s->mtx);
#endif

  return 0;
}


int vmd_tilestack_pop(vmd_tilestack_t *s, vmd_tasktile_t *t) {
#if defined(VMDTHREADS)
  vmd_mutex_lock(&s->mtx);
#endif

  if (s->top < 0) {
#if defined(VMDTHREADS)
    vmd_mutex_unlock(&s->mtx);
#endif
    return -1; /* empty stack */
  }

  *t = s->s[s->top];
  s->top--;

#if defined(VMDTHREADS)
  vmd_mutex_unlock(&s->mtx);
#endif

  return 0;
}


int vmd_tilestack_popall(vmd_tilestack_t *s) {
#if defined(VMDTHREADS)
  vmd_mutex_lock(&s->mtx);
#endif

  s->top = -1;

#if defined(VMDTHREADS)
  vmd_mutex_unlock(&s->mtx);
#endif

  return 0;
}


int vmd_tilestack_empty(vmd_tilestack_t *s) {
#if defined(VMDTHREADS)
  vmd_mutex_lock(&s->mtx);
#endif

  if (s->top < 0) {
#if defined(VMDTHREADS)
    vmd_mutex_unlock(&s->mtx);
#endif
    return 1;
  }

#if defined(VMDTHREADS)
  vmd_mutex_unlock(&s->mtx);
#endif

  return 0;
}


/*
 * shared iterators
 */

/* initialize a shared iterator */
int vmd_shared_iterator_init(vmd_shared_iterator_t *it) {
  memset(it, 0, sizeof(vmd_shared_iterator_t));
#if defined(VMDTHREADS)
  vmd_mutex_init(&it->mtx);
#endif
  return 0;
}

/* destroy a shared iterator */
int vmd_shared_iterator_destroy(vmd_shared_iterator_t *it) {
#if defined(VMDTHREADS)
  vmd_mutex_destroy(&it->mtx);
#endif
  return 0;
}

/* set shared iterator parameters */
int vmd_shared_iterator_set(vmd_shared_iterator_t *it, 
                            vmd_tasktile_t *tile) {
#if defined(VMDTHREADS)
  vmd_mutex_lock(&it->mtx);
#endif
  it->start = tile->start;
  it->current = tile->start;
  it->end = tile->end;
  it->fatalerror = 0;
#if defined(VMDTHREADS)
  vmd_mutex_unlock(&it->mtx);
#endif
  return 0;
}


/* iterate the shared iterator, over a requested half-open interval */
int vmd_shared_iterator_next_tile(vmd_shared_iterator_t *it, int reqsize, 
                                  vmd_tasktile_t *tile) {
  int rc=VMD_SCHED_CONTINUE;

#if defined(VMDTHREADS)
  vmd_mutex_lock(&it->mtx);
#endif
  if (!it->fatalerror) {
    tile->start=it->current; /* set start to the current work unit    */
    it->current+=reqsize;    /* increment by the requested tile size  */
    tile->end=it->current;   /* set the (exclusive) endpoint          */

    /* if start is beyond the last work unit, we're done */
    if (tile->start >= it->end) {
      tile->start=0;
      tile->end=0;
      rc = VMD_SCHED_DONE;
    }

    /* if the endpoint (exclusive) for the requested tile size */
    /* is beyond the last work unit, roll it back as needed     */
    if (tile->end > it->end) {
      tile->end = it->end; 
    }
  } else {
    rc = VMD_SCHED_DONE;
  }
#if defined(VMDTHREADS)
  vmd_mutex_unlock(&it->mtx);
#endif

  return rc;
}


/* worker thread calls this to indicate a fatal error */
int vmd_shared_iterator_setfatalerror(vmd_shared_iterator_t *it) {
#if defined(VMDTHREADS)
  vmd_mutex_lock(&it->mtx);
#endif
  it->fatalerror=1; 
#if defined(VMDTHREADS)
  vmd_mutex_unlock(&it->mtx);
#endif
  return 0;
}

/* master thread calls this to query for fatal errors */
int vmd_shared_iterator_getfatalerror(vmd_shared_iterator_t *it) {
  int rc=0;
#if defined(VMDTHREADS)
  vmd_mutex_lock(&it->mtx);
#endif
  if (it->fatalerror)
    rc = -1;
#if defined(VMDTHREADS)
  vmd_mutex_unlock(&it->mtx);
#endif
  return rc;
}



/* launch up to numprocs threads using shared iterator as a load balancer */
int vmd_threadlaunch(int numprocs, void *clientdata, void * fctn(void *),
                     vmd_tasktile_t *tile) {
  vmd_shared_iterator_t iter;
  vmd_threadlaunch_t *parms=NULL;
  vmd_thread_t * threads=NULL;
  int i, rc;

  /* XXX have to ponder what the right thing to do is here */
#if !defined(VMDTHREADS)
  numprocs=1;
#endif

  /* initialize shared iterator and set the iteration and range */
  vmd_shared_iterator_init(&iter);
  if (vmd_shared_iterator_set(&iter, tile))
    return -1;

  /* allocate array of threads */
  threads = (vmd_thread_t *) calloc(numprocs * sizeof(vmd_thread_t), 1);
  if (threads == NULL)
    return -1;

  /* allocate and initialize array of thread parameters */
  parms = (vmd_threadlaunch_t *) malloc(numprocs * sizeof(vmd_threadlaunch_t));
  if (parms == NULL) {
    free(threads);
    return -1;
  }
  for (i=0; i<numprocs; i++) {
    parms[i].iter = &iter;
    parms[i].threadid = i;
    parms[i].threadcount = numprocs;
    parms[i].clientdata = clientdata;
  }

#if defined(VMDTHREADS)
  if (numprocs == 1) {
    /* XXX we special-case the single worker thread  */
    /*     scenario because this greatly reduces the */
    /*     GPU kernel launch overhead since a new    */
    /*     contexts doesn't have to be created, and  */
    /*     in the simplest case with a single-GPU we */
    /*     will just be using the same device anyway */
    /*     Ideally we shouldn't need to do this....  */
    /* single thread does all of the work */
    fctn((void *) &parms[0]);
  } else {
    /* spawn child threads to do the work */
    for (i=0; i<numprocs; i++) {
      vmd_thread_create(&threads[i], fctn, &parms[i]);
    }

    /* join the threads after work is done */
    for (i=0; i<numprocs; i++) {
      vmd_thread_join(threads[i], NULL);
    }
  }
#else
  /* single thread does all of the work */
  fctn((void *) &parms[0]);
#endif

  /* free threads/parms */
  free(parms);
  free(threads);

  /* query error status for return to caller */
  rc=vmd_shared_iterator_getfatalerror(&iter);

  /* destroy the shared iterator */
  vmd_shared_iterator_destroy(&iter);

  return rc;
}


/* worker thread can call this to get its ID and number of peers */
int vmd_threadlaunch_getid(void *voidparms, int *threadid, int *threadcount) {
  vmd_threadlaunch_t *worker = (vmd_threadlaunch_t *) voidparms;
  if (threadid != NULL)
    *threadid = worker->threadid;

  if (threadcount != NULL)
    *threadcount = worker->threadcount;

  return 0;
}


/* worker thread can call this to get its client data pointer */
int vmd_threadlaunch_getdata(void *voidparms, void **clientdata) {
  vmd_threadlaunch_t *worker = (vmd_threadlaunch_t *) voidparms;
  if (clientdata != NULL)
    *clientdata = worker->clientdata;

  return 0;
}

/* iterate the shared iterator over the requested half-open interval */
int vmd_threadlaunch_next_tile(void *voidparms, int reqsize, 
                               vmd_tasktile_t *tile) {
  vmd_threadlaunch_t *worker = (vmd_threadlaunch_t *) voidparms;
  return vmd_shared_iterator_next_tile(worker->iter, reqsize, tile);
}

/* worker thread calls this to indicate that an unrecoverable error occured */
int vmd_threadlaunch_setfatalerror(void *voidparms) {
  vmd_threadlaunch_t *worker = (vmd_threadlaunch_t *) voidparms;
  return vmd_shared_iterator_setfatalerror(worker->iter);
}

#ifdef __cplusplus
}
#endif


