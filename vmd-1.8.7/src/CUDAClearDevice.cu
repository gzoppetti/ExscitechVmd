/***************************************************************************
 *cr
 *cr            (C) Copyright 2007-2009 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: CUDAClearDevice.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.7 $      $Date: 2009/06/03 18:49:12 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   CUDA utility to clear all global and constant GPU memory areas to 
 *   known values.
 *
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#include "utilities.h"
#include "CUDAKernels.h"

#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
  return NULL; }}

// a full-sized 64-kB constant memory buffer to use to clear
// any existing device state
__constant__ static float constbuf[16384];

// maximum number of allocations to use to soak up all available RAM
#define MAXLOOPS 16

void * vmd_cuda_devpool_clear_device_mem(void * voidparms) {
  int i, id, count, dev;
  char *bufs[MAXLOOPS];
  size_t bufszs[MAXLOOPS];
  size_t totalsz=0;
  float zerobuf[16 * 1024];
  int verbose=0;

  if (getenv("VMDCUDAVERBOSE") != NULL)
    verbose=1;

  memset(zerobuf, 0, sizeof(zerobuf));
  memset(bufs, 0, MAXLOOPS * sizeof(sizeof(char *)));
  memset(bufszs, 0, MAXLOOPS * sizeof(sizeof(size_t)));

  vmd_threadpool_worker_getid(voidparms, &id, &count);
  vmd_threadpool_worker_getdevid(voidparms, &dev);

  // clear constant memory
  cudaMemcpyToSymbol(constbuf, zerobuf, sizeof(zerobuf), 0);
  CUERR

  // allocate, clear, and deallocate all global memory we can touch
  size_t sz(1024 * 1024 * 1024); /* start with 1GB buffer size */
  for (i=0; i<MAXLOOPS; i++) {
    while ((sz > 0) && 
           (cudaMalloc((void **) &bufs[i], sz) != cudaSuccess)) {
      cudaGetLastError();
      sz >>= 1;
    }
    bufszs[i] = sz;
    totalsz += sz; 
    if (verbose) {
      printf("devpool thread[%d / %d], dev %d buf[%d] size: %d\n", id, count, dev, i, sz);
    }
  }

  for (i=0; i<MAXLOOPS; i++) {
    if ((bufs[i] != NULL) && (bufszs[i] > 0)) {
      cudaMemset(bufs[i], 0, bufszs[i]);
    }
  }
  CUERR

  for (i=0; i<MAXLOOPS; i++) {
    if ((bufs[i] != NULL) && (bufszs[i] > 0)) {
      cudaFree(bufs[i]);
    } 
  }
  CUERR

  if (verbose)
    printf("  Device %d cleared %d MB of GPU memory\n", dev, totalsz / (1024 * 1024));

  return NULL;
}

