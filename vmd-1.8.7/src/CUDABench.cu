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
 *      $RCSfile: CUDABench.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.16 $      $Date: 2009/06/18 18:50:53 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Short benchmark kernels to measure GPU performance
 *
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#include "utilities.h"
#include "VMDThreads.h"
#include "CUDAKernels.h" 

#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
  return -1; }}


//
// Floating point multiply-add benchmark components
//

// FMADD16 macro contains a sequence of operations that the compiler
// won't optimize out, and will translate into a densely packed block
// of multiply-add instructions with no intervening register copies/moves
// or other instructions. 
#define FMADD16 \
    tmp0  = tmp0*tmp4+tmp7;     \
    tmp1  = tmp1*tmp5+tmp0;     \
    tmp2  = tmp2*tmp6+tmp1;     \
    tmp3  = tmp3*tmp7+tmp2;     \
    tmp4  = tmp4*tmp0+tmp3;     \
    tmp5  = tmp5*tmp1+tmp4;     \
    tmp6  = tmp6*tmp2+tmp5;     \
    tmp7  = tmp7*tmp3+tmp6;     \
    tmp8  = tmp8*tmp12+tmp15;   \
    tmp9  = tmp9*tmp13+tmp8;    \
    tmp10 = tmp10*tmp14+tmp9;   \
    tmp11 = tmp11*tmp15+tmp10;  \
    tmp12 = tmp12*tmp8+tmp11;   \
    tmp13 = tmp13*tmp9+tmp12;   \
    tmp14 = tmp14*tmp10+tmp13;  \
    tmp15 = tmp15*tmp11+tmp14;

// CUDA grid, thread block, loop, and MADD operation counts
#define GRIDSIZEX   6144
#define BLOCKSIZEX  64
#define GLOOPS      500
#define MADDCOUNT   64

// FLOP counting
#define FLOPSPERLOOP (MADDCOUNT * 16)

//
// Benchmark peak Multiply-Add instruction performance, in GFLOPS
//
__global__ static void madd_kernel(float *doutput) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7;
  float tmp8,tmp9,tmp10,tmp11,tmp12,tmp13,tmp14,tmp15;
  tmp0=tmp1=tmp2=tmp3=tmp4=tmp5=tmp6=tmp7=0.0f;
  tmp8=tmp9=tmp10=tmp11=tmp12=tmp13=tmp14=tmp15 = 0.0f;

  tmp15=tmp7 = blockIdx.x * 0.001; // prevent compiler from optimizing out
  tmp1 = blockIdx.y * 0.001;       // the body of the loop...

  int loop;
  for(loop=0; loop<GLOOPS; loop++){
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
    FMADD16
  }

  doutput[tid] = tmp0+tmp1+tmp2+tmp3+tmp4+tmp5+tmp6+tmp7
                 +tmp8+tmp9+tmp10+tmp11+tmp12+tmp13+tmp14+tmp15;
}


static int cudamaddgflops(int cudadev, double *gflops, int testloops) {
  float *doutput = NULL;
  dim3 Bsz, Gsz;
  vmd_timerhandle timer;
  int i;

  cudaError_t rc;
  rc = cudaSetDevice(cudadev);
  if (rc != cudaSuccess) {
#if CUDART_VERSION >= 2010
    rc = cudaGetLastError(); // query last error and reset error state
    if (rc != cudaErrorSetOnActiveProcess)
      return NULL; // abort and return an error
#else
    cudaGetLastError(); // just ignore and reset error state, since older CUDA
                        // revs don't have a cudaErrorSetOnActiveProcess enum
#endif
  }


  // setup CUDA grid and block sizes
  Bsz.x = BLOCKSIZEX;
  Bsz.y = 1;
  Bsz.z = 1;
  Gsz.x = GRIDSIZEX;
  Gsz.y = 1;
  Gsz.z = 1;

  // allocate output array
  cudaMalloc((void**)&doutput, BLOCKSIZEX * GRIDSIZEX * sizeof(float));
  CUERR // check and clear any existing errors

  timer=vmd_timer_create();
  vmd_timer_start(timer);
  for (i=0; i<testloops; i++) { 
    madd_kernel<<<Gsz, Bsz>>>(doutput);
    cudaThreadSynchronize(); // wait for kernel to finish
  }
  CUERR // check and clear any existing errors
  vmd_timer_stop(timer);

  double runtime = vmd_timer_time(timer);
  double gflop = ((double) GLOOPS) * ((double) FLOPSPERLOOP) *
                  ((double) BLOCKSIZEX) * ((double) GRIDSIZEX) * (1.0e-9) * testloops;
  
  *gflops = gflop / runtime;

  cudaFree(doutput);
  CUERR // check and clear any existing errors

  vmd_timer_destroy(timer);

  return 0;
}

typedef struct {
  int deviceid;
  int testloops;
  double gflops;
} maddthrparms;

static void * cudamaddthread(void *voidparms) {
  maddthrparms *parms = (maddthrparms *) voidparms;
  cudamaddgflops(parms->deviceid, &parms->gflops, parms->testloops);
  return NULL;
}

int vmd_cuda_madd_gflops(int numdevs, int *devlist, double *gflops,
                         int testloops) {
  maddthrparms *parms;
  vmd_thread_t * threads;
  int i;

  /* allocate array of threads */
  threads = (vmd_thread_t *) calloc(numdevs * sizeof(vmd_thread_t), 1);

  /* allocate and initialize array of thread parameters */
  parms = (maddthrparms *) malloc(numdevs * sizeof(maddthrparms));
  for (i=0; i<numdevs; i++) {
    if (devlist != NULL)
      parms[i].deviceid = devlist[i];
    else
      parms[i].deviceid = i;

    parms[i].testloops = testloops;
    parms[i].gflops = 0.0;
  }

#if defined(VMDTHREADS)
  /* spawn child threads to do the work */
  /* thread 0 must also be processed this way otherwise    */
  /* we'll permanently bind the main thread to some device */
  for (i=0; i<numdevs; i++) {
    vmd_thread_create(&threads[i], cudamaddthread, &parms[i]);
  }

  /* join the threads after work is done */
  for (i=0; i<numdevs; i++) {
    vmd_thread_join(threads[i], NULL);
  }
#else
  /* single thread does all of the work */
  cudamaddthread((void *) &parms[0]);
#endif

  for (i=0; i<numdevs; i++) {
    gflops[i] = parms[i].gflops; 
  }

  /* free thread parms */
  free(parms);
  free(threads);

  return 0;
}






//
// Host-GPU memory bandwidth benchmark components
//

#define BWITER 500

static int cudabusbw(int cudadev, double *hdmbsec, double *dhmbsec,
                     double *phdmbsec, double *pdhmbsec) {
  float *hdata = NULL;   // non-pinned DMA buffer
  float *phdata = NULL;  // pinned DMA buffer
  float *ddata = NULL;

  int i;
  int memsz = 1024 * 1024 * sizeof(float);
  double runtime;
  vmd_timerhandle timer;

  // attach to the selected device
  cudaError_t rc;
  rc = cudaSetDevice(cudadev);
  if (rc != cudaSuccess) {
#if CUDART_VERSION >= 2010
    rc = cudaGetLastError(); // query last error and reset error state
    if (rc != cudaErrorSetOnActiveProcess)
      return NULL; // abort and return an error
#else
    cudaGetLastError(); // just ignore and reset error state, since older CUDA
                        // revs don't have a cudaErrorSetOnActiveProcess enum
#endif
  }

  // allocate non-pinned output array
  hdata = (float *) malloc(memsz); 

  // allocate pinned output array
  cudaMallocHost((void**) &phdata, memsz);
  CUERR // check and clear any existing errors

  // allocate device memory
  cudaMalloc((void**) &ddata, memsz);
  CUERR // check and clear any existing errors

  //
  // Host to device timings
  //

  // non-pinned
  timer=vmd_timer_create();
  vmd_timer_start(timer);
  for (i=0; i<BWITER; i++) {
    cudaMemcpy(ddata, hdata, memsz,  cudaMemcpyHostToDevice);
    CUERR // check and clear any existing errors
  }
  vmd_timer_stop(timer);
  runtime = vmd_timer_time(timer);
  *hdmbsec = ((double) BWITER) * ((double) memsz) / runtime / (1024.0 * 1024.0);

  // pinned
  timer=vmd_timer_create();
  vmd_timer_start(timer);
  for (i=0; i<BWITER; i++) {
    cudaMemcpy(ddata, phdata, memsz,  cudaMemcpyHostToDevice);
    CUERR // check and clear any existing errors
  }
  vmd_timer_stop(timer);
  runtime = vmd_timer_time(timer);
  *phdmbsec = ((double) BWITER) * ((double) memsz) / runtime / (1024.0 * 1024.0);
 
  //
  // Device to host timings
  //

  // non-pinned
  timer=vmd_timer_create();
  vmd_timer_start(timer);
  for (i=0; i<BWITER; i++) {
    cudaMemcpy(hdata, ddata, memsz,  cudaMemcpyDeviceToHost);
    CUERR // check and clear any existing errors
  }
  vmd_timer_stop(timer);
  runtime = vmd_timer_time(timer);
  *dhmbsec = ((double) BWITER) * ((double) memsz) / runtime / (1024.0 * 1024.0);

  // pinned
  timer=vmd_timer_create();
  vmd_timer_start(timer);
  for (i=0; i<BWITER; i++) {
    cudaMemcpy(phdata, ddata, memsz,  cudaMemcpyDeviceToHost);
    CUERR // check and clear any existing errors
  }
  vmd_timer_stop(timer);
  runtime = vmd_timer_time(timer);
  *pdhmbsec = ((double) BWITER) * ((double) memsz) / runtime / (1024.0 * 1024.0);
 
  cudaFree(ddata);
  CUERR // check and clear any existing errors
  cudaFreeHost(phdata);
  CUERR // check and clear any existing errors
  free(hdata);

  vmd_timer_destroy(timer);

  return 0;
}

typedef struct {
  int deviceid;
  double hdmbsec;
  double phdmbsec;
  double dhmbsec;
  double pdhmbsec;
} busbwthrparms;

static void * cudabusbwthread(void *voidparms) {
  busbwthrparms *parms = (busbwthrparms *) voidparms;
  cudabusbw(parms->deviceid, &parms->hdmbsec, &parms->phdmbsec,
            &parms->dhmbsec, &parms->pdhmbsec);
  return NULL;
}

int vmd_cuda_bus_bw(int numdevs, int *devlist, 
                    double *hdmbsec, double *phdmbsec,
                    double *dhmbsec, double *pdhmbsec) {
  busbwthrparms *parms;
  vmd_thread_t * threads;
  int i;

  /* allocate array of threads */
  threads = (vmd_thread_t *) calloc(numdevs * sizeof(vmd_thread_t), 1);

  /* allocate and initialize array of thread parameters */
  parms = (busbwthrparms *) malloc(numdevs * sizeof(busbwthrparms));
  for (i=0; i<numdevs; i++) {
    if (devlist != NULL)
      parms[i].deviceid = devlist[i];
    else
      parms[i].deviceid = i;
    parms[i].hdmbsec = 0.0;
    parms[i].phdmbsec = 0.0;
    parms[i].dhmbsec = 0.0;
    parms[i].pdhmbsec = 0.0;
  }

#if defined(VMDTHREADS)
  /* spawn child threads to do the work */
  /* thread 0 must also be processed this way otherwise    */
  /* we'll permanently bind the main thread to some device */
  for (i=0; i<numdevs; i++) {
    vmd_thread_create(&threads[i], cudabusbwthread, &parms[i]);
  }

  /* join the threads after work is done */
  for (i=0; i<numdevs; i++) {
    vmd_thread_join(threads[i], NULL);
  }
#else
  /* single thread does all of the work */
  cudabusbwthread((void *) &parms[0]);
#endif

  for (i=0; i<numdevs; i++) {
    hdmbsec[i] = parms[i].hdmbsec; 
    phdmbsec[i] = parms[i].phdmbsec; 
    dhmbsec[i] = parms[i].dhmbsec; 
    pdhmbsec[i] = parms[i].pdhmbsec; 
  }

  /* free thread parms */
  free(parms);
  free(threads);

  return 0;
}



