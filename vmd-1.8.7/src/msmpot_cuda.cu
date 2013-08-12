/***************************************************************************
 *cr
 *cr            (C) Copyright 2008-2009 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *      $RCSfile: msmpot_cuda.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.2 $      $Date: 2009/04/29 15:49:47 $
 *
 ***************************************************************************/
//
// msmcuda.cu
//
#include "msmpot_internal.h"


MsmpotCuda *Msmpot_cuda_create(void) {
  MsmpotCuda *mc = (MsmpotCuda *) calloc(1, sizeof(MsmpotCuda));
  return mc;
}


void Msmpot_cuda_destroy(MsmpotCuda *mc) {
  Msmpot_cuda_cleanup(mc);
  free(mc);
}


void Msmpot_cuda_cleanup(MsmpotCuda *mc) {
  Msmpot_cuda_cleanup_latcut(mc);
  Msmpot_cuda_cleanup_shortrng(mc);
  free(mc->dev);
}


static int list_devices(MsmpotCuda *);
static int real_devices(MsmpotCuda *);
static int set_device(MsmpotCuda *, int devnum);


int Msmpot_cuda_setup(MsmpotCuda *mc, Msmpot *msm) {
  int err = OK;

  mc->msmpot = msm;

  msm->use_cuda_shortrng = 0;  // be pessimistic
  msm->use_cuda_latcut = 0;

  err = list_devices(mc);
  if (MSMPOT_ERROR_CUDA == err || MSMPOT_ERROR_AVAIL == err) return OK;
  else if (err) return ERROR(err);

  err = set_device(mc, 0);  // just use device 0
  if (MSMPOT_ERROR_CUDA == err || MSMPOT_ERROR_AVAIL == err) return OK;
  else if (err) return ERROR(err);
  mc->devnum = 0;

  err = Msmpot_cuda_setup_shortrng(mc);
  if (MSMPOT_ERROR_NONE == err) msm->use_cuda_shortrng = 1;
  else if (MSMPOT_ERROR_CUDA != err && MSMPOT_ERROR_AVAIL != err) {
    return ERROR(err);
  }

  err = Msmpot_cuda_setup_latcut(mc);
  if (MSMPOT_ERROR_NONE == err) msm->use_cuda_latcut = 1;
  else if (MSMPOT_ERROR_CUDA != err && MSMPOT_ERROR_AVAIL != err) {
    return ERROR(err);
  }

  return OK;
}


int list_devices(MsmpotCuda *mc) {
  void *v;
  int ndevs, i;

  if (mc->dev) return real_devices(mc);  // we already have device list

  cudaGetDeviceCount(&ndevs);
  if (ndevs < 1) return ERROR(MSMPOT_ERROR_CUDA);

  v = realloc(mc->dev, ndevs * sizeof(struct cudaDeviceProp));
  if (NULL == v) return ERROR(MSMPOT_ERROR_ALLOC);
  mc->dev = (struct cudaDeviceProp *) v;
  mc->ndevs = ndevs;

  for (i = 0;  i < ndevs;  i++) {
    cudaGetDeviceProperties(mc->dev + i, i);
    CUERR;
  }
  return real_devices(mc);
}


// verify CUDA devices are real rather than emulation mode
int real_devices(MsmpotCuda *mc) {
  const int ndevs = mc->ndevs;
  int i;

  for (i = 0;  i < ndevs;  i++) {
    if (9999 == mc->dev[i].major && 9999 == mc->dev[i].minor) {
      return ERROR(MSMPOT_ERROR_CUDA);  // emulation mode
    }
  }
  return OK;
}


int set_device(MsmpotCuda *mc, int devnum) {
  int rc = cudaSetDevice(devnum);
  if (rc != cudaSuccess) {
#if CUDART_VERSION >= 2010
    rc = cudaGetLastError(); // query last error and reset error state
    if (rc != cudaErrorSetOnActiveProcess) {
      return ERROR(MSMPOT_ERROR_CUDA); // abort and return an error
    }
#else
    cudaGetLastError(); // just ignore and reset error state, since older CUDA
                        // revs don't have a cudaErrorSetOnActiveProcess enum
#endif
  }
  return OK;
}
