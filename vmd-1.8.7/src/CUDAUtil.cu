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
 *      $RCSfile: CUDAUtil.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.27 $        $Date: 2009/06/01 06:25:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   CUDA API wrapper for use by the CUDAAccel C++ class 
 ***************************************************************************/
#include <string.h>
#include <stdio.h>
#include "CUDAKernels.h"
#include "VMDThreads.h"

#if defined(__cplusplus)
extern "C" {
#endif

// report true if the driver is compatible with the runtime
static int vmd_cuda_drv_runtime_compatible() {
#if CUDART_VERSION >= 2020
  int cuda_driver_version=-1;
  int cuda_runtime_version=0;

  cudaDriverGetVersion(&cuda_driver_version);
  cudaRuntimeGetVersion(&cuda_runtime_version);
//  printf("CUDA driver version: %d\n", cuda_driver_version);
//  printf("CUDA runtime version: %d\n", cuda_runtime_version);

  if (cuda_driver_version == 0) 
    return VMDCUDA_ERR_NODEVICES;

  if (cuda_driver_version < cuda_runtime_version)
    return VMDCUDA_ERR_DRVMISMATCH;
#endif  

  return VMDCUDA_ERR_NONE;
}


int vmd_cuda_device_props(int dev, char *name, int namelen,
                          int *devmajor, int *devminor, 
                          unsigned long *memb, int *clockratekhz,
                          int *smcount, int *overlap, int *kerneltimeout,
                          int *canmaphostmem, int *computemode) {
  cudaError_t rc;
  cudaDeviceProp deviceProp;

  int vercheck;
  if ((vercheck = vmd_cuda_drv_runtime_compatible()) != VMDCUDA_ERR_NONE) {
    return vercheck;
  }

  memset(&deviceProp, 0, sizeof(cudaDeviceProp));
  if ((rc=cudaGetDeviceProperties(&deviceProp, dev)) != cudaSuccess) {
    // printf("error: %s\n", cudaGetErrorString(rc));
    if (rc == cudaErrorNotYetImplemented)
      return VMDCUDA_ERR_EMUDEVICE;
    return VMDCUDA_ERR_GENERAL;
  }

  if (name)
    strncpy(name, deviceProp.name, namelen);
  if (devmajor)
    *devmajor = deviceProp.major;
  if (devminor)
    *devminor = deviceProp.minor;
  if (memb)
    *memb = deviceProp.totalGlobalMem;
  if (clockratekhz)
    *clockratekhz = deviceProp.clockRate;
#if CUDART_VERSION >= 2000
  if (smcount)
    *smcount = deviceProp.multiProcessorCount;
  if (overlap)
    *overlap = (deviceProp.deviceOverlap != 0);
#else
  if (smcount)
    *smcount = -1;
  if (overlap)
    *overlap = 0; 
#endif
#if CUDART_VERSION >= 2010
  if (kerneltimeout)
    *kerneltimeout = (deviceProp.kernelExecTimeoutEnabled != 0);
#else
  if (kerneltimeout)
    *kerneltimeout = 0;
#endif
#if CUDART_VERSION >= 2020
  if (canmaphostmem)
    *canmaphostmem = (deviceProp.canMapHostMemory != 0);
  if (computemode)
    *computemode = deviceProp.computeMode;
#else
  if (canmaphostmem)
    *canmaphostmem = 0;
  if (computemode)
    *computemode = VMDCUDA_COMPUTEMODE_DEFAULT;
#endif
  return VMDCUDA_ERR_NONE;
}


int vmd_cuda_num_devices(int *numdev) {
  int i;
  int devcount=0;
  int usabledevs=0;
  *numdev = 0;

  int vercheck;
  if ((vercheck = vmd_cuda_drv_runtime_compatible()) != VMDCUDA_ERR_NONE) {
    return vercheck;
  }

  if (cudaGetDeviceCount(&devcount) != cudaSuccess) {
    return VMDCUDA_ERR_NODEVICES;
  }

  // Do a sanity check in case we get complete gibberish back,
  // but no error. This can occur if we have either a driver or 
  // CUDA runtime that's badly mismatched.
  if (devcount > 100 || devcount < 0)
    return VMDCUDA_ERR_DRVMISMATCH;

  // disregard emulation mode as unusable for our purposes
  for (i=0; i<devcount; i++) {
    int devmajor, devminor, rc;

    rc = vmd_cuda_device_props(i, NULL, 0, &devmajor, &devminor, 
                               NULL, NULL, NULL, NULL, NULL, NULL, NULL);

    if (rc == VMDCUDA_ERR_NONE) {
      // Check for emulation mode devices, and ignore if found
      if (((devmajor >= 1) && (devminor >= 0)) &&
          ((devmajor != 9999) && (devminor != 9999))) {
        usabledevs++;
      }
    } else if (rc != VMDCUDA_ERR_EMUDEVICE) {
      return VMDCUDA_ERR_SOMEDEVICES;
    }
  } 

  *numdev = usabledevs;

  return VMDCUDA_ERR_NONE;
}


void * vmd_cuda_devpool_setdevice(void * voidparms) {
  int count, id, dev;
  cudaDeviceProp deviceProp;
  char *mesg;
  vmd_threadpool_worker_getid(voidparms, &id, &count);
  vmd_threadpool_worker_getdata(voidparms, (void **) &mesg);
  vmd_threadpool_worker_getdevid(voidparms, &dev);

  /* set active device */
  cudaSetDevice(dev);

#if CUDART_VERSION >= 2000
  /* Query SM count and clock rate, and compute a speed scaling value */
  /* the current code uses a GeForce GTX 280 / Tesla C1060 as the     */
  /* "1.0" reference value, with 30 SMs, and a 1.3 GHz clock rate     */ 
  memset(&deviceProp, 0, sizeof(cudaDeviceProp));
  if (cudaGetDeviceProperties(&deviceProp, dev) == cudaSuccess) {
    float smscale = ((float) deviceProp.multiProcessorCount) / 30.0f;
    double clockscale = ((double) deviceProp.clockRate) / 1295000.0;
    float speedscale = smscale * ((float) clockscale);

#if 0
    printf("clock rate: %lf\n", (double) deviceProp.clockRate);
    printf("scale: %.4f smscale: %.4f clockscale: %.4f\n", 
           speedscale, smscale, clockscale);  
#endif
    
    vmd_threadpool_worker_setdevspeed(voidparms, speedscale);
  }
#endif

  if (mesg != NULL)
    printf("devpool thread[%d / %d], device %d message: '%s'\n", id, count, dev, mesg);

  return NULL;
}



#if defined(__cplusplus)
}
#endif


