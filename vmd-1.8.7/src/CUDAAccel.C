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
 *	$RCSfile: CUDAAccel.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.33 $	$Date: 2009/06/01 06:25:00 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Class to store and handle enumeration and initialization of 
 *   CUDA GPU accelerator devices.
 ***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "config.h"     // rebuild on config changes
#include "Inform.h"
#include "ResizeArray.h"
#include "CUDAAccel.h"
#include "CUDAKernels.h"
#include "VMDThreads.h"

CUDAAccel::CUDAAccel(void) {
  cudaavail = 0;
  numdevices = 0;
  int usabledevices = 0;
  cudapool=NULL;

#if defined(VMDCUDA)
  int rc = 0;
  if ((rc=vmd_cuda_num_devices(&numdevices)) != VMDCUDA_ERR_NONE) {
    numdevices = 0;

    // Only emit error messages when there are CUDA GPUs on the machine
    // but that they can't be used for some reason
    // XXX turning this off for the time being, as some people have 
    //     NVIDIA drivers installed on machines with no NVIDIA GPU, as can
    //     happen with some distros that package the drivers by default.
    switch (rc) {
      case VMDCUDA_ERR_NODEVICES:
      case VMDCUDA_ERR_SOMEDEVICES:
        msgInfo << "No CUDA accelerator devices available." << sendmsg;
        break;

#if 0
      case VMDCUDA_ERR_SOMEDEVICES:
        msgWarn << "One or more CUDA accelerators may exist but are not usable." << sendmsg; 
        msgWarn << "Check to make sure that GPU drivers are up to date." << sendmsg;
        break;
#endif

      case VMDCUDA_ERR_DRVMISMATCH:
        msgWarn << "Detected a mismatch between CUDA runtime and GPU driver" << sendmsg; 
        msgWarn << "Check to make sure that GPU drivers are up to date." << sendmsg;
        msgInfo << "No CUDA accelerator devices available." << sendmsg;
        break;
    }
   
    return;
  }

  if (numdevices > 0) {
    cudaavail = 1;

    int i;
    for (i=0; i<numdevices; i++) {
      cudadevprops dp;
      memset(&dp, 0, sizeof(dp));
      if (!vmd_cuda_device_props(i, dp.name, sizeof(dp.name),
                                &dp.major, &dp.minor,
                                &dp.membytes, &dp.clockratekhz, 
                                &dp.smcount, &dp.overlap,
                                &dp.kernelexectimeoutenabled,
                                &dp.canmaphostmem, &dp.computemode)) {
        dp.deviceid=i; // save the device index

        if (!(dp.kernelexectimeoutenabled && getenv("VMDCUDANODISPLAYGPUS")) &&
            (dp.computemode != computeModeProhibited)) {
          devprops.append(dp);
          usabledevices++;
        }
      } else {
        msgWarn << "  Failed to retrieve properties for CUDA accelerator " << i << sendmsg; 
      }
    }
  }
  numdevices=usabledevices;

  print_cuda_devices();

  devpool_init();
#endif
}

// destructor
CUDAAccel::~CUDAAccel(void) {
  devpool_fini();
}


void CUDAAccel::devpool_init(void) {
  cudapool=NULL;
#if defined(VMDTHREADS) && defined(VMDCUDA)
  if (!cudaavail || numdevices == 0)
    return;

  // only use as many GPUs as CPU cores we're allowed to use
  int workercount=numdevices;
  if (workercount > vmd_thread_numprocessors())
    workercount=vmd_thread_numprocessors();

  int *devlist = new int[workercount];
  int i;
  for (i=0; i<workercount; i++) {
    devlist[i]=device_index(i);
  }

  msgInfo << "Creating CUDA device pool and initializing hardware..." << sendmsg;
  cudapool=vmd_threadpool_create(workercount, devlist);
  delete [] devlist;

  // associate each worker thread with a specific GPU
  if (getenv("VMDCUDAVERBOSE") != NULL)
    vmd_threadpool_launch(cudapool, vmd_cuda_devpool_setdevice, (void*)"VMD CUDA Dev Init", 1);
  else
    vmd_threadpool_launch(cudapool, vmd_cuda_devpool_setdevice, NULL, 1);

  // clear all available device memory on each of the GPUs
  vmd_threadpool_launch(cudapool, vmd_cuda_devpool_clear_device_mem, NULL, 1);
#endif
}

void CUDAAccel::devpool_fini(void) {
  if (!cudapool)
    return;

#if defined(VMDTHREADS) && defined(VMDCUDA)
  devpool_wait();
  vmd_threadpool_destroy(cudapool);
#endif
  cudapool=NULL;
}

int CUDAAccel::devpool_launch(void *fctn(void *), void *parms, int blocking) {
  if (!cudapool)
    return -1;

  return vmd_threadpool_launch(cudapool, fctn, parms, blocking); 
}

int CUDAAccel::devpool_wait(void) {
  if (!cudapool)
    return -1;

  return vmd_threadpool_wait(cudapool);
}

void CUDAAccel::print_cuda_devices(void) {
  if (getenv("VMDCUDANODISPLAYGPUS")) {
    msgInfo << "Ignoring CUDA-capable GPUs used for display" << sendmsg;
  }

  if (!cudaavail || numdevices == 0) {
    msgInfo << "No CUDA accelerator devices available." << sendmsg;
    return;
  }

  msgInfo << "Detected " << numdevices << " available CUDA " 
          << ((numdevices > 1) ? "accelerators:" : "accelerator:") << sendmsg;
  int i;
  for (i=0; i<numdevices; i++) {
    char outstr[1024];
    memset(outstr, 0, sizeof(outstr));
    sprintf(outstr, "  [%d] %-18s %2d SM_%d.%d @ %.2f GHz, %4dMB RAM",
            device_index(i), device_name(i), 
            (device_sm_count(i) > 0) ? device_sm_count(i) : 0,
            device_version_major(i), device_version_minor(i),
            device_clock_ghz(i),
            (int) (device_membytes(i) / (1024 * 1024)));
    msgInfo << outstr;

    if (device_computemode(i) == computeModeProhibited) {
      msgInfo << ", Compute Mode: Prohibited";
    } else {
      if (device_kerneltimeoutenabled(i))
        msgInfo << ", KTO";

      if (device_overlap(i))
        msgInfo << ", OIO";

      if (device_canmaphostmem(i))
        msgInfo << ", ZCP";
    }

    msgInfo << sendmsg; 
  } 
}

int CUDAAccel::num_devices(void) {
  return numdevices;
}

int CUDAAccel::device_index(int dev) {
  return devprops[dev].deviceid;
}

const char *CUDAAccel::device_name(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return NULL;
  return devprops[dev].name; 
}

int CUDAAccel::device_version_major(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return 0; 
  return devprops[dev].major;
}

int CUDAAccel::device_version_minor(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return 0; 
  return devprops[dev].minor;
}

unsigned long CUDAAccel::device_membytes(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return 0; 
  return devprops[dev].membytes;
}

float CUDAAccel::device_clock_ghz(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return 0; 
  return (float) (devprops[dev].clockratekhz / 1000000.0);
}

int CUDAAccel::device_sm_count(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return -1; 
  return devprops[dev].smcount;
}

int CUDAAccel::device_overlap(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return -1; 
  return devprops[dev].overlap;
}

int CUDAAccel::device_kerneltimeoutenabled(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return -1; 
  return devprops[dev].kernelexectimeoutenabled;
}

int CUDAAccel::device_canmaphostmem(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return -1; 
  return devprops[dev].canmaphostmem;
}

int CUDAAccel::device_computemode(int dev) {
  if (!cudaavail || dev < 0 || dev >= numdevices)
    return -1; 
  return devprops[dev].computemode;
}


