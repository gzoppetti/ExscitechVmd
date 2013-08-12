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
 *      $RCSfile: cmd_vmdbench.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.11 $       $Date: 2009/06/18 18:50:53 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   text commands for benchmarking hardware performance
 ***************************************************************************/

#include <tcl.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Benchmark.h"
#include "config.h"
#include "VMDApp.h"
#include "TclCommands.h"
#include "CUDAKernels.h"
#include "VMDThreads.h"

static void cmd_vmdbench_usage(Tcl_Interp *interp) {
  Tcl_AppendResult(interp,
      "usage: vmdbench <command> [args...]\n"
      "vmdbench stream    [N]       -- built-in STREAM memory bandwidth test\n",
      "vmdbench cudamadd  [devices] -- CUDA multiply-add arithmetic (*)\n",
      "vmdbench cudabusbw [devices] -- CUDA host/device bus bandwidth (*)\n",
      "(*) Only available in CUDA-enabled builds of VMD\n",
      NULL);
}

int text_cmd_vmdbench(ClientData cd, Tcl_Interp *interp, int argc, 
                      const char *argv[]) {

//  VMDApp *app = (VMDApp *)cd;

  if (argc == 1) {
    cmd_vmdbench_usage(interp);
    return TCL_ERROR;
  }

  if (argc >= 2) {
    if (!strupncmp(argv[1], "stream", CMDLEN)) {
      double times[8], mbsec[8];
      int N = 1024*1024 * 16;

      if (argc == 3) {
        if (Tcl_GetInt(interp, argv[2], &N) != TCL_OK) {
          Tcl_AppendResult(interp, " in vmdbench stream", NULL);
          return TCL_ERROR;
        }
      }

      int rc = stream_bench(N, times, mbsec);
      if (rc) {
        Tcl_AppendResult(interp,
          "unable to complete stream benchmark, out of memory", NULL);
        return TCL_ERROR;
      }

      Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
      const char *benchnames[] = {
        "copy (double)",
        "scale (double)",
        "add (double)",
        "triad (double)",
        "copy (float)",
        "scale (float)",
        "add (float)",
        "triad (float)"
      };

      Tcl_Obj *colNameObj = Tcl_NewListObj(0, NULL);
      Tcl_ListObjAppendElement(interp, colNameObj, Tcl_NewStringObj("Test", -1)); 
      Tcl_ListObjAppendElement(interp, colNameObj, Tcl_NewStringObj("Time", -1)); 
      Tcl_ListObjAppendElement(interp, colNameObj, Tcl_NewStringObj("MB/sec", -1)); 
      Tcl_ListObjAppendElement(interp, tcl_result, colNameObj);

      int i;     
      for (i=0; i<8; i++) {
        Tcl_Obj *rowListObj = Tcl_NewListObj(0, NULL);
        Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewStringObj(benchnames[i], -1)); 
        Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewDoubleObj(times[i])); 
        Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewDoubleObj(mbsec[i])); 
        Tcl_ListObjAppendElement(interp, tcl_result, rowListObj);

      }
      Tcl_SetObjResult(interp, tcl_result);

      return TCL_OK;
    } else if (!strupncmp(argv[1], "cudamadd", CMDLEN)) {
#if defined(VMDCUDA)
      int numdevs, physnumdevs;
      int *devlist = NULL;
      vmd_cuda_num_devices(&physnumdevs);
      numdevs = physnumdevs;
#if !defined(VMDTHREADS)
      numdevs = 1;
#endif

      // handle optional device list arguments
      if (argc > 2) {
        if ((argc-2) > numdevs) {
          Tcl_AppendResult(interp, "vmdbench: bad device argument", NULL);
          return TCL_ERROR;
        } else {
          numdevs = argc-2;
        }
        devlist = (int *) malloc(numdevs * sizeof(int));
        int arg, dev;
        for (arg=0; arg<numdevs; arg++) {
          if (Tcl_GetInt(interp, argv[arg+2], &dev) != TCL_OK) {
            Tcl_AppendResult(interp, "vmdbench: bad device argument", NULL);
            free(devlist);
            return TCL_ERROR;
          }
          if (dev < 0 || dev >= physnumdevs) {
            Tcl_AppendResult(interp, "vmdbench: device argument out of range", NULL);
            free(devlist);
            return TCL_ERROR;
          }
          devlist[arg] = dev;
        } 
      }

      double *gflops = (double *) malloc(numdevs * sizeof(double));
      int testloops=1;
      if (getenv("VMDMADDLOOPS") != NULL)
        testloops = atoi(getenv("VMDMADDLOOPS"));

      vmd_cuda_madd_gflops(numdevs, devlist, gflops, testloops);

      Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
      Tcl_Obj *colNameObj = Tcl_NewListObj(0, NULL);
      Tcl_ListObjAppendElement(interp, colNameObj, Tcl_NewStringObj("Device", -1));
      Tcl_ListObjAppendElement(interp, colNameObj, Tcl_NewStringObj("GFLOPS", -1));
      Tcl_ListObjAppendElement(interp, tcl_result, colNameObj);

      int i;
      for (i=0; i<numdevs; i++) {
        Tcl_Obj *rowListObj = Tcl_NewListObj(0, NULL);
        if (devlist != NULL) 
          Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewIntObj(devlist[i]));
        else
          Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewIntObj(i));
        Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewDoubleObj(gflops[i]));
        Tcl_ListObjAppendElement(interp, tcl_result, rowListObj);
      }
      Tcl_SetObjResult(interp, tcl_result);

      if (devlist)
        free(devlist);

      return TCL_OK;
#else 
      Tcl_AppendResult(interp, "CUDA Acceleration not available in this build", NULL);
      return TCL_ERROR;
#endif
    } else if (!strupncmp(argv[1], "cudabusbw", CMDLEN)) {
#if defined(VMDCUDA)
      int numdevs, physnumdevs;
      int *devlist = NULL;
      vmd_cuda_num_devices(&physnumdevs);
      numdevs = physnumdevs;
#if !defined(VMDTHREADS)
      numdevs = 1;
#endif

      // handle optional device list arguments
      if (argc > 2) {
        if ((argc-2) > numdevs) {
          Tcl_AppendResult(interp, "vmdbench: bad device argument", NULL);
          return TCL_ERROR;
        } else {
          numdevs = argc-2;
        }
        devlist = (int *) malloc(numdevs * sizeof(int));
        int arg, dev;
        for (arg=0; arg<numdevs; arg++) {
          if (Tcl_GetInt(interp, argv[arg+2], &dev) != TCL_OK) {
            Tcl_AppendResult(interp, "vmdbench: bad device argument", NULL);
            free(devlist);
            return TCL_ERROR;
          }
          if (dev < 0 || dev >= physnumdevs) {
            Tcl_AppendResult(interp, "vmdbench: device argument out of range", NULL);
            free(devlist);
            return TCL_ERROR;
          }
          devlist[arg] = dev;
        } 
      }

      double *hdmbsec = (double *) malloc(numdevs * sizeof(double));
      double *phdmbsec = (double *) malloc(numdevs * sizeof(double));
      double *dhmbsec = (double *) malloc(numdevs * sizeof(double));
      double *pdhmbsec = (double *) malloc(numdevs * sizeof(double));

      vmd_cuda_bus_bw(numdevs, devlist, hdmbsec, phdmbsec, dhmbsec, pdhmbsec);

      Tcl_Obj *tcl_result = Tcl_NewListObj(0, NULL);
      Tcl_Obj *colNameObj = Tcl_NewListObj(0, NULL);
      Tcl_ListObjAppendElement(interp, colNameObj, Tcl_NewStringObj("Device", -1));
      Tcl_ListObjAppendElement(interp, colNameObj, Tcl_NewStringObj("Host-device bandwidth (MB/sec)", -1));
      Tcl_ListObjAppendElement(interp, colNameObj, Tcl_NewStringObj("Host-device pinned bandwidth (MB/sec)", -1));
      Tcl_ListObjAppendElement(interp, colNameObj, Tcl_NewStringObj("Device-host bandwidth (MB/sec)", -1));
      Tcl_ListObjAppendElement(interp, colNameObj, Tcl_NewStringObj("Device-host pinned bandwidth (MB/sec)", -1));
      Tcl_ListObjAppendElement(interp, tcl_result, colNameObj);

      int i;
      for (i=0; i<numdevs; i++) {
        Tcl_Obj *rowListObj = Tcl_NewListObj(0, NULL);
        if (devlist != NULL) 
          Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewIntObj(devlist[i]));
        else
          Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewIntObj(i));

        Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewDoubleObj(hdmbsec[i]));
        Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewDoubleObj(phdmbsec[i]));
        Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewDoubleObj(dhmbsec[i]));
        Tcl_ListObjAppendElement(interp, rowListObj, Tcl_NewDoubleObj(pdhmbsec[i]));
        Tcl_ListObjAppendElement(interp, tcl_result, rowListObj);
      }
      Tcl_SetObjResult(interp, tcl_result);
      return TCL_OK;
#else 
      Tcl_AppendResult(interp, "CUDA Acceleration not available in this build", NULL);
      return TCL_ERROR;
#endif
    } else {
      cmd_vmdbench_usage(interp);
      return TCL_ERROR;
    }
  } else {
    cmd_vmdbench_usage(interp);
    return TCL_ERROR;
  }
  
  // if here, everything worked out ok
  return TCL_OK;
}


