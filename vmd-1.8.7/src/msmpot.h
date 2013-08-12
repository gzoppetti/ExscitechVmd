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
 *      $RCSfile: msmpot.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.2 $      $Date: 2009/04/29 15:49:47 $
 *
 ***************************************************************************/
/*
 * msmpot.h
 *
 * Library interface file.
 */

#ifndef MSMPOT_H
#define MSMPOT_H

#ifdef __cplusplus
extern "C" {
#endif

  struct Msmpot_t;
  typedef struct Msmpot_t Msmpot;   /* handle to private MSM data structure */

  Msmpot *Msmpot_create(void);      /* constructor */

  void Msmpot_destroy(Msmpot *);    /* destructor */

  /* Compute map of electrostatic potential.  Map returned in "epotmap" with
   * return value 0 for success or nonzero for failure.
   *
   * There is some setup overhead on first call.  The overhead is reduced for
   * subsequent calls if map dimensions and number of atoms remain the same
   * and if atoms have the same bounding box.  The overhead is eliminated if
   * the atom bounds are set by Msmpot_bounds() preceding each compute call. */
  int Msmpot_compute(Msmpot *,
      float *epotmap,               /* electrostatic potential map
                                       assumed to be length mx*my*mz,
                                       stored flat in row-major order, i.e.,
                                       &ep[i,j,k] == ep + ((k*my+j)*mx+i) */
      long mx, long my, long mz,    /* lattice dimensions of map:
                                       must be 2^m or 3*2^m for periodic,
                                       must be positive for aperiodic */
      float dx, float dy, float dz, /* lattice spacing:
                                       positive for aperiodic, 0 for periodic */
      float lx, float ly, float lz, /* cell lengths:
                                       positive for periodic, 0 for aperiodic */
      float x0, float y0, float z0, /* minimum reference position of map */
      const float *atom,            /* atoms stored x/y/z/q (length 4*natoms) */
      long natoms                   /* number of atoms */
      );

  const char *Msmpot_error_string(int err);

  enum {
    MSMPOT_CUBIC_INTERP    = 1,
    MSMPOT_QUINTIC_INTERP  = 2,
    MSMPOT_QUINTIC2_INTERP = 3
  };

  enum {
    MSMPOT_TAYLOR2_SPLIT  = 1,
    MSMPOT_TAYLOR3_SPLIT  = 2,
    MSMPOT_TAYLOR4_SPLIT  = 3
  };

  /* the "unknown" error must always be last,
   * the corresponding error strings are in msmpot.c */
  enum {
    MSMPOT_ERROR_NONE    =  0,
    MSMPOT_ERROR_ASSERT  = -1,
    MSMPOT_ERROR_ALLOC   = -2,
    MSMPOT_ERROR_BADPRM  = -3,
    MSMPOT_ERROR_AVAIL   = -4,
    MSMPOT_ERROR_CUDA    = -5,
    MSMPOT_ERROR_UNKNOWN = -6
  };


#if 0

  /*
   * This is optional.  Provide the atom bounding box before each call to
   * Msmpot_compute() to save some setup time.
   */
  int Msmpot_bounds(Msmpot *,
      float xmin, float xmax,       /* min and max values in x-dimension */
      float ymin, float ymax,       /* min and max values in y-dimension */
      float zmin, float zmax        /* min and max values in z-dimension */
      );

  /*
   * Optionally make use of GPUs to accelerate computation.  (Requires
   * building with MSMPOT_CUDA macro defined and having the CUDA kernels
   * compiled and linking to the CUDA runtime libraries.)  The "ngpus"
   * must agree with the number of available CUDA devices and the "gpu"
   * array must be of length "ngpus."  Each element of "gpu" describes
   * device use as follows:
   *
   *   -1 = don't use this device
   *    0 = don't care how this device is used
   *    1 = use this device for long-range part
   *    2 = use this device for short-range part
   *    3 = use this device for both parts
   *
   * The GPU hardware will be used for each subsequent Msmpot_compute().
   * Returns 0 on successful setup or nonzero for failure.
   */
  int Msmpot_usegpu(Msmpot *,
      int ngpus,                    /* number of GPUs */
      const int *gpu                /* how to use each GPU */
      );

  /*
   * Advanced configuration is optional.  Choose MSM parameters to improve
   * either performance or accuracy.  Set any of the parameters to zero to
   * accept default choices.  Returns 0 for success, nonzero for failure.
   *
   *   interp:  0 = cubic, 1 = quintic, 2 = septic, 3 = nonic
   *
   *   split:   0 = C2 Taylor, 1 = C3 Taylor, 2 = C4 Taylor, 3 = C5 Taylor,
   *            4 = C6 Taylor, 5 = C7 Taylor, 6 = C8 Taylor
   */
  int Msmpot_configure(Msmpot *,
      float cutoff,                 /* cutoff for short-range part */
      float hmin,                   /* minimum long-range grid spacing */
      float binsize,                /* size of atom bins for geom hashing */
      int nlevels,                  /* maximum number of levels to use */
      int interp,                   /* interpolant */
      int split                     /* splitting */
      );

  const char *Msmpot_errmsg(Msmpot *);  /* returns the error message */

#endif /* 0 */


#ifdef __cplusplus
}
#endif

#endif /* MSMPOT_H */
