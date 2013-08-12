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
 *      $RCSfile: msmpot_internal.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.3 $      $Date: 2009/04/29 15:49:47 $
 *
 ***************************************************************************/
/*
 * msmpot_internal.h
 */


#ifndef MSMPOT_INTERNAL_H
#define MSMPOT_INTERNAL_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "msmpot.h"
#ifdef MSMPOT_CUDA
#include "msmpot_cuda.h"
#endif

/* avoid parameter name collisions with AIX5 "hz" macro */
#undef hz

#ifdef __cplusplus
extern "C" {
#endif


  /* default parameters */
#undef  DEFAULT_HMIN
#define DEFAULT_HMIN     2.f

#undef  DEFAULT_CUTOFF
#define DEFAULT_CUTOFF  12.f

#undef  DEFAULT_INTERP
#define DEFAULT_INTERP  MSMPOT_CUBIC_INTERP

#undef  DEFAULT_SPLIT
#define DEFAULT_SPLIT   MSMPOT_TAYLOR2_SPLIT

#undef  DEFAULT_CELLEN
#define DEFAULT_CELLEN   4.f


  /*** MsmpotLattice *********************************************************/

  typedef struct MsmpotLattice_t {
    long nbufsz;    /* total size of buffer allocation */
    long ia, ib;    /* index ia <= i <= ib */
    long ja, jb;    /* index ja <= j <= jb */
    long ka, kb;    /* index ka <= k <= kb */
    long ni, nj;    /* ni == ib-ia+1, nj == jb-ja+1 */
    float *buffer;  /* allocated buffer */
    float *data;    /* pointer shifted by ia, ja, ka */
  } MsmpotLattice;

  MsmpotLattice *Msmpot_lattice_create(void);
  void Msmpot_lattice_destroy(MsmpotLattice *);

  int Msmpot_lattice_setup(MsmpotLattice *,
      long ia, long ib, long ja, long jb, long ka, long kb);

  int Msmpot_lattice_zero(MsmpotLattice *);

  /* calculate index into flat data array */
#undef  INDEX
#define INDEX(p,i,j,k)  (((k) * p->nj + (j)) * p->ni + (i))

  /* point to lattice element indexed by i,j,k */
#undef  ELEM
#define ELEM(p,i,j,k)   (p->data + INDEX(p,i,j,k))

#undef  RANGE_CHECK
#define RANGE_CHECK(p,i,j,k) \
  ASSERT(p->ia <= (i) && (i) <= p->ib); \
  ASSERT(p->ja <= (j) && (j) <= p->jb); \
  ASSERT(p->ka <= (k) && (k) <= p->kb)


  /*** Msmpot ****************************************************************/

  struct Msmpot_t {
    const float *atom;    /* point to caller's atom array */
    long natoms;          /* store number of atoms */

    float xmin, xmax, ymin, ymax, zmin, zmax;  /* bounding box for atoms */

    float *epotmap;
    long mx, my, mz;
    float dx, dy, dz;
    float lx, ly, lz;
    float xm0, ym0, zm0;  /* origin of epotmap */

    /* fundamental MSM parameters:
     * cutoff "a" is set to 12A by default, good for atomic systems;
     * lattice spacings hx, hy, hz are determined to be powers of 2
     * scalings of dx, dy, dz while maintaining 4 <= a/hx, a/hy, a/hz <= 6
     * for sufficient accuracy;
     * for nonperiodic dimensions, the finest level lattice is chosen to be
     * smallest size aligned with epotmap containing both atoms and epotmap;
     * for periodic dimensions, the finest level lattice fits within cell
     * defined by epotmap parameters above;
     * the number of levels is determined to reduce coarsest level lattice
     * to be as small as possible for the given boundary conditions */

    float hminx, hminy, hminz;  /* default 2A, can be set by user;
                                 * hmaxx is (1.5 * hminx), etc. */

    float a;              /* default 12A finest level cutoff, can be set */
    float hx, hy, hz;     /* the finest level lattice spacings */

    int interp;           /* ID for the interpolant */
    int split;            /* ID for the splitting */

    int nlevels;          /* number of lattice levels */

    /* lattices for calculating long-range part:
     * the finest level lattice is 0, e.g. q0 = qh[0];
     * positions of lattice points are determined by power-of-2 scalings
     * of hx, hy, hz and translated by finest level lattice origin,
     * e.g. q0[i,j,k] is at position (i*hx - xl0, j*hy - yl0, k*hz - zl0);
     * note that lattice indexes can be negative */

    MsmpotLattice **qh;   /* lattices of charge, 0==finest */
    MsmpotLattice **eh;   /* lattices of potential, 0==finest */
    MsmpotLattice **gc;   /* bounded-size lattices of weights for each level */
    int maxlevels;        /* alloc length of (Lattice *) arrays >= nlevels */

    int ispx, ispy, ispz; /* is periodic in x, y, z? */

    /* Interpolating from finest lattice to epotmap:
     * want ratio hx/dx to be rational 2^(px2)*3^(px3),
     * where px2 is unrestricted and px3=0 or px3=1.
     * The interpolation of epotmap from finest lattice then has
     * a fixed cycle of coefficients that can be precomputed.
     * The calculation steps through MSM lattice points and
     * adds their contribution to surrounding epotmap points. */

    int px2, py2, pz2;    /* powers of 2 */
    int px3, py3, pz3;    /* powers of 3 */
    float hx_dx, hy_dy, hz_dz;  /* scaling is integer for px2 >= 0 */

    int cycle_x, cycle_y, cycle_z;  /* counts MSM points between alignment */
    int rmap_x, rmap_y, rmap_z;     /* radius of map points about MSM point */

    int max_phi_x, max_phi_y, max_phi_z;  /* alloc length of phi arrays */
    float *phi_x;         /* coefficients, size cycle_x * (2*rmap_x + 1) */
    float *phi_y;         /* coefficients, size cycle_y * (2*rmap_y + 1) */
    float *phi_z;         /* coefficients, size cycle_z * (2*rmap_z + 1) */

    long max_ezd, max_eyzd;  /* alloc length of interp temp buffer space */
    float *ezd;           /* interpolation temp row buffer, length mz */
    float *eyzd;          /* interpolation temp plane buffer, length my*mz */

    /* grid cell hashing to calculate short-range part:
     * use cursor linked list implementation with two arrays, in which the
     * "first_atom_index" array is the "first" atom in that cell's list
     * (where -1 indicates an empty cell), and each atom through
     * "next_atom_index" points to the next atom in that cell's list
     * (where -1 indicates the end of the list) */

    float cellen;         /* dimension of grid cells, set for performance */
    float inv_cellen;     /* 1/cellen */

    long nxcells, nycells, nzcells;    /* dimensions of grid cell rectangle */
    long ncells;                       /* total number of grid cells */
    long maxcells;                     /* allocated number of grid cells */
    long maxatoms;                     /* allocated number of atoms */
    long *first_atom_index;            /* length maxcells >= ncells */
    long *next_atom_index;             /* length maxatoms >= natoms */

#ifdef MSMPOT_CUDA
    MsmpotCuda *msmcuda;
    int use_cuda_shortrng;
    int use_cuda_latcut;
#endif
  };


  /* for internal use only */
  void Msmpot_cleanup(Msmpot *msm);
  void Msmpot_default(Msmpot *msm);
  int Msmpot_setup(Msmpot *msm);

  int Msmpot_compute_shortrng(Msmpot *msm, const float *atom, long natoms);
  int Msmpot_compute_longrng(Msmpot *msm);

  int Msmpot_compute_longrng_cubic(Msmpot *msm);

  /* function return values:
   * return OK for success or MSMPOT_ERROR_* value for error */
#undef  OK
#define OK  0


  /* exception handling:
   * MSMPOT_DEBUG turns on error reporting to stderr stream, 
   * in any case propagate error number back up the call stack */
#undef  ERROR
#ifndef MSMPOT_DEBUG
#define ERROR(err)     (err)
#define ERRMSG(err,s)  (err)
#else
  /* report error to stderr stream, return "err" */
  int Msmpot_report_error(int err, const char *msg, const char *fn, int ln);
#define ERROR(err)     Msmpot_report_error(err, NULL, __FILE__, __LINE__);
#define ERRMSG(err,s)  Msmpot_report_error(err, s, __FILE__, __LINE__);
#endif


  /* check assertions when debugging, raise exception if failure */
#ifndef MSMPOT_DEBUG
#define ASSERT(expr)
#else
#define ASSERT(expr) \
  do { \
    if ( !(expr) ) { \
      return ERRMSG(MSMPOT_ERROR_ASSERT, #expr); \
    } \
  } while (0)
#endif


  /* SPOLY() calculates the polynomial part of the
   * normalized smoothing of 1/r, i.e. g_1((r/a)**2).
   *
   *   pg - float*, points to variable to receive the result
   *   s - (ra/)**2, assumed to be between 0 and 1
   *   split - identify the type of smoothing used to split the potential */
#undef  SPOLY
#define SPOLY(pg, s, split) \
  do { \
    double _s = s;  /* where s=(r/a)**2 */ \
    double _g = 0; \
    ASSERT(0 <= _s && _s <= 1); \
    switch (split) { \
      case MSMPOT_TAYLOR2_SPLIT: \
        _g = 1 + (_s-1)*(-1./2 + (_s-1)*(3./8)); \
        break; \
      case MSMPOT_TAYLOR3_SPLIT: \
        _g = 1 + (_s-1)*(-1./2 + (_s-1)*(3./8 + (_s-1)*(-5./16))); \
        break; \
      case MSMPOT_TAYLOR4_SPLIT: \
        _g = 1 + (_s-1)*(-1./2 + (_s-1)*(3./8 + (_s-1)*(-5./16 \
                + (_s-1)*(35./128)))); \
        break; \
      default: \
        return ERROR(MSMPOT_ERROR_BADPRM); \
    } \
    *(pg) = _g; \
  } while (0)
  /* closing ';' from use as function call */


#ifdef __cplusplus
}
#endif

#endif /* MSMPOT_INTERNAL_H */
