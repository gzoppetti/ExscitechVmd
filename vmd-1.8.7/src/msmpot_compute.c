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
 *      $RCSfile: msmpot_compute.c,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.2 $      $Date: 2009/04/29 15:49:47 $
 *
 ***************************************************************************/
/*
 * msmpot_compute.c
 */

#include "msmpot_internal.h"


static int check_params(Msmpot *msm, const float *epotmap,
    long mx, long my, long mz, float dx, float dy, float dz,
    float lx, float ly, float lz, const float *atom, long natoms);


int Msmpot_compute(Msmpot *msm,
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
    ) {

  int err = 0;

  err = check_params(msm, epotmap, mx, my, mz, dx, dy, dz, lx, ly, lz,
      atom, natoms);
  if (err) return ERROR(err);

  /* store user parameters */
  msm->atom = atom;
  msm->natoms = natoms;
  msm->epotmap = epotmap;
  msm->mx = mx;
  msm->my = my;
  msm->mz = mz;
  msm->dx = dx;
  msm->dy = dy;
  msm->dz = dz;
  msm->lx = lx;
  msm->ly = ly;
  msm->lz = lz;
  msm->xm0 = x0;
  msm->ym0 = y0;
  msm->zm0 = z0;

  err = Msmpot_setup(msm);
  if (err) return ERROR(err);

  memset(epotmap, 0, mx*my*mz*sizeof(float));  /* clear epotmap */

#ifdef MSMPOT_CUDA
  if (msm->use_cuda_shortrng) {
    err = Msmpot_cuda_compute_shortrng(msm->msmcuda);
    if (err) {  /* fall back on CPU version if any problems using CUDA */
      err = Msmpot_compute_shortrng(msm, msm->atom, msm->natoms);
      if (err) return ERROR(err);
    }
  }
  else {
    err = Msmpot_compute_shortrng(msm, msm->atom, msm->natoms);
    if (err) return ERROR(err);
  }
#else
  err = Msmpot_compute_shortrng(msm, msm->atom, msm->natoms);
  if (err) return ERROR(err);
#endif

  err = Msmpot_compute_longrng(msm);
  if (err) return ERROR(err);

  return OK;
}


int check_params(Msmpot *msm, const float *epotmap, long mx, long my, long mz,
    float dx, float dy, float dz, float lx, float ly, float lz,
    const float *atom, long natoms) {

  if (NULL == epotmap || NULL == atom || natoms <= 0) {
    return ERROR(MSMPOT_ERROR_BADPRM);
  }
  else if (dx < 0 || dy < 0 || dz < 0 || lx < 0 || ly < 0 || lz < 0) {
    return ERROR(MSMPOT_ERROR_BADPRM);
  }
  else if (mx <= 0 || my <= 0 || mz <= 0) {
    return ERROR(MSMPOT_ERROR_BADPRM);
  }

  if      (dx > 0 && 0 == lx)  msm->ispx = 0;
  else if (lx > 0 && 0 == dx)  msm->ispx = 1;
  else    return ERROR(MSMPOT_ERROR_BADPRM);

  if      (dy > 0 && 0 == ly)  msm->ispy = 0;
  else if (ly > 0 && 0 == dy)  msm->ispy = 1;
  else    return ERROR(MSMPOT_ERROR_BADPRM);

  if      (dz > 0 && 0 == lz)  msm->ispz = 0;
  else if (lz > 0 && 0 == dz)  msm->ispz = 1;
  else    return ERROR(MSMPOT_ERROR_BADPRM);

  if (msm->ispx) {
    long m = mx;
    while (m > 3) {
      if (m & 1) return ERROR(MSMPOT_ERROR_BADPRM);  /* can't be odd > 3 */
      m >>= 1;   /* divide by 2 */
    }
  }

  if (msm->ispy) {
    long m = my;
    while (m > 3) {
      if (m & 1) return ERROR(MSMPOT_ERROR_BADPRM);  /* can't be odd > 3 */
      m >>= 1;   /* divide by 2 */
    }
  }

  if (msm->ispz) {
    long m = mz;
    while (m > 3) {
      if (m & 1) return ERROR(MSMPOT_ERROR_BADPRM);  /* can't be odd > 3 */
      m >>= 1;   /* divide by 2 */
    }
  }

  /* can't handle PBCs yet */
  if (msm->ispx || msm->ispy || msm->ispz) {
    return ERROR(MSMPOT_ERROR_BADPRM);
  }

  return OK;
}



/*** long-range part *********************************************************/


int Msmpot_compute_longrng(Msmpot *msm) {
  int err = 0;

  /* permit only cubic interpolation - for now */
  if (msm->interp != MSMPOT_CUBIC_INTERP) {
    return ERROR(MSMPOT_ERROR_BADPRM);
  }

  err = Msmpot_compute_longrng_cubic(msm);
  if (err) return ERROR(err);

  return OK;
}



/*** short-range part ********************************************************/


static int geometric_hashing(Msmpot *msm, const float *atom, long natoms);
static int generic_split(Msmpot *msm, const float *atom);


int Msmpot_compute_shortrng(Msmpot *msm, const float *atom, long natoms) {
  int err = 0;

  err = geometric_hashing(msm, atom, natoms);
  if (err) return ERROR(err);

  /* fall back on generic version */
  err = generic_split(msm, atom);
  if (err) return ERROR(err);

  return OK;
}


int geometric_hashing(Msmpot *msm, const float *atom, long natoms) {
  long i, j, k;
  long n;   /* index atoms */
  long nc;  /* index grid cells */
  const float xmin = msm->xmin;
  const float ymin = msm->ymin;
  const float zmin = msm->zmin;
  const float inv_cellen = msm->inv_cellen;
  float x, y, z;  /* atom position relative to (xmin,ymin,zmin) */
  float q;        /* atom charge */
  long *first = msm->first_atom_index;  /* first index in grid cell list */
  long *next = msm->next_atom_index;    /* next index in grid cell list */
  const long nxcells = msm->nxcells;
  const long nycells = msm->nycells;
  const long ncells = msm->ncells;

  /* must clear gridcells and next links before we hash */
  for (nc = 0;  nc < ncells;  nc++)  first[nc] = -1;
  for (n = 0;  n < natoms;  n++)  next[n] = -1;

  for (n = 0;  n < natoms;  n++) {

    /* atoms with zero charge make no contribution */
    q = atom[4*n + 3];
    if (0==q) continue;

    x = atom[4*n    ] - xmin;
    y = atom[4*n + 1] - ymin;
    z = atom[4*n + 2] - zmin;
    i = (long) floorf(x * inv_cellen);
    j = (long) floorf(y * inv_cellen);
    k = (long) floorf(z * inv_cellen);
    nc = (k*nycells + j)*nxcells + i;
    next[n] = first[nc];
    first[nc] = n;
  }
  return OK;
}


/* must perform geometric hashing first */
int generic_split(Msmpot *msm, const float *atom) {

  const float xm0 = msm->xm0;  /* epotmap origin */
  const float ym0 = msm->ym0;
  const float zm0 = msm->zm0;

  const float dx = msm->dx;    /* epotmap spacings */
  const float dy = msm->dy;
  const float dz = msm->dz;
  const float inv_dx = 1/dx;
  const float inv_dy = 1/dy;
  const float inv_dz = 1/dz;

  const float a = msm->a;      /* cutoff for splitting */
  const float a2 = a*a;
  const float a_1 = 1/a;
  const float inv_a2 = a_1 * a_1;

  float x, y, z;     /* position of atom relative to epotmap origin */
  float q;           /* charge on atom */

  float xstart, ystart;  /* start of rx and ry */

  long i, j, k;
  long ic, jc, kc;   /* closest map point less than or equal to atom */
  long ia, ib;       /* extent of surrounding box in x-direction */
  long ja, jb;       /* extent of surrounding box in y-direction */
  long ka, kb;       /* extent of surrounding box in z-direction */
  long n;            /* index atoms */
  long nc;           /* index grid cells */
  long index;        /* index into epotmap */
  long koff, jkoff;  /* tabulate strides into epotmap */
  float rx, ry, rz;  /* distances between an atom and a map point */
  float rz2, ryrz2;  /* squared circle and cylinder distances */
  float r2;          /* squared pairwise distance */
  float s;           /* normalized distance squared */
  float gs;          /* result of normalized splitting */
  float e;           /* contribution to short-range potential */

  const int split = msm->split;

  const long mx = msm->mx;  /* lengths of epotmap lattice */
  const long my = msm->my;
  const long mz = msm->mz;

  const long mri = (long) ceilf(a * inv_dx) - 1;
  const long mrj = (long) ceilf(a * inv_dy) - 1;
  const long mrk = (long) ceilf(a * inv_dz) - 1;
                     /* lengths (measured in points) of ellipsoid axes */

  const long ncells = msm->ncells;
  const long *first = msm->first_atom_index;
  const long *next = msm->next_atom_index;

  float *epotmap = msm->epotmap;
  float *pem = NULL;        /* point into epotmap */

  for (nc = 0;  nc < ncells;  nc++) {
    for (n = first[nc];  n != -1;  n = next[n]) {

      /* position of atom relative to epotmap origin */
      x = atom[4*n    ] - xm0;
      y = atom[4*n + 1] - ym0;
      z = atom[4*n + 2] - zm0;

      /* charge on atom */
      q = atom[4*n + 3];

      /* find closest map point with position less than or equal to atom */
      ic = (long) floorf(x * inv_dx);
      jc = (long) floorf(y * inv_dy);
      kc = (long) floorf(z * inv_dz);

      /* find extent of surrounding box of map points */
      ia = ic - mri;
      ib = ic + mri + 1;
      ja = jc - mrj;
      jb = jc + mrj + 1;
      ka = kc - mrk;
      kb = kc + mrk + 1;

      /* trim box edges to be within map */
      if (ia < 0)   ia = 0;
      if (ib >= mx) ib = mx - 1;
      if (ja < 0)   ja = 0;
      if (jb >= my) jb = my - 1;
      if (ka < 0)   ka = 0;
      if (kb >= mz) kb = mz - 1;

      /* loop over surrounding map points, add contribution into epotmap */
      xstart = ia*dx - x;
      ystart = ja*dy - y;
      rz = ka*dz - z;
      for (k = ka;  k <= kb;  k++, rz += dz) {
        koff = k*my;
        rz2 = rz*rz;
#ifdef MSMPOT_CHECK_CIRCLE_CPU
        /* clipping to the circle makes it slower */
        if (rz2 >= a2) continue;
#endif
        ry = ystart;
        for (j = ja;  j <= jb;  j++, ry += dy) {
          jkoff = (koff + j)*mx;
          ryrz2 = ry*ry + rz2;
#ifdef MSMPOT_CHECK_CYLINDER_CPU
          /* clipping to the cylinder is faster */
          if (ryrz2 >= a2) continue;
#endif
          rx = xstart;
          index = jkoff + ia;
          pem = epotmap + index;
#if 0
#if defined(__INTEL_COMPILER)
          for (i = ia;  i <= ib;  i++, pem++, rx += dx) {
            r2 = rx*rx + ryrz2;
            s = r2 * inv_a2;
            gs = 1.875f + s*(-1.25f + s*0.375f);  /* TAYLOR2 */
            e = q * (1/sqrtf(r2) - a_1 * gs);
            *pem += (r2 < a2 ? e : 0);  /* LOOP VECTORIZED! */
          }
#else
          for (i = ia;  i <= ib;  i++, pem++, rx += dx) {
            r2 = rx*rx + ryrz2;
            if (r2 >= a2) continue;
            s = r2 * inv_a2;
            gs = 1.875f + s*(-1.25f + s*0.375f);  /* TAYLOR2 */
            e = q * (1/sqrtf(r2) - a_1 * gs);
            *pem += e;
          }
#endif
#else
          for (i = ia;  i <= ib;  i++, pem++, rx += dx) {
            r2 = rx*rx + ryrz2;
            if (r2 >= a2) continue;
            s = r2 * inv_a2;
            SPOLY(&gs, s, split);  /* macro expands into switch */
            e = q * (1/sqrtf(r2) - a_1 * gs);
            *pem += e;
          }
#endif

        }
      } /* end loop over surrounding map points */

    } /* end loop over atoms in grid cell */
  } /* end loop over grid cells */

  return OK;
}

