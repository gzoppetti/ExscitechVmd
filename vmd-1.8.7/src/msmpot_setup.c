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
 *      $RCSfile: msmpot_setup.c,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.2 $      $Date: 2009/04/29 15:49:47 $
 *
 ***************************************************************************/
/*
 * setup.c
 */

#include "msmpot_internal.h"


/* called by Msmpot_destroy() */
void Msmpot_cleanup(Msmpot *msm) {
  int i;
  for (i = 0;  i < msm->maxlevels;  i++) {
    Msmpot_lattice_destroy(msm->qh[i]);
    Msmpot_lattice_destroy(msm->eh[i]);
    Msmpot_lattice_destroy(msm->gc[i]);
  }
  free(msm->qh);
  free(msm->eh);
  free(msm->gc);
  free(msm->ezd);
  free(msm->eyzd);
  free(msm->phi_x);
  free(msm->phi_y);
  free(msm->phi_z);
  free(msm->first_atom_index);
  free(msm->next_atom_index);
}


/* called by Msmpot_create() */
void Msmpot_default(Msmpot *msm) {
  /* setup default parameters */
  msm->hminx = msm->hminy = msm->hminz = DEFAULT_HMIN;
  msm->a = DEFAULT_CUTOFF;
  msm->interp = DEFAULT_INTERP;
  msm->split = DEFAULT_SPLIT;
  msm->cellen = DEFAULT_CELLEN;
}


static int setup_mapinterp(Msmpot *msm);
int setup_mapinterp_1Dscaling(Msmpot *msm,
    float hmin,          /* input: minimum MSM lattice spacing */
    float delta,         /* input: map lattice spacing */
    float *ph,           /* determine MSM lattice spacing h */
    float *ph_delta,     /* scaling between maps h/delta */
    int *pp2, int *pp3,  /* scaling represented as powers of 2 and 3 */
    int *pcycle,         /* number of MSM points until next map alignment */
    int *prmap           /* radius of map points about MSM point */
    );
static int setup_mapinterp_1Dcoef(Msmpot *msm,
    int cycle,           /* input: number of MSM points until next map align */
    int rmap,            /* input: radius of map points about MSM point */
    float h_delta,       /* input: scaling between maps h/delta */
    float **p_phi,       /* coefficients that weight map about MSM points */
    int *p_max_phi       /* size of memory allocation */
    );

static int setup_minmax(Msmpot *msm);
static int setup_hierarchy(Msmpot *msm);
static int setup_gridcells(Msmpot *msm);


/* called by Msmpot_compute() */
int Msmpot_setup(Msmpot *msm) {
  int err = 0;

  /* determine map interpolation parameters
   * and MSM lattice spacings hx, hy, hz */
  err = setup_mapinterp(msm);
  if (err) return ERROR(err);

  /* have to find extent of atom positions */
  err = setup_minmax(msm);
  if (err) return ERROR(err);

  /* set up hierarchy of lattices for long-range part */
  err = setup_hierarchy(msm);
  if (err) return ERROR(err);

  /* set up grid cell hashing for short-range part */
  err = setup_gridcells(msm);
  if (err) return ERROR(err);

#ifdef MSMPOT_CUDA
  /* set up CUDA device */
  err = Msmpot_cuda_setup(msm->msmcuda, msm);
  if (err) return ERROR(err);
#endif

  return OK;
}


typedef struct InterpParams_t {
  int nu;
  int stencil;
  int omega;
} InterpParams;

static InterpParams INTERP_PARAMS[] = {
  { 0, 0, 0 },    /* not set */
  { 1, 4, 6 },    /* cubic */
  { 2, 6, 10 },   /* quintic */
  { 2, 6, 10 },   /* quintic, C2 */
};


int setup_mapinterp(Msmpot *msm) {
  long mymz = msm->my * msm->mz;
  int err = 0;

  ASSERT(msm->mx > 0);
  ASSERT(msm->my > 0);
  ASSERT(msm->mz > 0);
  if (msm->max_eyzd < mymz) {
    float *t;
    t = (float *) realloc(msm->eyzd, mymz * sizeof(float));
    if (NULL == t) return ERROR(MSMPOT_ERROR_ALLOC);
    msm->eyzd = t;
    msm->max_eyzd = mymz;
  }
  if (msm->max_ezd < msm->mz) {
    float *t;
    t = (float *) realloc(msm->ezd, msm->mz * sizeof(float));
    if (NULL == t) return ERROR(MSMPOT_ERROR_ALLOC);
    msm->ezd = t;
    msm->max_ezd = msm->mz;
  }

  err |= setup_mapinterp_1Dscaling(msm,
      msm->hminx, msm->dx, &(msm->hx), &(msm->hx_dx),
      &(msm->px2), &(msm->px3), &(msm->cycle_x), &(msm->rmap_x));
  err |= setup_mapinterp_1Dscaling(msm,
      msm->hminy, msm->dy, &(msm->hy), &(msm->hy_dy),
      &(msm->py2), &(msm->py3), &(msm->cycle_y), &(msm->rmap_y));
  err |= setup_mapinterp_1Dscaling(msm,
      msm->hminz, msm->dz, &(msm->hz), &(msm->hz_dz),
      &(msm->pz2), &(msm->pz3), &(msm->cycle_z), &(msm->rmap_z));
  if (err) return ERROR(err);

  err |= setup_mapinterp_1Dcoef(msm, msm->cycle_x, msm->rmap_x, msm->hx_dx,
      &(msm->phi_x), &(msm->max_phi_x));
  err |= setup_mapinterp_1Dcoef(msm, msm->cycle_y, msm->rmap_y, msm->hy_dy,
      &(msm->phi_y), &(msm->max_phi_y));
  err |= setup_mapinterp_1Dcoef(msm, msm->cycle_z, msm->rmap_z, msm->hz_dz,
      &(msm->phi_z), &(msm->max_phi_z));
  if (err) return ERROR(err);

  return OK;
}


int setup_mapinterp_1Dcoef(Msmpot *msm,
    int cycle,           /* input: number of MSM points until next map align */
    int rmap,            /* input: radius of map points about MSM point */
    float h_delta,       /* input: scaling between maps h/delta */
    float **p_phi,       /* coefficients that weight map about MSM points */
    int *p_max_phi       /* size of memory allocation */
    ) {
  float *phi = NULL;
  const int diam = 2*rmap + 1;
  const int nphi = cycle * diam;
  const float delta_h = 1 / h_delta;
  float t;
  int i, k;

  if (*p_max_phi < nphi) {  /* allocate more memory if we need it */
    phi = (float *) realloc(*p_phi, nphi * sizeof(float));
    if (NULL == phi) return ERROR(MSMPOT_ERROR_ALLOC);
    *p_phi = phi;
    *p_max_phi = nphi;
  }
  ASSERT(*p_phi != NULL);

  for (k = 0;  k < cycle;  k++) {
    phi = *p_phi + k * diam + rmap;  /* center of weights for this cycle */
    switch (msm->interp) {
      case MSMPOT_CUBIC_INTERP:
        for (i = -rmap;  i <= rmap;  i++) {
          t = fabsf(i * delta_h);
          if (t <= 1) {
            phi[i] = (1 - t) * (1 + t - 1.5f * t * t);
          }
          else if (t <= 2) {
            phi[i] = 0.5f * (1 - t) * (2 - t) * (2 - t);
          }
          else {
            phi[i] = 0;
          }
        }
        break;
      case MSMPOT_QUINTIC_INTERP:
        for (i = -rmap;  i <= rmap;  i++) {
          t = fabsf(i * delta_h);
          if (t <= 1) {
            phi[i] = (1-t*t) * (2-t) * (0.5f + t * (0.25f - (5.f/12)*t));
          }
          else if (t <= 2) {
            phi[i] = (1-t)*(2-t)*(3-t) * ((1.f/6) + t*(0.375f - (5.f/24)*t));
          }
          else if (t <= 3) {
            phi[i] = (1.f/24) * (1-t) * (2-t) * (3-t) * (3-t) * (4-t);
          }
          else {
            phi[i] = 0;
          }
        }
        break;
      default:
        return ERROR(MSMPOT_ERROR_BADPRM);
    } /* end switch on interp */
  } /* end loop k over cycles */
  return OK;
}


int setup_mapinterp_1Dscaling(Msmpot *msm,
    float hmin,          /* input: minimum MSM lattice spacing */
    float delta,         /* input: map lattice spacing */
    float *ph,           /* determine MSM lattice spacing h */
    float *ph_delta,     /* scaling between maps h/delta */
    int *pp2, int *pp3,  /* scaling represented as powers of 2 and 3 */
    int *pcycle,         /* number of MSM points until next map alignment */
    int *prmap           /* radius of map points about MSM point */
    ) {
  float h = delta; /* determine h in [hmin,hmax] as special scaling of delta */
  int p2 = 0;      /* count powers of 2 */
  int p3 = 0;      /* count powers of 3 */
  const int nu = INTERP_PARAMS[msm->interp].nu;
  float h_delta;

  ASSERT(delta > 0.f);
  ASSERT(hmin > 0.f);
  while (h < hmin) {
    if (h > 0.75f * hmin) {
      p2--;
      p3++;
      h *= 1.5f;
    }
    else {
      p2++;
      h *= 2.f;
    }
  }
  while (h > 1.5f * hmin) {  /* hmax == 1.5f * hmin */
    if (h < 2.f * hmin) {
      p2 -= 2;
      p3++;
      h *= 0.75f;
    }
    else {
      p2--;
      h *= 0.5f;
    }
  }
  if (p2 > 30 || p2 < -30) return ERROR(MSMPOT_ERROR_BADPRM);
  ASSERT(hmin <= h && h <= 1.5f * hmin);
  ASSERT(p3==0 || p3==1);
  *pp2 = p2;
  *pp3 = p3;
  *ph = h;

  /* calculate exact h/delta, it's a machine number */
  h_delta = (p2 < 0 ? 1.f / (1 << -p2) : (float) (1 << p2));
  h_delta *= (p3 ? 3 : 1);
  *ph_delta = h_delta;

  *pcycle = (p2 < 0 ? (1 << -p2) : 1);
  *prmap = (int) ceilf(h_delta * (nu + 1)) - 1;
  return OK;
}


int setup_minmax(Msmpot *msm) {
  const long natoms = msm->natoms;
  long n;
  const float *atom = msm->atom;
  float xmin, xmax, ymin, ymax, zmin, zmax;
  float x, y, z;

  ASSERT(natoms > 0);
  xmin = xmax = atom[0];
  ymin = ymax = atom[1];
  zmin = zmax = atom[2];
  for (n = 1;  n < natoms;  n++) {
    x = atom[ 4*n + 0 ];
    y = atom[ 4*n + 1 ];
    z = atom[ 4*n + 2 ];
    if (xmin > x)      xmin = x;
    else if (xmax < x) xmax = x;
    if (ymin > y)      ymin = y;
    else if (ymax < y) ymax = y;
    if (zmin > z)      zmin = z;
    else if (zmax < z) zmax = z;
  }
  msm->xmin = xmin;
  msm->xmax = xmax;
  msm->ymin = ymin;
  msm->ymax = ymax;
  msm->zmin = zmin;
  msm->zmax = zmax;
  return OK;
}


int setup_gridcells(Msmpot *msm) {
  const long natoms = msm->natoms;
  long ncells;
  long *a;

  msm->inv_cellen = 1/msm->cellen;

  /* find grid cell dimensions */
  msm->nxcells = (long) floorf((msm->xmax - msm->xmin) *
      msm->inv_cellen) + 1;
  msm->nycells = (long) floorf((msm->ymax - msm->ymin) *
      msm->inv_cellen) + 1;
  msm->nzcells = (long) floorf((msm->zmax - msm->zmin) *
      msm->inv_cellen) + 1;

  /* allocate and initialize array of first atom index for each cell,
   * length is number of grid cells */
  ncells = msm->nxcells * msm->nycells * msm->nzcells;
  ASSERT(ncells > 0);
  if (msm->maxcells < ncells) {
    a = (long *) realloc(msm->first_atom_index, ncells * sizeof(long));
    if (NULL == a) return ERROR(MSMPOT_ERROR_ALLOC);
    msm->first_atom_index = a;
    msm->maxcells = ncells;
  }
  else {
    a = msm->first_atom_index;
    ASSERT(a != NULL);
  }
  msm->ncells = ncells;

  /* allocate and initialize array of next atom index for each cell,
   * length is number of atoms */
  ASSERT(natoms > 0);
  if (msm->maxatoms < natoms) {
    a = (long *) realloc(msm->next_atom_index, natoms * sizeof(long));
    if (NULL == a) return ERROR(MSMPOT_ERROR_ALLOC);
    msm->next_atom_index = a;
    msm->maxatoms = natoms;
  }
  else {
    a = msm->next_atom_index;
    ASSERT(a != NULL);
  }

  return OK;
}


int setup_hierarchy(Msmpot *msm) {
  const int nu = INTERP_PARAMS[msm->interp].nu;
  const int omega = INTERP_PARAMS[msm->interp].omega;
  const int split = msm->split;
  int level, maxlevels;
  int err = 0;

  const float a = msm->a;
  const float hx = msm->hx;
  const float hy = msm->hy;
  const float hz = msm->hz;

  /* maximum extent of epotmap */
  float xm1 = msm->xm0 + msm->dx * (msm->mx - 1);
  float ym1 = msm->ym0 + msm->dy * (msm->my - 1);
  float zm1 = msm->zm0 + msm->dz * (msm->mz - 1);

  /* smallest possible extent of finest spaced MSM lattice */
  float xlo = (msm->xmin < msm->xm0 ? msm->xmin : msm->xm0);
  float ylo = (msm->ymin < msm->ym0 ? msm->ymin : msm->ym0);
  float zlo = (msm->zmin < msm->zm0 ? msm->zmin : msm->zm0);
  float xhi = (msm->xmax > xm1 ? msm->xmax : xm1);
  float yhi = (msm->ymax > ym1 ? msm->ymax : ym1);
  float zhi = (msm->zmax > zm1 ? msm->zmax : zm1);

  /* indexes for MSM lattice */
  long ia = ((long) floorf((xlo - msm->xm0) / hx)) - nu;
  long ja = ((long) floorf((ylo - msm->ym0) / hy)) - nu;
  long ka = ((long) floorf((zlo - msm->zm0) / hz)) - nu;
  long ib = ((long) floorf((xhi - msm->xm0) / hx)) + 1 + nu;
  long jb = ((long) floorf((yhi - msm->ym0) / hy)) + 1 + nu;
  long kb = ((long) floorf((zhi - msm->zm0) / hz)) + 1 + nu;
  long ni = ib - ia + 1;
  long nj = jb - ja + 1;
  long nk = kb - ka + 1;

  long omega3 = omega * omega * omega;
  long nhalf = (long) sqrtf(ni * nj * nk);
  long lastnelems = (nhalf > omega3 ? nhalf : omega3);
  long nelems, n;
  long i, j, k;

  MsmpotLattice *p = NULL;
  float scaling;

  n = ni;
  if (n < nj) n = nj;
  if (n < nk) n = nk;
  for (maxlevels = 1;  n > 0;  n >>= 1)  maxlevels++;
  if (msm->maxlevels < maxlevels) {
    MsmpotLattice **t;
    t = (MsmpotLattice **) realloc(msm->qh, maxlevels*sizeof(MsmpotLattice *));
    if (NULL == t) return ERROR(MSMPOT_ERROR_ALLOC);
    msm->qh = t;
    t = (MsmpotLattice **) realloc(msm->eh, maxlevels*sizeof(MsmpotLattice *));
    if (NULL == t) return ERROR(MSMPOT_ERROR_ALLOC);
    msm->eh = t;
    t = (MsmpotLattice **) realloc(msm->gc, maxlevels*sizeof(MsmpotLattice *));
    if (NULL == t) return ERROR(MSMPOT_ERROR_ALLOC);
    msm->gc = t;
    for (level = msm->maxlevels;  level < maxlevels;  level++) {
      msm->qh[level] = Msmpot_lattice_create();
      if (NULL == msm->qh[level]) return ERROR(MSMPOT_ERROR_ALLOC);
      msm->eh[level] = Msmpot_lattice_create();
      if (NULL == msm->eh[level]) return ERROR(MSMPOT_ERROR_ALLOC);
      msm->gc[level] = Msmpot_lattice_create();
      if (NULL == msm->gc[level]) return ERROR(MSMPOT_ERROR_ALLOC);
    }
    msm->maxlevels = maxlevels;
  }

  level = 0;
  do {
    err = Msmpot_lattice_setup(msm->qh[level], ia, ib, ja, jb, ka, kb);
    if (err) return ERROR(err);
    err = Msmpot_lattice_setup(msm->eh[level], ia, ib, ja, jb, ka, kb);
    if (err) return ERROR(err);
    nelems = ni * nj * nk;
    ia = -((-ia+1)/2) - nu;
    ja = -((-ja+1)/2) - nu;
    ka = -((-ka+1)/2) - nu;
    ib = (ib+1)/2 + nu;
    jb = (jb+1)/2 + nu;
    kb = (kb+1)/2 + nu;
    ni = ib - ia + 1;
    nj = jb - ja + 1;
    nk = kb - ka + 1;
    level++;
  } while (nelems > lastnelems);
  msm->nlevels = level;

  /* ellipsoid axes for lattice cutoff weights */
  ni = (long) ceilf(2*a/hx) - 1;
  nj = (long) ceilf(2*a/hy) - 1;
  nk = (long) ceilf(2*a/hz) - 1;
  scaling = 1;
  for (level = 0;  level < msm->nlevels - 1;  level++) {
    p = msm->gc[level];
    err = Msmpot_lattice_setup(p, -ni, ni, -nj, nj, -nk, nk);
    if (err) return ERROR(err);
    for (k = -nk;  k <= nk;  k++) {
      for (j = -nj;  j <= nj;  j++) {
        for (i = -ni;  i <= ni;  i++) {
          float s, t, gs, gt, g;
          s = ( (i*hx)*(i*hx) + (j*hy)*(j*hy) + (k*hz)*(k*hz) ) / (a*a);
          t = 0.25f * s;
          if (t >= 1) {
            g = 0;
          }
          else if (s >= 1) {
            gs = 1/sqrtf(s);
            SPOLY(&gt, t, split);
            g = scaling * (gs - 0.5f * gt) / a;
          }
          else {
            SPOLY(&gs, s, split);
            SPOLY(&gt, t, split);
            g = scaling * (gs - 0.5f * gt) / a;
          }
          RANGE_CHECK(p, i, j, k);
          *ELEM(p, i, j, k) = g;
        }
      }
    } /* end loops over k-j-i */
    scaling *= 0.5f;
  } /* end loop over levels */

  /* calculate coarsest level weights, ellipsoid axes are length of lattice */
  ni = (msm->qh[level])->ib - (msm->qh[level])->ia;
  nj = (msm->qh[level])->jb - (msm->qh[level])->ja;
  nk = (msm->qh[level])->kb - (msm->qh[level])->ka;
  p = msm->gc[level];
  err = Msmpot_lattice_setup(p, -ni, ni, -nj, nj, -nk, nk);
  for (k = -nk;  k <= nk;  k++) {
    for (j = -nj;  j <= nj;  j++) {
      for (i = -ni;  i <= ni;  i++) {
        float s, gs;
        s = ( (i*hx)*(i*hx) + (j*hy)*(j*hy) + (k*hz)*(k*hz) ) / (a*a);
        if (s >= 1) {
          gs = 1/sqrtf(s);
        }
        else {
          SPOLY(&gs, s, split);
        }
        RANGE_CHECK(p, i, j, k);
        *ELEM(p, i, j, k) = scaling * gs/a;
      }
    }
  } /* end loops over k-j-i for coarsest level weights */

  return OK;
}
