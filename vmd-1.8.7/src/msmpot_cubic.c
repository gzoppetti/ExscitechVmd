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
 *      $RCSfile: msmpot_cubic.c,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.2 $      $Date: 2009/04/29 15:49:47 $
 *
 ***************************************************************************/
/*
 * cubic.c - smooth cubic "numerical Hermite" interpolation
 */

#include "msmpot_internal.h"

static int anterpolation(Msmpot *msm);
static int interpolation(Msmpot *msm);
static int restriction(Msmpot *msm, int level);
static int prolongation(Msmpot *msm, int level);
static int latticecutoff(Msmpot *msm, int level);


int Msmpot_compute_longrng_cubic(Msmpot *msm) {
  int err = 0;
  int level;

#ifdef MSMPOT_CUDA
  if (msm->use_cuda_latcut) {
    err = anterpolation(msm);
    if (err) return ERROR(err);

    for (level = 0;  level < msm->nlevels - 1;  level++) {
      err = restriction(msm, level);
      if (err) return ERROR(err);
    }

    if ((err = Msmpot_cuda_condense_qgrids(msm->msmcuda)) != OK ||
        (err = Msmpot_cuda_compute_latcut(msm->msmcuda)) != OK ||
        (err = Msmpot_cuda_expand_egrids(msm->msmcuda)) != OK) {
      /* fall back on CPU lattice cutoff */
      for (level = 0;  level < msm->nlevels - 1;  level++) {
        err = latticecutoff(msm, level);
        if (err) return ERROR(err);
      }
    }

    err = latticecutoff(msm, level);  /* top level */
    if (err) return ERROR(err);

    for (level--;  level >= 0;  level--) {
      err = prolongation(msm, level);
      if (err) return ERROR(err);
    }

    err = interpolation(msm);
    if (err) return ERROR(err);
  }
  else
#endif
  {
    err = anterpolation(msm);
    if (err) return ERROR(err);

    for (level = 0;  level < msm->nlevels - 1;  level++) {
      err = latticecutoff(msm, level);
      if (err) return ERROR(err);
      err = restriction(msm, level);
      if (err) return ERROR(err);
    }

    err = latticecutoff(msm, level);  /* top level */
    if (err) return ERROR(err);

    for (level--;  level >= 0;  level--) {
      err = prolongation(msm, level);
      if (err) return ERROR(err);
    }

    err = interpolation(msm);
    if (err) return ERROR(err);
  }

  return OK;
}


int anterpolation(Msmpot *msm)
{
  const float *atom = msm->atom;
  const long natoms = msm->natoms;

  float xphi[4], yphi[4], zphi[4];  /* phi grid func along x, y, z */
  float rx_hx, ry_hy, rz_hz;        /* distance from origin */
  float t;                          /* normalized distance for phi */
  float ck, cjk;
  const float hx_1 = 1/msm->hx;
  const float hy_1 = 1/msm->hy;
  const float hz_1 = 1/msm->hz;
  const float xm0 = msm->xm0;
  const float ym0 = msm->ym0;
  const float zm0 = msm->zm0;
  float q;

  MsmpotLattice *qhlat = msm->qh[0];
  float *qh = qhlat->data;
  const long ni = qhlat->ni;
  const long nj = qhlat->nj;

  long n, i, j, k, ilo, jlo, klo, index;
  long koff, jkoff;

  Msmpot_lattice_zero(qhlat);

  for (n = 0;  n < natoms;  n++) {

    /* atomic charge */
    q = atom[4*n + 3];
    if (0==q) continue;

    /* distance between atom and origin measured in grid points */
    rx_hx = (atom[4*n    ] - xm0) * hx_1;
    ry_hy = (atom[4*n + 1] - ym0) * hy_1;
    rz_hz = (atom[4*n + 2] - zm0) * hz_1;

    /* find smallest numbered grid point in stencil */
    ilo = (long) floorf(rx_hx) - 1;
    jlo = (long) floorf(ry_hy) - 1;
    klo = (long) floorf(rz_hz) - 1;

    ASSERT(qhlat->ia <= ilo && ilo + 3 <= qhlat->ib);
    ASSERT(qhlat->ja <= jlo && jlo + 3 <= qhlat->jb);
    ASSERT(qhlat->ka <= klo && klo + 3 <= qhlat->kb);

    /* find t for x dimension and compute xphi */
    t = rx_hx - (float) ilo;
    xphi[0] = 0.5f * (1 - t) * (2 - t) * (2 - t);
    t--;
    xphi[1] = (1 - t) * (1 + t - 1.5f * t * t);
    t--;
    xphi[2] = (1 + t) * (1 - t - 1.5f * t * t);
    t--;
    xphi[3] = 0.5f * (1 + t) * (2 + t) * (2 + t);

    /* find t for y dimension and compute yphi */
    t = ry_hy - (float) jlo;
    yphi[0] = 0.5f * (1 - t) * (2 - t) * (2 - t);
    t--;
    yphi[1] = (1 - t) * (1 + t - 1.5f * t * t);
    t--;
    yphi[2] = (1 + t) * (1 - t - 1.5f * t * t);
    t--;
    yphi[3] = 0.5f * (1 + t) * (2 + t) * (2 + t);

    /* find t for z dimension and compute zphi */
    t = rz_hz - (float) klo;
    zphi[0] = 0.5f * (1 - t) * (2 - t) * (2 - t);
    t--;
    zphi[1] = (1 - t) * (1 + t - 1.5f * t * t);
    t--;
    zphi[2] = (1 + t) * (1 - t - 1.5f * t * t);
    t--;
    zphi[3] = 0.5f * (1 + t) * (2 + t) * (2 + t);

    /* determine charge on 64=4*4*4 grid point stencil of qh */
    for (k = 0;  k < 4;  k++) {
      koff = (k + klo) * nj;
      ck = zphi[k] * q;
      for (j = 0;  j < 4;  j++) {
        jkoff = (koff + (j + jlo)) * ni;
        cjk = yphi[j] * ck;
        for (i = 0;  i < 4;  i++) {
          index = jkoff + (i + ilo);
          RANGE_CHECK(qhlat, i+ilo, j+jlo, k+klo);
          ASSERT(INDEX(qhlat, i+ilo, j+jlo, k+klo) == index);
          qh[index] += xphi[i] * cjk;
        }
      }
    }

  } /* end loop over atoms */
  return OK;
} /* anterpolation */


int interpolation(Msmpot *msm) {
  float *epotmap = msm->epotmap;

  float *ezd = msm->ezd;
  float *eyzd = msm->eyzd;

  const MsmpotLattice *ehlat = msm->eh[0];
  const float *eh = ehlat->data;
  const long ia = ehlat->ia;
  const long ib = ehlat->ib;
  const long ja = ehlat->ja;
  const long jb = ehlat->jb;
  const long ka = ehlat->ka;
  const long kb = ehlat->kb;
  const long nrow_eh = ehlat->ni;
  const long nstride_eh = nrow_eh * ehlat->nj;

  const long mx = msm->mx;
  const long my = msm->my;
  const long mz = msm->mz;

  const long size_ezd = mz * sizeof(float);
  const long size_eyzd = my * mz * sizeof(float);

  const long imask = msm->cycle_x - 1;
  const long jmask = msm->cycle_y - 1;
  const long kmask = msm->cycle_z - 1;

  const long rmap_x = msm->rmap_x;
  const long rmap_y = msm->rmap_y;
  const long rmap_z = msm->rmap_z;

  const long diam_x = 2*rmap_x + 1;
  const long diam_y = 2*rmap_y + 1;
  const long diam_z = 2*rmap_z + 1;

  const float *base_phi_x = msm->phi_x;
  const float *base_phi_y = msm->phi_y;
  const float *base_phi_z = msm->phi_z;
  const float *phi = NULL;

  const float hx_dx = msm->hx_dx;
  const float hy_dy = msm->hy_dy;
  const float hz_dz = msm->hz_dz;

  long ih, jh, kh;
  long im, jm, km;
  long i, j, k;
  long index_plane_eh, index_eh;
  long index_jk, offset_k, offset;
  long lower, upper;


  for (ih = ia;  ih <= ib;  ih++) {
    memset(eyzd, 0, size_eyzd);

    for (jh = ja;  jh <= jb;  jh++) {
      memset(ezd, 0, size_ezd);
      index_plane_eh = jh * nrow_eh + ih;

      for (kh = ka;  kh <= kb;  kh++) {
        index_eh = kh * nstride_eh + index_plane_eh;
        km = (long) floorf(kh * hz_dz);
        lower = km - rmap_z;
        if (lower < 0) lower = 0;
        upper = km + rmap_z;
        if (upper >= mz) upper = mz-1;
        phi = base_phi_z + diam_z * (kh & kmask) + rmap_z;
        for (k = lower;  k <= upper;  k++) {
          ezd[k] += phi[k-km] * eh[index_eh];
        }
      }

      for (k = 0;  k < mz;  k++) {
        offset = k * my;
        jm = (long) floorf(jh * hy_dy);
        lower = jm - rmap_y;
        if (lower < 0) lower = 0;
        upper = jm + rmap_y;
        if (upper >= my) upper = my-1;
        phi = base_phi_y + diam_y * (jh & jmask) + rmap_y;
        for (j = lower;  j <= upper;  j++) {
          eyzd[offset + j] += phi[j-jm] * ezd[k];
        }
      }
    }

    for (k = 0;  k < mz;  k++) {
      offset_k = k * my;

      for (j = 0;  j < my;  j++) {
        index_jk = offset_k + j;
        offset = index_jk * mx;

        im = (long) floorf(ih * hx_dx);
        lower = im - rmap_x;
        if (lower < 0) lower = 0;
        upper = im + rmap_x;
        if (upper >= mx) upper = mx-1;
        phi = base_phi_x + diam_x * (ih & imask) + rmap_x;
        for (i = lower;  i <= upper;  i++) {
          epotmap[offset + i] += phi[i-im] * eyzd[index_jk];
        }
      }
    }

  }
  return OK;
} /* interpolation */


#if 0
  const int scalexp = mg->scalexp;
  const int sdelta = mg->sdelta;

  const float *phi = mg->phi;
  float *ezd = mg->ezd;
  float *eyzd = mg->eyzd;

  floatLattice *egrid = mg->egrid[0];
  const float *eh = egrid->data(egrid);
  const long int ia = egrid->ia;
  const long int ib = egrid->ib;
  const long int ja = egrid->ja;
  const long int jb = egrid->jb;
  const long int ka = egrid->ka;
  const long int kb = egrid->kb;
  const long int nrow_eh = egrid->ni;
  const long int nstride_eh = nrow_eh * egrid->nj;

  const long int size_eyzd = numplane * numcol * sizeof(float);
  const long int size_ezd = numplane * sizeof(float);

  long int i, j, k;
  long int ih, jh, kh;
  long int im, jm, km;
  long int lower, upper;
  long int index_eh, index_plane_eh;
  long int offset, offset_k, index_jk, index;

  printf("doing factored interpolation\n");
  for (ih = ia;  ih <= ib;  ih++) {
    memset(eyzd, 0, size_eyzd);

    for (jh = ja;  jh <= jb;  jh++) {
      memset(ezd, 0, size_ezd);
      index_plane_eh = jh * nrow_eh + ih;

      for (kh = ka;  kh <= kb;  kh++) {
        index_eh = kh * nstride_eh + index_plane_eh;
        km = (kh << scalexp);
        lower = km - sdelta;
        if (lower < 0) lower = 0;
        upper = km + sdelta;
        if (upper >= numplane) upper = numplane-1;
        for (k = lower;  k <= upper;  k++) {
          ezd[k] += phi[k-km] * eh[index_eh];
        }
      }

      for (k = 0;  k < numplane;  k++) {
        offset = k * numcol;
        jm = (jh << scalexp);
        lower = jm - sdelta;
        if (lower < 0) lower = 0;
        upper = jm + sdelta;
        if (upper >= numcol) upper = numcol-1;
        for (j = lower;  j <= upper;  j++) {
          eyzd[offset + j] += phi[j-jm] * ezd[k];
        }
      }
    }

    for (k = 0;  k < numplane;  k++) {
      offset_k = k * numcol;

      for (j = 0;  j < numcol;  j++) {
        index_jk = offset_k + j;
        offset = index_jk * numpt;

        im = (ih << scalexp);
        lower = im - sdelta;
        if (lower < 0) lower = 0;
        upper = im + sdelta;
        if (upper >= numpt) upper = numpt-1;
        for (i = lower;  i <= upper;  i++) {
          grideners[offset + i] += phi[i-im] * eyzd[index_jk];
        }
      }
    }

  }

  for (k = 0;  k < numplane;  k++) {
    offset_k = k * numcol;
    for (j = 0;  j < numcol;  j++) {
      offset = (offset_k + j) * numpt;
      for (i = 0;  i < numpt;  i++) {
        index = offset + i;
        grideners[index] = (excludepos[index] ? 0 : grideners[index]);
      }
    }
  }


#if 0

  float e;

  const int scalexp = mg->scalexp;
  const int dim = (1 << scalexp);
  const long int mask = (long int)(dim-1);

  floatLattice *egrid = mg->egrid[0];
  const float *eh = egrid->data(egrid);
  const long int eni = egrid->ni;
  const long int enj = egrid->nj;

  long int index;
  long int i, j, k;
  long int im, jm, km;
  long int ip, jp, kp;

  floatLattice *w;
  const float *wt;
  long int wni, wnj;
  long int ia, ib, ja, jb, ka, kb;

  long int ii, jj, kk;
  long int woff_k, woff_jk, eoff_k, eoff_jk;

  /* loop over grideners */
  for (k = 0;  k < numplane;  k++) {
    for (j = 0;  j < numcol;  j++) {
      for (i = 0;  i < numpt;  i++) {

        /* index into grideners and excludepos arrays */
        index = (k*numcol + j)*numpt + i;
        if (excludepos[index]) continue;

	/* find closest mgpot point less than or equal to */
	im = (i >> scalexp);
	jm = (j >> scalexp);
	km = (k >> scalexp);

	/* find corresponding potinterp lattice */
	ip = (int)(i & mask);
	jp = (int)(j & mask);
	kp = (int)(k & mask);

	w = mg->potinterp[(kp*dim + jp)*dim + ip];
	wt = w->data(w);
	wni = w->ni;
	wnj = w->nj;
	ia = w->ia;
	ib = w->ib;
	ja = w->ja;
	jb = w->jb;
	ka = w->ka;
	kb = w->kb;

	/* loop over wt, summing weighted eh contributions to e */
	e = 0;
	for (kk = ka;  kk <= kb;  kk++) {
	  woff_k = kk*wnj;
	  eoff_k = (km + kk)*enj;
	  for (jj = ja;  jj <= jb;  jj++) {
	    woff_jk = (woff_k + jj)*wni;
	    eoff_jk = (eoff_k + (jm+jj))*eni;
	    for (ii = ia;  ii <= ib;  ii++) {
	      ASSERT(w->index(w, ii, jj, kk) == woff_jk + ii);
	      ASSERT(egrid->index(egrid, im+ii, jm+jj, km+kk)
		  == eoff_jk + (im+ii));
	      e += wt[woff_jk + ii] * eh[eoff_jk + (im+ii)];
	    }
	  }
	}
	grideners[index] += e;
	/* end loop over wt */

      }
    }
  } /* end loop over grideners */

#endif

#endif



/* constants for grid transfer operations
 * cubic "numerical Hermite" interpolation */

/* length of stencil */
enum { NSTENCIL = 5 };

/* phi interpolating function along one dimension of grid stencil */
static const float Phi[NSTENCIL] = { -0.0625f, 0.5625f, 1, 0.5625f, -0.0625f };

/* stencil offsets from a central grid point on a finer grid level */
/* (these offsets are where phi weights above have been evaluated) */
static const int Offset[NSTENCIL] = { -3, -1, 0, 1, 3 };


int restriction(Msmpot *msm, int level)
{
  float cjk, q2h_sum;

  /* lattices of charge, finer grid and coarser grid */
  const MsmpotLattice *qhlat = msm->qh[level];
  const float *qh = qhlat->data;
  MsmpotLattice *q2hlat = msm->qh[level+1];
  float *q2h = q2hlat->data;

  /* finer grid index ranges and dimensions */
  const long ia1 = qhlat->ia;   /* lowest x-index */
  const long ib1 = qhlat->ib;   /* highest x-index */
  const long ja1 = qhlat->ja;   /* lowest y-index */
  const long jb1 = qhlat->jb;   /* highest y-index */
  const long ka1 = qhlat->ka;   /* lowest z-index */
  const long kb1 = qhlat->kb;   /* highest z-index */
  const long ni1 = qhlat->ni;   /* length along x-dim */
  const long nj1 = qhlat->nj;   /* length along y-dim */

  /* coarser grid index ranges and dimensions */
  const long ia2 = q2hlat->ia;  /* lowest x-index */
  const long ib2 = q2hlat->ib;  /* highest x-index */
  const long ja2 = q2hlat->ja;  /* lowest y-index */
  const long jb2 = q2hlat->jb;  /* highest y-index */
  const long ka2 = q2hlat->ka;  /* lowest z-index */
  const long kb2 = q2hlat->kb;  /* highest z-index */
  const long ni2 = q2hlat->ni;  /* length along x-dim */
  const long nj2 = q2hlat->nj;  /* length along y-dim */

  /* other variables */
  long i1, j1, k1, index1, jk1off, k1off;
  long i2, j2, k2, index2, jk2off, k2off;
  long i, j, k;

  /* loop over coarser grid points */
  for (k2 = ka2;  k2 <= kb2;  k2++) {
    k2off = k2 * nj2;    /* coarser grid index offset for k-coord */
    k1 = k2 * 2;         /* k-coord of same-space point on finer grid */
    for (j2 = ja2;  j2 <= jb2;  j2++) {
      jk2off = (k2off + j2) * ni2;    /* add offset for j-coord coarser */
      j1 = j2 * 2;                    /* j-coord same-space finer grid */
      for (i2 = ia2;  i2 <= ib2;  i2++) {
        index2 = jk2off + i2;         /* index in coarser grid */
        i1 = i2 * 2;                  /* i-coord same-space finer grid */

        /* sum weighted charge contribution from finer grid stencil */
        q2h_sum = 0;
        for (k = 0;  k < NSTENCIL;  k++) {
          /* early loop termination if outside lattice */
          if (k1 + Offset[k] < ka1) continue;
          else if (k1 + Offset[k] > kb1) break;
          k1off = (k1 + Offset[k]) * nj1;  /* offset k-coord finer grid */
          for (j = 0;  j < NSTENCIL;  j++) {
            /* early loop termination if outside lattice */
            if (j1 + Offset[j] < ja1) continue;
            else if (j1 + Offset[j] > jb1) break;
            jk1off = (k1off + (j1 + Offset[j])) * ni1;  /* add offset j */
            cjk = Phi[j] * Phi[k];              /* mult weights in each dim */
            for (i = 0;  i < NSTENCIL;  i++) {
              /* early loop termination if outside lattice */
              if (i1 + Offset[i] < ia1) continue;
              else if (i1 + Offset[i] > ib1) break;
              index1 = jk1off + (i1 + Offset[i]);    /* index in finer grid */
              RANGE_CHECK(qhlat, i1+Offset[i], j1+Offset[j], k1+Offset[k]);
              ASSERT(INDEX(qhlat, i1+Offset[i], j1+Offset[j], k1+Offset[k])
                  == index1);
              q2h_sum += Phi[i] * cjk * qh[index1];  /* sum weighted charge */
            }
          }
        } /* end loop over finer grid stencil */

        RANGE_CHECK(q2hlat, i2, j2, k2);
	ASSERT(INDEX(q2hlat, i2, j2, k2) == index2);
        q2h[index2] = q2h_sum;  /* store charge to coarser grid */

      }
    }
  } /* end loop over each coarser grid points */
  return OK;
}


int prolongation(Msmpot *msm, int level)
{
  float ck, cjk;

  /* lattices of charge, finer grid and coarser grid */
  MsmpotLattice *ehlat = msm->eh[level];
  float *eh = ehlat->data;
  const MsmpotLattice *e2hlat = msm->eh[level+1];
  const float *e2h = e2hlat->data;

  /* finer grid index ranges and dimensions */
  const long ia1 = ehlat->ia;   /* lowest x-index */
  const long ib1 = ehlat->ib;   /* highest x-index */
  const long ja1 = ehlat->ja;   /* lowest y-index */
  const long jb1 = ehlat->jb;   /* highest y-index */
  const long ka1 = ehlat->ka;   /* lowest z-index */
  const long kb1 = ehlat->kb;   /* highest z-index */
  const long ni1 = ehlat->ni;   /* length along x-dim */
  const long nj1 = ehlat->nj;   /* length along y-dim */

  /* coarser grid index ranges and dimensions */
  const long ia2 = e2hlat->ia;  /* lowest x-index */
  const long ib2 = e2hlat->ib;  /* highest x-index */
  const long ja2 = e2hlat->ja;  /* lowest y-index */
  const long jb2 = e2hlat->jb;  /* highest y-index */
  const long ka2 = e2hlat->ka;  /* lowest z-index */
  const long kb2 = e2hlat->kb;  /* highest z-index */
  const long ni2 = e2hlat->ni;  /* length along x-dim */
  const long nj2 = e2hlat->nj;  /* length along y-dim */

  /* other variables */
  long i1, j1, k1, index1, jk1off, k1off;
  long i2, j2, k2, index2, jk2off, k2off;
  long i, j, k;

  /* loop over coarser grid points */
  for (k2 = ka2;  k2 <= kb2;  k2++) {
    k2off = k2 * nj2;    /* coarser grid index offset for k-coord */
    k1 = k2 * 2;         /* k-coord of same-space point on finer grid */
    for (j2 = ja2;  j2 <= jb2;  j2++) {
      jk2off = (k2off + j2) * ni2;    /* add offset for j-coord coarser */
      j1 = j2 * 2;                    /* j-coord same-space finer grid */
      for (i2 = ia2;  i2 <= ib2;  i2++) {
        index2 = jk2off + i2;         /* index in coarser grid */
        i1 = i2 * 2;                  /* i-coord same-space finer grid */

        /* sum weighted charge contribution from finer grid stencil */
        RANGE_CHECK(e2hlat, i2, j2, k2);
        ASSERT(INDEX(e2hlat, i2, j2, k2) == index2);
        for (k = 0;  k < NSTENCIL;  k++) {
          /* early loop termination if outside lattice */
          if (k1 + Offset[k] < ka1) continue;
          else if (k1 + Offset[k] > kb1) break;
          k1off = (k1 + Offset[k]) * nj1;  /* offset k-coord finer grid */
	  ck = Phi[k] * e2h[index2];
          for (j = 0;  j < NSTENCIL;  j++) {
            /* early loop termination if outside lattice */
            if (j1 + Offset[j] < ja1) continue;
            else if (j1 + Offset[j] > jb1) break;
            jk1off = (k1off + (j1 + Offset[j])) * ni1;  /* add offset j */
            cjk = Phi[j] * ck;              /* mult weights in each dim */
            for (i = 0;  i < NSTENCIL;  i++) {
              /* early loop termination if outside lattice */
              if (i1 + Offset[i] < ia1) continue;
              else if (i1 + Offset[i] > ib1) break;
              index1 = jk1off + (i1 + Offset[i]);    /* index in finer grid */
              RANGE_CHECK(ehlat, i1+Offset[i], j1+Offset[j], k1+Offset[k]);
              ASSERT(INDEX(ehlat, i1+Offset[i], j1+Offset[j], k1+Offset[k])
                  == index1);
	      eh[index1] += Phi[i] * cjk;            /* sum weighted charge */
            }
          }
        } /* end loop over finer grid stencil */

      }
    }
  } /* end loop over each coarser grid points */
  return OK;
}


int latticecutoff(Msmpot *msm, int level)
{
  float eh_sum;

  /* lattices of charge and potential */
  const MsmpotLattice *qhlat = msm->qh[level];
  const float *qh = qhlat->data;
  MsmpotLattice *ehlat = msm->eh[level];
  float *eh = ehlat->data;
  const long ia = qhlat->ia;   /* lowest x-index */
  const long ib = qhlat->ib;   /* highest x-index */
  const long ja = qhlat->ja;   /* lowest y-index */
  const long jb = qhlat->jb;   /* highest y-index */
  const long ka = qhlat->ka;   /* lowest z-index */
  const long kb = qhlat->kb;   /* highest z-index */
  const long ni = qhlat->ni;   /* length along x-dim */
  const long nj = qhlat->nj;   /* length along y-dim */

  /* lattice of weights for pairwise grid point interactions within cutoff */
  const MsmpotLattice *gclat = msm->gc[level];
  const float *gc = gclat->data;
  const long gia = gclat->ia;  /* lowest x-index */
  const long gib = gclat->ib;  /* highest x-index */
  const long gja = gclat->ja;  /* lowest y-index */
  const long gjb = gclat->jb;  /* highest y-index */
  const long gka = gclat->ka;  /* lowest z-index */
  const long gkb = gclat->kb;  /* highest z-index */
  const long gni = gclat->ni;  /* length along x-dim */
  const long gnj = gclat->nj;  /* length along y-dim */

  long i, j, k;
  long gia_clip, gib_clip;
  long gja_clip, gjb_clip;
  long gka_clip, gkb_clip;
  long koff, jkoff, index;
  long id, jd, kd;
  long knoff, jknoff, nindex;
  long kgoff, jkgoff, ngindex;

  /* loop over all grid points */
  for (k = ka;  k <= kb;  k++) {

    /* clip gc ranges to keep offset for k index within grid */
    gka_clip = (k + gka < ka ? ka - k : gka);
    gkb_clip = (k + gkb > kb ? kb - k : gkb);

    koff = k * nj;  /* find eh flat index */

    for (j = ja;  j <= jb;  j++) {

      /* clip gc ranges to keep offset for j index within grid */
      gja_clip = (j + gja < ja ? ja - j : gja);
      gjb_clip = (j + gjb > jb ? jb - j : gjb);

      jkoff = (koff + j) * ni;  /* find eh flat index */

      for (i = ia;  i <= ib;  i++) {

        /* clip gc ranges to keep offset for i index within grid */
        gia_clip = (i + gia < ia ? ia - i : gia);
        gib_clip = (i + gib > ib ? ib - i : gib);

        index = jkoff + i;  /* eh flat index */

        /* sum over "sphere" of weighted charge */
        eh_sum = 0;
        for (kd = gka_clip;  kd <= gkb_clip;  kd++) {
          knoff = (k + kd) * nj;  /* find qh flat index */
          kgoff = kd * gnj;       /* find gc flat index */

          for (jd = gja_clip;  jd <= gjb_clip;  jd++) {
            jknoff = (knoff + (j + jd)) * ni;  /* find qh flat index */
            jkgoff = (kgoff + jd) * gni;       /* find gc flat index */

            for (id = gia_clip;  id <= gib_clip;  id++) {
              nindex = jknoff + (i + id);  /* qh flat index */
              ngindex = jkgoff + id;       /* gc flat index */

              RANGE_CHECK(qhlat, i+id, j+jd, k+kd);
              ASSERT(INDEX(qhlat, i+id, j+jd, k+kd) == nindex);

              RANGE_CHECK(gclat, id, jd, kd);
	      ASSERT(INDEX(gclat, id, jd, kd) == ngindex);

              eh_sum += qh[nindex] * gc[ngindex];  /* sum weighted charge */
            }
          }
        } /* end loop over "sphere" of charge */

        RANGE_CHECK(ehlat, i, j, k);
	ASSERT(INDEX(ehlat, i, j, k) == index);
        eh[index] = eh_sum;  /* store potential */
      }
    }
  } /* end loop over all grid points */
  return OK;
}
