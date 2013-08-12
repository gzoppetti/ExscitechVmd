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
 *      $RCSfile: msmpot_cuda_shortrng.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.2 $      $Date: 2009/04/29 15:49:47 $
 *
 ***************************************************************************/
/*
 * msmcuda_shortrng.cu
 */

#include "msmpot_internal.h"


/*
 * neighbor list storage uses 64000 bytes
 */
static __constant__ int NbrListLen;
static __constant__ int3 NbrList[NBRLIST_MAXLEN];


#define BIN_LENGTH      4.f  /* spatial length in Angstroms */
#define BIN_INVLEN  (1.f / BIN_LENGTH)
/* assuming density of 1 atom / 10 A^3, expectation is 6.4 atoms per bin
 * so that bin fill should be about 80% (for non-empty regions of space) */

#define BIN_SHIFT         5  /* # of bits to shift for mul/div by BIN_SIZE */
#define BIN_CACHE_MAXLEN  1  /* max number of atom bins to cache */
#define REGION_SIZE     512  /* number of floats in lattice region */


/*
 * The following code is adapted from kernel
 *   cuda_cutoff_potential_lattice10overlap()
 * from source file mgpot_cuda_binsmall.cu from VMD cionize plugin.
 *
 * potential lattice is decomposed into size 8^3 lattice point "regions"
 *
 * THIS IMPLEMENTATION:  one thread per lattice point
 * thread block size 128 gives 4 thread blocks per region
 * kernel is invoked for each x-y plane of regions,
 * where gridDim.x is 4*(x region dimension) so that blockIdx.x 
 * can absorb the z sub-region index in its 2 lowest order bits
 *
 * Regions are stored contiguously in memory in row-major order
 *
 * The bins have to not only cover the region, but they need to surround
 * the outer edges so that region sides and corners can still use
 * neighbor list stencil.  The binZeroAddr is actually a shifted pointer into
 * the bin array (binZeroAddr = binBaseAddr + (c*binDim_y + c)*binDim_x + c)
 * where c = ceil(cutoff / binsize).  This allows for negative offsets to
 * be added to myBinIndex.
 *
 * The (0,0,0) spatial origin corresponds to lower left corner of both
 * regionZeroAddr and binZeroAddr.  The atom coordinates are translated
 * during binning to enforce this assumption.
 */
__global__ static void cuda_shortrange(
    int binDim_x,
    int binDim_y,
    float *binZeroAddr,     /* address of atom bins starting at origin */
    float h,                /* lattice spacing */
    float cutoff2,          /* square of cutoff distance */
    float invcut,           /* 1/cutoff */
    float *regionZeroAddr,  /* address of lattice regions starting at origin */
    int zRegionDim
    )
{
  __shared__ float AtomBinCache[BIN_CACHE_MAXLEN * BIN_DEPTH * 4];
  __shared__ float *myRegionAddr;
  __shared__ int3 myBinIndex;

  const int xRegionIndex = blockIdx.x;
  const int yRegionIndex = blockIdx.y;

  /* thread id */
  const int tid = (threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x
    + threadIdx.x;
  /* blockDim.x == 8, blockDim.y == 2, blockDim.z == 8 */

  /* neighbor index */
  int nbrid;

  /* constants for TAYLOR2 softening */
  /* XXX is it more efficient to read these values from const memory? */
  float gc0, gc1, gc2;
  gc1 = invcut * invcut;
  gc2 = gc1 * gc1;
  gc0 = 1.875f * invcut;
  gc1 *= -1.25f * invcut;
  gc2 *= 0.375f * invcut;

  int zRegionIndex;
  for (zRegionIndex=0; zRegionIndex < zRegionDim; zRegionIndex++) {

    /* this is the start of the sub-region indexed by tid */
    myRegionAddr = regionZeroAddr + ((zRegionIndex*gridDim.y
          + yRegionIndex)*gridDim.x + xRegionIndex)*REGION_SIZE;
      
    /* spatial coordinate of this lattice point */
    float x = (8 * xRegionIndex + threadIdx.x) * h;
    float y = (8 * yRegionIndex + threadIdx.y) * h;
    float z = (8 * zRegionIndex + threadIdx.z) * h;

    /* bin number determined by center of region */
    myBinIndex.x = (int) floorf((8 * xRegionIndex + 4) * h * BIN_INVLEN);
    myBinIndex.y = (int) floorf((8 * yRegionIndex + 4) * h * BIN_INVLEN);
    myBinIndex.z = (int) floorf((8 * zRegionIndex + 4) * h * BIN_INVLEN);

    float energy0 = 0.f;
    float energy1 = 0.f;
    float energy2 = 0.f;
    float energy3 = 0.f;

    for (nbrid = 0;  nbrid < NbrListLen;  nbrid++) {

      /* thread block caches one bin */
      if (tid < 32) {
        int i = myBinIndex.x + NbrList[nbrid].x;
        int j = myBinIndex.y + NbrList[nbrid].y;
        int k = myBinIndex.z + NbrList[nbrid].z;

        /* determine global memory location of atom bin */
        float *p_global = ((float *) binZeroAddr)
          + (((__mul24(k, binDim_y) + j)*binDim_x + i) << BIN_SHIFT);

        AtomBinCache[tid] = p_global[tid];
      }
      __syncthreads();

      {
        int i;

        for (i = 0;  i < BIN_DEPTH;  i++) {
          int off = (i << 2);

          float aq = AtomBinCache[off + 3];
          if (0.f == aq) break;  /* no more atoms in bin */

          float dx = AtomBinCache[off    ] - x;
          float dz = AtomBinCache[off + 2] - z;
          float dxdz2 = dx*dx + dz*dz;
#ifdef CHECK_CYLINDER
          if (dxdz2 >= cutoff2) continue;
#endif
          float dy = AtomBinCache[off + 1] - y;
          float r2 = dy*dy + dxdz2;

          if (r2 < cutoff2) {
            float gr2 = gc0 + r2*(gc1 + r2*gc2);
            energy0 += aq * (rsqrtf(r2) - gr2);
          }
          dy -= 2.0f*h;
          r2 = dy*dy + dxdz2;
          if (r2 < cutoff2) {
            float gr2 = gc0 + r2*(gc1 + r2*gc2);
            energy1 += aq * (rsqrtf(r2) - gr2);
          }
          dy -= 2.0f*h;
          r2 = dy*dy + dxdz2;
          if (r2 < cutoff2) {
            float gr2 = gc0 + r2*(gc1 + r2*gc2);
            energy2 += aq * (rsqrtf(r2) - gr2);
          }
          dy -= 2.0f*h;
          r2 = dy*dy + dxdz2;
          if (r2 < cutoff2) {
            float gr2 = gc0 + r2*(gc1 + r2*gc2);
            energy3 += aq * (rsqrtf(r2) - gr2);
          }
        } /* end loop over atoms in bin */
      } /* end loop over cached atom bins */
      __syncthreads();

    } /* end loop over neighbor list */

    /* store into global memory */
    myRegionAddr[(tid>>4)*64 + (tid&15)     ] = energy0;
    myRegionAddr[(tid>>4)*64 + (tid&15) + 16] = energy1;
    myRegionAddr[(tid>>4)*64 + (tid&15) + 32] = energy2;
    myRegionAddr[(tid>>4)*64 + (tid&15) + 48] = energy3;

  } /* end loop over zRegionIndex */

}


/*
 * call when finished
 */
void Msmpot_cuda_cleanup_shortrng(MsmpotCuda *mc) {
  cudaFree(mc->dev_binBase);
  cudaFree(mc->dev_padEpotmap);
  free(mc->nbrlist);
  free(mc->extra_atom);
  free(mc->bincntBase);
  free(mc->binBase);
  free(mc->padEpotmap);
}


/*
 * call once or whenever parameters are changed
 */
int Msmpot_cuda_setup_shortrng(MsmpotCuda *mc) {
  const Msmpot *msm = mc->msmpot;
  const long mx = msm->mx;
  const long my = msm->my;
  const long mz = msm->mz;
  long rmx, rmy, rmz;
  long pmx, pmy, pmz, pmall;
  long nxbins, nybins, nzbins, nbins, nbpad;
  const float binlen = BIN_LENGTH;      /* XXX won't work for PBC */
  const float invbinlen = BIN_INVLEN;   /* XXX won't work for PBC */
  const float cutoff = msm->a;
  const float dx = msm->dx;
  const float dy = msm->dy;
  const float dz = msm->dz;
  float sqbindiag, r, r2;
  int c, bpr0, bpr1;
  int cmin, cmax;
  int nbrx, nbry, nbrz;
  int cnt, i, j, k;
  int *nbrlist = NULL;

  /* XXX kernel must have same lattice spacing in each dimension */
  if (dx != dy || dx != dz) return ERROR(MSMPOT_ERROR_AVAIL);

  /* set length of atom bins */
  mc->binlen = binlen;
  mc->invbinlen = invbinlen;

  /* count "regions" of 8^3 map points */
  rmx = (mx >> 3) + ((mx & 7) ? 1 : 0);  /* = (long) ceilf(mx/8.f) */
  rmy = (my >> 3) + ((my & 7) ? 1 : 0);  /* = (long) ceilf(my/8.f) */
  rmz = (mz >> 3) + ((mz & 7) ? 1 : 0);  /* = (long) ceilf(mz/8.f) */

  /* padded epotmap dimensions */
  pmx = (rmx << 3);                      /* = 8 * rmx */
  pmy = (rmy << 3);                      /* = 8 * rmy */
  pmz = (rmz << 3);                      /* = 8 * rmz */

  /* allocate space for padded epotmap */
  pmall = pmx * pmy * pmz;
  if (mc->maxpm < pmall) {
    void *v = realloc(mc->padEpotmap, pmall * sizeof(float));
    if (NULL == v) return ERROR(MSMPOT_ERROR_ALLOC);
    mc->padEpotmap = (float *) v;
    mc->maxpm = pmall;
  }
  mc->pmx = pmx;
  mc->pmy = pmy;
  mc->pmz = pmz;

  /*
   * "improved" generic neighborlist creation
   * adapted from mgpot_cuda_binsmall.cu
   */
  c = (int) ceilf(cutoff * invbinlen);  /* number of bins covering cutoff */
  sqbindiag = 0.f;

  bpr0 = (int) floorf(8.f*dx*invbinlen);
  bpr1 = (int) ceilf(8.f*dx*invbinlen);
  if (bpr0 == bpr1) {  /* atom bins exactly cover region in x-direction */
    nbrx = c + (bpr0 >> 1);  /* bpr0 / 2 */
    /* if atom bin cover is odd, use square of half-bin-length */
    sqbindiag += ((bpr0 & 1) ? 0.25f : 1.f)*binlen*binlen;
  }
  else {
    nbrx = (int) ceilf((cutoff + 4.f*dx + binlen)*invbinlen);
    sqbindiag += binlen*binlen;
  }

  bpr0 = (int) floorf(8*dy*invbinlen);
  bpr1 = (int) ceilf(8*dy*invbinlen);
  if (bpr0 == bpr1) {  /* atom bins exactly cover region in y-direction */
    nbry = c + (bpr0 >> 1);  /* bpr0 / 2 */
    /* if atom bin cover is odd, use square of half-bin-length */
    sqbindiag += ((bpr0 & 1) ? 0.25f : 1.f)*binlen*binlen;
  }
  else {
    nbry = (int) ceilf((cutoff + 4.f*dy + binlen)*invbinlen);
    sqbindiag += binlen*binlen;
  }

  bpr0 = (int) floorf(8*dz*invbinlen);
  bpr1 = (int) ceilf(8*dz*invbinlen);
  if (bpr0 == bpr1) {  /* atom bins exactly cover region in z-direction */
    nbrz = c + (bpr0 >> 1);  /* bpr0 / 2 */
    /* if atom bin cover is odd, use square of half-bin-length */
    sqbindiag += ((bpr0 & 1) ? 0.25f : 1.f)*binlen*binlen;
  }
  else {
    nbrz = (int) ceilf((cutoff + 4.f*dz + binlen)*invbinlen);
    sqbindiag += binlen*binlen;
  }

  r = cutoff + 4.f*sqrtf(dx*dx + dy*dy + dz*dz) + sqrtf(sqbindiag);
  r2 = r*r;

  if (mc->nbrlistmax < 3*NBRLIST_MAXLEN) {
    void *v = realloc(mc->nbrlist, 3*NBRLIST_MAXLEN*sizeof(int));
    if (NULL == v) return ERROR(MSMPOT_ERROR_ALLOC);
    mc->nbrlist = (int *) v;
    mc->nbrlistmax = 3*NBRLIST_MAXLEN;
  }
  nbrlist = mc->nbrlist;

  cnt = 0;
  cmin = cmax = 0;
  for (k = -nbrz;  k <= nbrz;  k++) {
    for (j = -nbry;  j <= nbry;  j++) {
      for (i = -nbrx;  i <= nbrx;  i++) {
        if ((i*i + j*j + k*k)*binlen*binlen >= r2) continue;
        if (3*NBRLIST_MAXLEN == cnt) return ERROR(MSMPOT_ERROR_AVAIL);
        nbrlist[cnt++] = i;
        nbrlist[cnt++] = j;
        nbrlist[cnt++] = k;
        if (cmin > i)      cmin = i;
        else if (cmax < i) cmax = i;
        if (cmin > j)      cmin = j;
        else if (cmax < j) cmax = j;
        if (cmin > k)      cmin = k;
        else if (cmax < k) cmax = k;
      }
    }
  }
  mc->nbrlistlen = cnt / 3;

  nbpad = -cmin;
  if (nbpad < cmax) nbpad = cmax;
  if (nbpad < c) nbpad = c;

  /* allocate space for atom bins */
  nxbins = (long) ceilf(pmx * dx * invbinlen) + 2*nbpad;
  nybins = (long) ceilf(pmy * dy * invbinlen) + 2*nbpad;
  nzbins = (long) ceilf(pmz * dz * invbinlen) + 2*nbpad;
  nbins = nxbins * nybins * nzbins;
  if (mc->maxbins < nbins) {
    void *v = realloc(mc->binBase, nbins * BIN_SIZE * sizeof(float));
    if (NULL == v) return ERROR(MSMPOT_ERROR_ALLOC);
    mc->binBase = (float *) v;
    v = realloc(mc->bincntBase, nbins * sizeof(char));
    if (NULL == v) return ERROR(MSMPOT_ERROR_ALLOC);
    mc->bincntBase = (char *) v;
    v = realloc(mc->extra_atom, nbins * 4 * sizeof(float));
    if (NULL == v) return ERROR(MSMPOT_ERROR_ALLOC);
    mc->extra_atom = (float *) v;
    mc->maxbins = nbins;
  }
  mc->binZero = mc->binBase
    + ((nbpad*nybins + nbpad)*nxbins + nbpad) * BIN_SIZE;
  mc->bincntZero = mc->bincntBase + (nbpad*nybins + nbpad)*nxbins + nbpad;
  mc->nxbins = nxbins;
  mc->nybins = nybins;
  mc->nzbins = nzbins;
  mc->nbins = nbins;
  mc->nbpad = nbpad;

  /*
   * allocate CUDA device memory
   * (for now make CUDA arrays same length as host arrays)
   */
  if (mc->dev_maxpm < mc->maxpm) {
    void *v = NULL;
    cudaFree(mc->dev_padEpotmap);
    CUERR;
    cudaMalloc(&v, mc->maxpm * sizeof(float));
    CUERR;
    mc->dev_padEpotmap = (float *) v;
    mc->dev_maxpm = mc->maxpm;
  }

  if (mc->dev_maxbins < mc->maxbins) {
    void *v = NULL;
    cudaFree(mc->dev_binBase);
    CUERR;
    cudaMalloc(&v, mc->maxbins * BIN_SIZE * sizeof(float));
    CUERR;
    mc->dev_binBase = (float *) v;
    mc->dev_maxbins = mc->maxbins;
  }
  /* add in the offset between host memory binZero and binBase */
  mc->dev_binZero = mc->dev_binBase + (mc->binZero - mc->binBase);

  /*
   * copy region neighborhood atom bin index offsets
   * to device constant memory
   */
  cudaMemcpyToSymbol(NbrListLen, &(mc->nbrlistlen), sizeof(int), 0);
  CUERR;
  cudaMemcpyToSymbol(NbrList, mc->nbrlist, mc->nbrlistlen * sizeof(int3), 0);
  CUERR;

  return OK;
}


int Msmpot_cuda_compute_shortrng(MsmpotCuda *mc) {
  Msmpot *msm = mc->msmpot;
  float *epotmap = msm->epotmap;
  float *padEpotmap = mc->padEpotmap;
  char *bincntZero = mc->bincntZero;
  float *binZero = mc->binZero;
  float *extra = mc->extra_atom;
  const float *atom = msm->atom;
  const long natoms = msm->natoms;
  const long nxbins = mc->nxbins;
  const long nybins = mc->nybins;
  const long nzbins = mc->nzbins;
  const long nbins = mc->nbins;
  const long nbpad = mc->nbpad;
  const long mx = msm->mx;
  const long my = msm->my;
  const long mz = msm->mz;
  const long mxRegions = (mc->pmx >> 3);
  const long myRegions = (mc->pmy >> 3);
  const long mzRegions = (mc->pmz >> 3);
  const long pmall = mc->pmx * mc->pmy * mc->pmz;
  const float cutoff = msm->a;
  const float delta = msm->dx;  /* epotmap spacing must be same in all dim */
  const float xm0 = msm->xm0;
  const float ym0 = msm->ym0;
  const float zm0 = msm->zm0;
  const float cutoff2 = cutoff * cutoff;
  const float invcut = 1.f / cutoff;
  long n, i, j, k;
  long extralen = 0;
  long num_excluded = 0;
  long mxRegionIndex, myRegionIndex, mzRegionIndex;
  long mxOffset, myOffset, mzOffset;
  long indexRegion, index;
  float *thisRegion;
  dim3 gridDim, blockDim;
  cudaStream_t shortrng_stream;
  int err = OK;

  /* perform geometric hashing of atoms into bins */
  memset(mc->binBase, 0, nbins * BIN_SIZE * sizeof(float));
  memset(mc->bincntBase, 0, nbins * sizeof(char));
  for (n = 0;  n < natoms;  n++) {
    float x, y, z, q;
    x = atom[4*n    ] - xm0;
    y = atom[4*n + 1] - ym0;
    z = atom[4*n + 2] - zm0;
    q = atom[4*n + 3];
    i = (long) floorf(x * BIN_INVLEN);
    j = (long) floorf(y * BIN_INVLEN);
    k = (long) floorf(z * BIN_INVLEN);
    if (i >= -nbpad && i < nxbins - nbpad &&
        j >= -nbpad && j < nybins - nbpad &&
        k >= -nbpad && k < nzbins - nbpad &&
        q != 0) {
      long index = (k * nybins + j) * nxbins + i;
      float *bin = binZero + index * BIN_SIZE;
      int bindex = bincntZero[index];
      if (bindex < BIN_DEPTH) {
        /* copy atom into bin and increase counter for this bin */
        bin[4*bindex    ] = x;
        bin[4*bindex + 1] = y;
        bin[4*bindex + 2] = z;
        bin[4*bindex + 3] = q;
        bincntZero[index]++;
      }
      else {
        /* add to array of extra atoms to be computed with CPU */
        if (extralen >= nbins) return ERROR(MSMPOT_ERROR_AVAIL);
        extra[4*extralen    ] = atom[4*n    ];
        extra[4*extralen + 1] = atom[4*n + 1];
        extra[4*extralen + 2] = atom[4*n + 2];
        extra[4*extralen + 3] = atom[4*n + 3];
        extralen++;
      }
    }
    else {
      /* excluded atoms are either outside bins or neutrally charged */
      num_excluded++;
    }
  }

  /* copy atom bins to device */
  cudaMemcpy(mc->dev_binBase, mc->binBase, nbins * BIN_SIZE * sizeof(float),
      cudaMemcpyHostToDevice);
  CUERR;

  gridDim.x = mxRegions;
  gridDim.y = myRegions;
  gridDim.z = 1;
  blockDim.x = 8;
  blockDim.y = 2;
  blockDim.z = 8;

  cudaStreamCreate(&shortrng_stream);  /* asynchronously invoke CUDA kernel */
  cuda_shortrange<<<gridDim, blockDim, 0>>>(nxbins, nybins, mc->dev_binZero,
      delta, cutoff2, invcut, mc->dev_padEpotmap, mzRegions);
  if (extralen > 0) {  /* call CPU to concurrently compute extra atoms */
    err = Msmpot_compute_shortrng(mc->msmpot, extra, extralen);
    if (err) return ERROR(err);
  }
  cudaStreamSynchronize(shortrng_stream);
  CUERR;
  cudaThreadSynchronize();
  cudaStreamDestroy(shortrng_stream);

  /* copy result regions from CUDA device */
  cudaMemcpy(padEpotmap, mc->dev_padEpotmap, pmall * sizeof(float),
      cudaMemcpyDeviceToHost);
  CUERR;

  /* transpose regions from padEpotmap and add into result epotmap */
  for (k = 0;  k < mz;  k++) {
    mzRegionIndex = (k >> 3);
    mzOffset = (k & 7);

    for (j = 0;  j < my;  j++) {
      myRegionIndex = (j >> 3);
      myOffset = (j & 7);

      for (i = 0;  i < mx;  i++) {
        mxRegionIndex = (i >> 3);
        mxOffset = (i & 7);

        thisRegion = padEpotmap
          + ((mzRegionIndex * myRegions + myRegionIndex) * mxRegions
              + mxRegionIndex) * REGION_SIZE;

        indexRegion = (mzOffset * 8 + myOffset) * 8 + mxOffset;
        index = (k * my + j) * mx + i;

        epotmap[index] += thisRegion[indexRegion];
      }
    }
  }

  return OK;
}

