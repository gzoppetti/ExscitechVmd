#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mgpot_cuda.h"

#if defined(__MCUDA__)
#include <cuda_pthreads.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#endif

#undef DEBUGGING

#if !defined(__MCUDA__)
extern "C" 
#endif
int mgpot_cuda_device_list(void) {
  int deviceCount;
  int dev;

  deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  printf("Detected %d CUDA accelerators:\n", deviceCount);
  for (dev = 0;  dev < deviceCount;  dev++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("  CUDA device[%d]: '%s'  Mem: %dMB  Rev: %d.%d\n",
        dev, deviceProp.name, deviceProp.totalGlobalMem / (1024*1024),
        deviceProp.major, deviceProp.minor);
  }
  return deviceCount;
}


#if !defined(__MCUDA__)
extern "C" 
#endif
int mgpot_cuda_device_set(int devnum) {
  printf("Opening CUDA device %d...\n", devnum);
  cudaSetDevice(devnum);
  CUERR;  /* check and clear any existing errors */
  return 0;
}


#if !defined(__MCUDA__)
extern "C" 
#endif
int mgpot_cuda_setup_latcut(Mgpot *mg) {
  const float h = mg->h;
  const float a = mg->a;
  const int split = mg->split;
  int nlevels = mg->nlevels - 1;  /* don't do top level on GPU */
  int nrad;
  int srad;
  int pad;
  int i, j, k, ii, jj, kk;
  int index;
  long btotal, stotal, memsz;
  float s, t, gs, gt;
  float lfac;
  float *wt;

  if (nlevels > MAXLEVELS) {
    return ERROR("number of levels %d exceeds maximum %d\n",
        nlevels, MAXLEVELS);
  }
  mg->lk_nlevels = nlevels;
  nrad = (int) ceilf(2*a/h) - 1;
  srad = (int) ceilf((nrad + 1) / 4.f);
  if (srad > 3) {
    return ERROR("subcube radius %d exceeds maximum radius %d\n",
        srad, 3);
  }
  mg->lk_srad = srad;
  pad = 1;  /* for non-periodic systems */
  mg->lk_padding = pad;
#ifdef DEBUGGING
  printf("a=%g  h=%g\n", a, h);
  printf("nrad=%d\n", nrad);
  printf("srad=%d\n", srad);
  printf("padding=%d\n", padding);
#endif

  mg->host_lfac = (float *) calloc(nlevels, sizeof(float));
  if (NULL==mg->host_lfac) return FAIL;
  lfac = 1.f;
  for (i = 0;  i < nlevels;  i++) {
    mg->host_lfac[i] = lfac;
    lfac *= 0.5f;
  }

  mg->host_sinfo = (int *) calloc(4 * nlevels, sizeof(int));
  if (NULL==mg->host_sinfo) return FAIL;
  stotal = 0;
  btotal = 0;
  for (i = 0;  i < nlevels;  i++) {
    /* determine lattice dimensions measured in subcubes */
    const floatLattice *f = mg->qgrid[i];
    int nx = mg->host_sinfo[ INDEX_X(i) ] = (int) ceilf(f->ni / 4.f) + 2*pad;
    int ny = mg->host_sinfo[ INDEX_Y(i) ] = (int) ceilf(f->nj / 4.f) + 2*pad;
    int nz = mg->host_sinfo[ INDEX_Z(i) ] = (int) ceilf(f->nk / 4.f) + 2*pad;
    stotal += nx * ny * nz;
    btotal += (nx - 2*pad) * (ny - 2*pad) * (nz - 2*pad);
    mg->host_sinfo[ INDEX_Q(i) ] = btotal;

    printf("\nlevel %d:  ni=%2d  nj=%2d  nk=%2d\n", i, f->ni, f->nj, f->nk);
    printf("          nx=%2d  ny=%2d  nz=%2d  stotal=%d\n",
        nx, ny, nz, stotal);
    printf("          bx=%2d  by=%2d  bz=%2d  btotal=%d\n",
        nx-2*pad, ny-2*pad, nz-2*pad, btotal);
  }
  printf("\n");
  /* stotal counts total number of subcubes for collapsed grid hierarchy */
  mg->subcube_total = stotal;
  mg->block_total = btotal;
  //printf("stotal=%d\n", stotal);
  //printf("btotal=%d\n", btotal);

#ifdef DEBUGGING
  printf("nlevels=%d\n", nlevels);
  for (i = 0;  i < nlevels;  i++) {
    printf("ni=%d  nj=%d  nk=%d\n",
        mg->qgrid[i]->ni, mg->qgrid[i]->nj, mg->qgrid[i]->nk);
    printf("nx=%d  ny=%d  nz=%d  nw=%d\n",
        mg->host_sinfo[ INDEX_X(i) ],
        mg->host_sinfo[ INDEX_Y(i) ],
        mg->host_sinfo[ INDEX_Z(i) ],
        mg->host_sinfo[ INDEX_Q(i) ]);
  }
#endif

  /* allocate and calculate weights for lattice cutoff */
  mg->host_wt = (float *) calloc((8*srad) * (8*srad) * (8*srad), sizeof(float));
  if (NULL==mg->host_wt) return FAIL;
  wt = mg->host_wt;
  for (kk = 0;  kk < 8*srad;  kk++) {
    for (jj = 0;  jj < 8*srad;  jj++) {
      for (ii = 0;  ii < 8*srad;  ii++) {
        index = (kk*(8*srad) + jj)*(8*srad) + ii;
        i = ii - 4*srad;  /* distance (in grid points) from center */
        j = jj - 4*srad;
        k = kk - 4*srad;
        s = (i*i + j*j + k*k) * h*h / (a*a);
        t = 0.25f * s;
        if (t >= 1) {
          wt[index] = 0;
        }
        else if (s >= 1) {
          gs = 1/sqrtf(s);
          mgpot_split(&gt, t, split);
          wt[index] = (gs - 0.5f * gt) / a;
        }
        else {
          mgpot_split(&gs, s, split);
          mgpot_split(&gt, t, split);
          wt[index] = (gs - 0.5f * gt) / a;
        }
      }
    }
  }

  /* allocate host memory flat array of subcubes */
  memsz = stotal * SUBCUBESZ * sizeof(float);
  mg->host_qgrids = (float *) malloc(memsz);
  if (NULL==mg->host_qgrids) return FAIL;
  mg->host_egrids = (float *) malloc(memsz);
  if (NULL==mg->host_egrids) return FAIL;

  /* allocate device global memory flat array of subcubes */
  printf("Allocating %.2fMB of device memory for grid hierarchy...\n",
      (2.f * memsz) / (1024.f * 1024.f));
  cudaMalloc((void **) &(mg->device_qgrids), memsz);
  CUERR;  /* check and clear any existing errors */
  cudaMalloc((void **) &(mg->device_egrids), memsz);
  CUERR;  /* check and clear any existing errors */

  return 0;
}

#if !defined(__MCUDA__)
extern "C" 
#endif
int mgpot_cuda_cleanup_latcut(Mgpot *mg) {
  free(mg->host_lfac);  /* free host memory allocations */
  free(mg->host_sinfo);
  free(mg->host_wt);
  free(mg->host_qgrids);
  free(mg->host_egrids);
  cudaFree(mg->device_qgrids);  /* free device memory allocations */
  cudaFree(mg->device_egrids);
  return 0;
}


/* condense q grid hierarchy into flat array of subcubes */
#if !defined(__MCUDA__)
extern "C" 
#endif
int mgpot_cuda_condense_qgrids(Mgpot *mg) {
  const int *host_sinfo = mg->host_sinfo;
  float *host_qgrids = mg->host_qgrids;

  const long memsz = mg->subcube_total * SUBCUBESZ * sizeof(float);
  const int nlevels = mg->lk_nlevels;
  const int pad = mg->lk_padding;
  int level, in, jn, kn, i, j, k;
  int isrc, jsrc, ksrc, subcube_index, grid_index, off;

  //printf("4\n");
  memset(host_qgrids, 0, memsz);  /* zero the qgrids subcubes */

  //printf("5\n");
  off = 0;
  for (level = 0;  level < nlevels;  level++) {
    const floatLattice *qgrid = mg->qgrid[level];
    const float *qbuffer = qgrid->buffer;

    const int nx = host_sinfo[ INDEX_X(level) ];
    const int ny = host_sinfo[ INDEX_Y(level) ];
    const int nz = host_sinfo[ INDEX_Z(level) ];
    //const int nw = host_sinfo[ INDEX_Q(level) ] - nx*ny*nz;

#ifdef DEBUGGING
    printf("level=%d\n", level);
    printf("  nx=%d  ny=%d  nz=%d\n", nx, ny, nz);
    printf("  ni=%d  nj=%d  nk=%d\n", qgrid->ni, qgrid->nj, qgrid->nk);
#endif

    for (kn = pad;  kn < nz-pad;  kn++) {
      for (jn = pad;  jn < ny-pad;  jn++) {
        for (in = pad;  in < nx-pad;  in++) {

          for (k = 0;  k < 4;  k++) {
            ksrc = (kn-pad)*4 + k;
            if (ksrc >= qgrid->nk) break;

            for (j = 0;  j < 4;  j++) {
              jsrc = (jn-pad)*4 + j;
              if (jsrc >= qgrid->nj) break;

              for (i = 0;  i < 4;  i++) {
                isrc = (in-pad)*4 + i;
                if (isrc >= qgrid->ni) break;

                grid_index = (ksrc * qgrid->nj + jsrc) * qgrid->ni + isrc;
                subcube_index = (((kn*ny + jn)*nx + in) + off) * SUBCUBESZ
                  + (k*4 + j)*4 + i;

                host_qgrids[subcube_index] = qbuffer[grid_index];
              }
            }
          } /* loop over points in a subcube */

        }
      }
    } /* loop over subcubes in a level */

    off += nx * ny * nz;  /* offset to next level */

  } /* loop over levels */

  return 0;
}


/* expand flat array of subcubes into e grid hierarchy */
#if !defined(__MCUDA__)
extern "C" 
#endif
int mgpot_cuda_expand_egrids(Mgpot *mg) {
  const int *host_sinfo = mg->host_sinfo;
  const float *host_egrids = mg->host_egrids;

  const int nlevels = mg->lk_nlevels;
  const int pad = mg->lk_padding;
  int level, in, jn, kn, i, j, k;
  int isrc, jsrc, ksrc, subcube_index, grid_index, off;

  off = 0;
  for (level = 0;  level < nlevels;  level++) {
    floatLattice *egrid = mg->egrid[level];
    float *ebuffer = egrid->buffer;

    const int nx = host_sinfo[ INDEX_X(level) ];
    const int ny = host_sinfo[ INDEX_Y(level) ];
    const int nz = host_sinfo[ INDEX_Z(level) ];
    //const int nw = host_sinfo[ INDEX_Q(level) ] - nx*ny*nz;

    for (kn = pad;  kn < nz-pad;  kn++) {
      for (jn = pad;  jn < ny-pad;  jn++) {
        for (in = pad;  in < nx-pad;  in++) {

          for (k = 0;  k < 4;  k++) {
            ksrc = (kn-pad)*4 + k;
            if (ksrc >= egrid->nk) break;

            for (j = 0;  j < 4;  j++) {
              jsrc = (jn-pad)*4 + j;
              if (jsrc >= egrid->nj) break;

              for (i = 0;  i < 4;  i++) {
                isrc = (in-pad)*4 + i;
                if (isrc >= egrid->ni) break;

                grid_index = (ksrc * egrid->nj + jsrc) * egrid->ni + isrc;
                subcube_index = (((kn*ny + jn)*nx + in) + off) * SUBCUBESZ
                  + (k*4 + j)*4 + i;

                ebuffer[grid_index] = host_egrids[subcube_index];
              }
            }
          } /* loop over points in a subcube */

        }
      }
    } /* loop over subcubes in a level */

    off += nx * ny * nz;  /* offset to level */

  } /* loop over levels */

  return 0;
}
