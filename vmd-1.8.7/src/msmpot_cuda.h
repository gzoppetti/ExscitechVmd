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
 *      $RCSfile: msmpot_cuda.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.2 $      $Date: 2009/04/29 15:49:47 $
 *
 ***************************************************************************/
/*
 * msmcuda.h
 */


#ifndef MSMPOT_MSMCUDA_H
#define MSMPOT_MSMCUDA_H

/*
 * detect and report error from CUDA
 */
#undef  CUERR
#define CUERR \
  do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
      return ERROR(MSMPOT_ERROR_CUDA); \
    } \
  } while (0)


/*
 * Keep NBRLIST_MAXLEN of 3-tuples in GPU const cache memory:
 *   (3 * 5333) ints  +  1 int (giving use length)  ==  64000 bytes
 */
#undef  NBRLIST_MAXLEN
#define NBRLIST_MAXLEN  5333

/*
 * 4 floats per atom:  x/y/z/q
 */
#undef  ATOM_SIZE
#define ATOM_SIZE       4

/*
 * number of atoms per bin
 */
#undef  BIN_DEPTH
#define BIN_DEPTH       8

/*
 * number of floats per bin
 */
#undef  BIN_SIZE
#define BIN_SIZE        (ATOM_SIZE * BIN_DEPTH)


#ifdef __cplusplus
extern "C" {
#endif

  typedef struct MsmpotCuda_t {
    Msmpot *msmpot;

    /* get CUDA device info */
    struct cudaDeviceProp *dev;
    int ndevs;
    int devnum;           /* device number */

    /* CUDA short-range part ("binsmall") */
    long pmx, pmy, pmz;                /* dimensions of padded epotmap */
    long maxpm;                        /* allocated points for padded map */ 
    float *padEpotmap;                 /* padded epotmap for CUDA grid */

    float binlen;         /* dimension of atom bins, set for performance */
    float invbinlen;      /* 1/binlen */

    long nxbins, nybins, nzbins;       /* dimensions of atom bin array */
    long nbins;                        /* total number of atom bins */
    long nbpad;                        /* amount of bin padding */
    long maxbins;                      /* allocated number of atom bins */
    float *binBase;                    /* start of allocated atom bin array */
    float *binZero;                    /* shifted from base for bin(0,0,0) */
    char *bincntBase;                  /* count atoms in each bin */
    char *bincntZero;                  /* shifted */

    float *extra_atom;                 /* extra atoms, length 4*maxbins */
    long num_extra_atoms;              /* count number of extra atoms */
                          /* permit up to 1 extra atom per bin on average */

    int *nbrlist;         /* list of neighbor index offsets */
    int nbrlistlen;       /* used length (multiple of 3 for 3-tuples) */
    int nbrlistmax;       /* fixed maximum length is (3*NBRLIST_MAXLEN) */

    float *dev_padEpotmap;             /* points to device memory */
    long dev_maxpm;                    /* allocated points on device */

    float *dev_binBase;                /* points to device memory */
    float *dev_binZero;                /* points to bin(0,0,0) on device */
    long dev_maxbins;                  /* allocated bins on device */

    /* CUDA lattice cutoff */
    int   lk_nlevels;      /* number of levels for latcut kernel */
    int   lk_srad;         /* subcube radius for latcut kernel */
    int   lk_padding;      /* padding around internal array of subcubes */
    long  subcube_total;   /* total number of subcubes for compressed grids */
    long  block_total;     /* total number of thread blocks */
    /*
     * host_   -->  memory allocated on host
     * device_ -->  global memory allocated on device
     */
    int   *host_sinfo;     /* subcube info copy to device const mem */
    float *host_lfac;      /* level factor copy to device const mem */
    long maxlevels;

    float *host_wt;        /* weights copy to device const mem */
    long maxwts;

    float *host_qgrids;    /* q-grid subcubes copy to device global mem */
    float *host_egrids;    /* e-grid subcubes copy to device global mem */
    float *device_qgrids;  /* q-grid subcubes allocate on device */
    float *device_egrids;  /* e-grid subcubes allocate on device */
    long maxgridpts;

  } MsmpotCuda;


  MsmpotCuda *Msmpot_cuda_create(void);
  void Msmpot_cuda_destroy(MsmpotCuda *);

  int Msmpot_cuda_setup(MsmpotCuda *, Msmpot *);
  void Msmpot_cuda_cleanup(MsmpotCuda *);

  int Msmpot_cuda_setup_shortrng(MsmpotCuda *);
  void Msmpot_cuda_cleanup_shortrng(MsmpotCuda *);
  int Msmpot_cuda_compute_shortrng(MsmpotCuda *);

  int Msmpot_cuda_setup_latcut(MsmpotCuda *);
  void Msmpot_cuda_cleanup_latcut(MsmpotCuda *);
  int Msmpot_cuda_compute_latcut(MsmpotCuda *);
  int Msmpot_cuda_condense_qgrids(MsmpotCuda *);
  int Msmpot_cuda_expand_egrids(MsmpotCuda *);


#ifdef __cplusplus
}
#endif


#endif /* MSMPOT_MSMCUDA_H */
