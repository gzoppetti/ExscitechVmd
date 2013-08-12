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
 *      $RCSfile: msmpot.c,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.2 $      $Date: 2009/04/29 15:49:46 $
 *
 ***************************************************************************/
/*
 * msmpot.c
 */

#include "msmpot_internal.h"

#undef  NELEMS
#define NELEMS(a)  ((int)(sizeof(a)/sizeof(a[0])))


/* assume that the error numbers in msmpot.h correspond to these strings */
static const char *ERROR_STRING[] = {
  "none",
  "assertion failed",
  "memory allocation",
  "bad parameter",
  "exceeded available resources",
  "CUDA failure",
  "unknown",
};


/* assume that the "unknown" error is listed last */
const char *Msmpot_error_string(int err) {
  if (err > 0 || err <= -NELEMS(ERROR_STRING)) {
    err = -NELEMS(ERROR_STRING) + 1;
  }
  return ERROR_STRING[-err];
}


#ifdef MSMPOT_DEBUG
/* report error to stderr stream, return "err" */
int Msmpot_report_error(int err, const char *msg, const char *fn, int ln) {
  if (msg) {
    fprintf(stderr, "MSMPOT ERROR (%s,%d): %s, %s\n",
        fn, ln, Msmpot_error_string(err), msg);
  }
  else {
    fprintf(stderr, "MSMPOT ERROR (%s,%d): %s\n",
        fn, ln, Msmpot_error_string(err));
  }
  return err;
}
#endif


Msmpot *Msmpot_create(void) {
  Msmpot *msm = (Msmpot *) calloc(1, sizeof(Msmpot));
  if (NULL == msm) return NULL;
#ifdef MSMPOT_CUDA
  msm->msmcuda = Msmpot_cuda_create();
  if (NULL == msm->msmcuda) {
    Msmpot_destroy(msm);
    return NULL;
  }
#endif
  Msmpot_default(msm);
  return msm;
}


void Msmpot_destroy(Msmpot *msm) {
#ifdef MSMPOT_CUDA
  if (msm->msmcuda) Msmpot_cuda_destroy(msm->msmcuda);
#endif
  Msmpot_cleanup(msm);
  free(msm);
}



/*** MsmpotLattice ***********************************************************/


MsmpotLattice *Msmpot_lattice_create(void) {
  MsmpotLattice *p = (MsmpotLattice *) calloc(1, sizeof(MsmpotLattice));
  return p;
}

void Msmpot_lattice_destroy(MsmpotLattice *p) {
  free(p->buffer);
  free(p);
}

int Msmpot_lattice_setup(MsmpotLattice *p,
    long ia, long ib, long ja, long jb, long ka, long kb) {
  long ni = ib - ia + 1;
  long nj = jb - ja + 1;
  long n = ni * nj * (kb - ka + 1);
  ASSERT(ia <= ib);
  ASSERT(ja <= jb);
  ASSERT(ka <= kb);
  ASSERT(n > 0);
  if (n >= p->nbufsz) {
    float *buffer = (float *) realloc(p->buffer, n * sizeof(float));
    if (NULL==buffer) return ERROR(MSMPOT_ERROR_ALLOC);
    p->buffer = buffer;
    p->nbufsz = n;
  }
  p->ia = ia;
  p->ib = ib;
  p->ja = ja;
  p->jb = jb;
  p->ka = ka;
  p->kb = kb;
  p->ni = ni;
  p->nj = nj;
  p->data = p->buffer + INDEX(p,-ia,-ja,-ka);
  return OK;
}

int Msmpot_lattice_zero(MsmpotLattice *p) {
  long n = (p->ib - p->ia + 1) * (p->jb - p->ja + 1) * (p->kb - p->ka + 1);
  memset(p->buffer, 0,  n * sizeof(float));
  return OK;
}
