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
 *      $RCSfile: CUDAOrbital.cu,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.82 $      $Date: 2009/07/10 23:37:08 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   This source file contains the CUDA kernels for computing molecular
 *  orbital amplitudes on a uniformly spaced grid, using one ore more GPUs.
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#include "VMDThreads.h"
#include "CUDAKernels.h"

#if defined(VMDTHREADS)
#define USE_CUDADEVPOOL 1
#endif

// #define USE_PINNED_MEMORY 1

// #define BUILD_STANDALONE 1
#if defined(BUILD_STANDALONE)
#include "rworbitals.h"
#endif

// Only enabled for testing an emulation of JIT code generation approach
// #define VMDMOJIT 1
// #define VMDMOJITSRC "/home/johns/mojit.cu"

#if 1
#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
  printf("Thread aborting...\n"); \
  return NULL; }}
#else
#define CUERR
#endif

#define ANGS_TO_BOHR 1.8897259877218677f

typedef union flint_t {
  float f;
  int i;
} flint;

typedef struct {
  int numatoms;
  const float *wave_f; 
  int num_wave_f;
  const float *basis_array;
  int num_basis;
  const float *atompos;
  const int *atom_basis;
  const int *num_shells_per_atom;
  const int *num_prim_per_shell;
  const int *shell_symmetry;
  int num_shells;
  const int *numvoxels;
  float voxelsize;
  const float *origin;
  float *orbitalgrid;
} orbthrparms;

/* thread prototype */
static void * cudaorbitalthread(void *);

// GPU block layout
#define UNROLLX       1
#define UNROLLY       1
#define BLOCKSIZEX    8 
#define BLOCKSIZEY    8
#define BLOCKSIZE     BLOCKSIZEX * BLOCKSIZEY

// required GPU array padding to match thread block size
#define TILESIZEX BLOCKSIZEX*UNROLLX
#define TILESIZEY BLOCKSIZEY*UNROLLY
#define GPU_X_ALIGNMASK (TILESIZEX - 1)
#define GPU_Y_ALIGNMASK (TILESIZEY - 1)

#define MEMCOALESCE  384

// orbital shell types 
#define S_SHELL 0
#define P_SHELL 1
#define D_SHELL 2
#define F_SHELL 3
#define G_SHELL 4
#define H_SHELL 5

//
// Constant arrays to store 
//
#define MAX_ATOM_SZ 256

#define MAX_ATOMPOS_SZ (MAX_ATOM_SZ)
__constant__ static float const_atompos[MAX_ATOMPOS_SZ * 3];

#define MAX_ATOM_BASIS_SZ (MAX_ATOM_SZ)
__constant__ static int const_atom_basis[MAX_ATOM_BASIS_SZ];

#define MAX_ATOMSHELL_SZ (MAX_ATOM_SZ)
__constant__ static int const_num_shells_per_atom[MAX_ATOMSHELL_SZ];

#define MAX_BASIS_SZ 6144 
__constant__ static float const_basis_array[MAX_BASIS_SZ];

#define MAX_SHELL_SZ 1024
__constant__ static int const_num_prim_per_shell[MAX_SHELL_SZ];
__constant__ static int const_shell_symmetry[MAX_SHELL_SZ];

#define MAX_WAVEF_SZ 6144
__constant__ static float const_wave_f[MAX_WAVEF_SZ];



//
// CUDA using const memory for almost all of the key arrays
//
__global__ static void cuorbitalconstmem(int numatoms, 
                          float voxelsize, 
                          float originx,
                          float originy,
                          float grid_z, 
                          float * orbitalgrid) {
  unsigned int xindex  = __umul24(blockIdx.x, blockDim.x) * UNROLLX
                         + threadIdx.x;
  unsigned int yindex  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int outaddr = (__umul24(gridDim.x, blockDim.x) * UNROLLX) * yindex
                         + xindex;
  float grid_x = originx + voxelsize * xindex;
  float grid_y = originy + voxelsize * yindex;

  // similar to C version
  int at;
  int prim, shell;

  // initialize value of orbital at gridpoint
  float value = 0.0f;

  // initialize the wavefunction and shell counters
  int ifunc = 0;
  int shell_counter = 0;

  // loop over all the QM atoms
  for (at = 0; at < numatoms; at++) {
    // calculate distance between grid point and center of atom
    int maxshell = const_num_shells_per_atom[at];
    int prim_counter = const_atom_basis[at];

    float xdist = (grid_x - const_atompos[3*at  ])*ANGS_TO_BOHR;
    float ydist = (grid_y - const_atompos[3*at+1])*ANGS_TO_BOHR;
    float zdist = (grid_z - const_atompos[3*at+2])*ANGS_TO_BOHR;

    float xdist2 = xdist*xdist;
    float ydist2 = ydist*ydist;
    float zdist2 = zdist*zdist;

    float dist2 = xdist2 + ydist2 + zdist2;

    // loop over the shells belonging to this atom (or basis function)
    for (shell=0; shell < maxshell; shell++) {
      float contracted_gto = 0.0f;

      // Loop over the Gaussian primitives of this contracted
      // basis function to build the atomic orbital
      int maxprim = const_num_prim_per_shell[shell_counter];
      int shell_type = const_shell_symmetry[shell_counter];
      for (prim=0; prim < maxprim;  prim++) {
        float exponent       = const_basis_array[prim_counter    ];
        float contract_coeff = const_basis_array[prim_counter + 1];

        // By premultiplying the stored exponent factors etc,
        // we can use exp2f() rather than exp(), giving us full
        // precision, but with the speed of __expf()
        contracted_gto += contract_coeff * exp2f(-exponent*dist2);
        prim_counter += 2;
      }

      /* multiply with the appropriate wavefunction coefficient */
      float tmpshell=0;
      switch (shell_type) {
        case S_SHELL:
          value += const_wave_f[ifunc++] * contracted_gto;
          break;

        case P_SHELL:
          tmpshell += const_wave_f[ifunc++] * xdist;
          tmpshell += const_wave_f[ifunc++] * ydist;
          tmpshell += const_wave_f[ifunc++] * zdist;
          value += tmpshell * contracted_gto;
          break;

        case D_SHELL:
          tmpshell += const_wave_f[ifunc++] * xdist2;
          tmpshell += const_wave_f[ifunc++] * xdist * ydist;
          tmpshell += const_wave_f[ifunc++] * ydist2;
          tmpshell += const_wave_f[ifunc++] * xdist * zdist;
          tmpshell += const_wave_f[ifunc++] * ydist * zdist;
          tmpshell += const_wave_f[ifunc++] * zdist2;
          value += tmpshell * contracted_gto;
          break;

        case F_SHELL:
          tmpshell += const_wave_f[ifunc++] * xdist2 * xdist;
          tmpshell += const_wave_f[ifunc++] * xdist2 * ydist;
          tmpshell += const_wave_f[ifunc++] * ydist2 * xdist;
          tmpshell += const_wave_f[ifunc++] * ydist2 * ydist;
          tmpshell += const_wave_f[ifunc++] * xdist2 * zdist;
          tmpshell += const_wave_f[ifunc++] * xdist * ydist * zdist;
          tmpshell += const_wave_f[ifunc++] * ydist2 * zdist;
          tmpshell += const_wave_f[ifunc++] * zdist2 * xdist;
          tmpshell += const_wave_f[ifunc++] * zdist2 * ydist;
          tmpshell += const_wave_f[ifunc++] * zdist2 * zdist;
          value += tmpshell * contracted_gto;
          break;

        case G_SHELL:
          tmpshell += const_wave_f[ifunc++] * xdist2 * xdist2; // xxxx
          tmpshell += const_wave_f[ifunc++] * xdist2 * xdist * ydist;  // xxxy
          tmpshell += const_wave_f[ifunc++] * xdist2 * ydist2; // xxyy
          tmpshell += const_wave_f[ifunc++] * ydist2 * ydist * xdist;  // xyyy
          tmpshell += const_wave_f[ifunc++] * ydist2 * ydist2; // yyyy
          tmpshell += const_wave_f[ifunc++] * xdist2 * xdist * zdist; // xxxz
          tmpshell += const_wave_f[ifunc++] * xdist2 * ydist * zdist; // xxyz
          tmpshell += const_wave_f[ifunc++] * ydist2 * xdist * zdist; // xyyz
          tmpshell += const_wave_f[ifunc++] * ydist2 * ydist * zdist; // yyyz
          tmpshell += const_wave_f[ifunc++] * xdist2 * zdist2; // xxzz
          tmpshell += const_wave_f[ifunc++] * zdist2 * xdist * ydist; // xyzz
          tmpshell += const_wave_f[ifunc++] * ydist2 * zdist2; // yyzz
          tmpshell += const_wave_f[ifunc++] * zdist2 * zdist * xdist; // zzzx
          tmpshell += const_wave_f[ifunc++] * zdist2 * zdist * ydist; // zzzy
          tmpshell += const_wave_f[ifunc++] * zdist2 * zdist2; // zzzz
          value += tmpshell * contracted_gto;
          break;
      } // end switch

      shell_counter++;
    }
  }

  orbitalgrid[outaddr] = value;
}



#if defined(VMDMOJIT)

//
// If we're testing performance of a JIT kernel, include the code here
//
#if defined(VMDMOJITSRC)
#include VMDMOJITSRC
#endif

//
// CUDA JIT code generator for constant memory
//
int cuda_jit_generate(const char * srcfilename, int numatoms,
                      const float *wave_f, const float *basis_array,
                      const int *atom_basis,
                      const int *num_shells_per_atom,
                      const int *num_prim_per_shell,
                      const int *shell_symmetry) {
  FILE *ofp=NULL;
  if (srcfilename) 
    ofp=fopen(srcfilename, "w");

  if (ofp == NULL)
    ofp=stdout; 

  // calculate the value of the wavefunction of the
  // selected orbital at the current grid point
  int at;
  int prim, shell;

  // initialize the wavefunction and shell counters
  int shell_counter = 0;

  fprintf(ofp, 
    "__global__ static void cuorbitalconstmem_jit(int numatoms,\n"
    "                          float voxelsize,\n"
    "                          float originx,\n"
    "                          float originy,\n"
    "                          float grid_z, \n"
    "                          float * orbitalgrid) {\n"
    "  unsigned int xindex  = __umul24(blockIdx.x, blockDim.x) * UNROLLX\n"
    "                         + threadIdx.x;\n"
    "  unsigned int yindex  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;\n"
    "  unsigned int outaddr = (__umul24(gridDim.x, blockDim.x) * UNROLLX) * yindex\n"
    "                         + xindex;\n"
    "  float grid_x = originx + voxelsize * xindex;\n"
    "  float grid_y = originy + voxelsize * yindex;\n"
 
    "  // similar to C version\n"
    "  int at;\n"
    "  // initialize value of orbital at gridpoint\n"
    "  float value = 0.0f;\n"
    "  // initialize the wavefunction and shell counters\n"
    "  int ifunc = 0;\n"
    "  // loop over all the QM atoms\n"
    "  for (at = 0; at < numatoms; at++) {\n"
    "    // calculate distance between grid point and center of atom\n"
//    "    int maxshell = const_num_shells_per_atom[at];\n"
//    "    int prim_counter = const_atom_basis[at];\n"
    "    float xdist = (grid_x - const_atompos[3*at  ])*ANGS_TO_BOHR;\n"
    "    float ydist = (grid_y - const_atompos[3*at+1])*ANGS_TO_BOHR;\n"
    "    float zdist = (grid_z - const_atompos[3*at+2])*ANGS_TO_BOHR;\n"
    "    float xdist2 = xdist*xdist;\n"
    "    float ydist2 = ydist*ydist;\n"
    "    float zdist2 = zdist*zdist;\n"
    "    float dist2 = xdist2 + ydist2 + zdist2;\n"
    "    float contracted_gto=0.0f;\n"
    "    float tmpshell=0.0f;\n"
    "\n"
  );

#if 0
  // loop over all the QM atoms generating JIT code for each type
  for (at=0; at<numatoms; at++) {
#else
  // generate JIT code for one atom type and assume they are all the same
  for (at=0; at<1; at++) {
#endif
    int maxshell = num_shells_per_atom[at];
    int prim_counter = atom_basis[at];

    // loop over the shells belonging to this atom
    for (shell=0; shell < maxshell; shell++) {
      // Loop over the Gaussian primitives of this contracted
      // basis function to build the atomic orbital
      int maxprim = num_prim_per_shell[shell_counter];
      int shell_type = shell_symmetry[shell_counter];
      for (prim=0; prim<maxprim; prim++) {
        float exponent       = basis_array[prim_counter    ];
        float contract_coeff = basis_array[prim_counter + 1];
        if (prim == 0) {
          fprintf(ofp, "    contracted_gto = %f * expf(-%f*dist2);\n",
                  contract_coeff, exponent);
        } else {
          fprintf(ofp, "    contracted_gto += %f * expf(-%f*dist2);\n",
                  contract_coeff, exponent);
        }
        prim_counter += 2;
      }

      /* multiply with the appropriate wavefunction coefficient */
      switch (shell_type) {
        case S_SHELL:
          fprintf(ofp, 
            "    // S_SHELL\n"
            "    value += const_wave_f[ifunc++] * contracted_gto;\n");
          break;

        case P_SHELL:
          fprintf(ofp,
            "    // P_SHELL\n"
            "    tmpshell = const_wave_f[ifunc++] * xdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist;\n"
            "    tmpshell += const_wave_f[ifunc++] * zdist;\n"
            "    value += tmpshell * contracted_gto;\n"
          );
          break;

        case D_SHELL:
          fprintf(ofp,
            "    // D_SHELL\n"
            "    tmpshell = const_wave_f[ifunc++] * xdist2;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist * ydist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist2;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist * zdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist * zdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * zdist2;\n"
            "    value += tmpshell * contracted_gto;\n"
          );
          break;

        case F_SHELL:
          fprintf(ofp,
            "    // F_SHELL\n"
            "    tmpshell = const_wave_f[ifunc++] * xdist2 * xdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist2 * ydist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist2 * xdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist2 * ydist;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist2 * zdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist * ydist * zdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist2 * zdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * zdist2 * xdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * zdist2 * ydist;\n"
            "    tmpshell += const_wave_f[ifunc++] * zdist2 * zdist;\n"
            "    value += tmpshell * contracted_gto;\n"
          );
          break;

        case G_SHELL:
          fprintf(ofp,
            "    // G_SHELL\n"
            "    tmpshell = const_wave_f[ifunc++] * xdist2 * xdist2;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist2 * xdist * ydist;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist2 * ydist2;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist2 * ydist * xdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist2 * ydist2;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist2 * xdist * zdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist2 * ydist * zdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist2 * xdist * zdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist2 * ydist * zdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * xdist2 * zdist2;\n"
            "    tmpshell += const_wave_f[ifunc++] * zdist2 * xdist * ydist;\n"
            "    tmpshell += const_wave_f[ifunc++] * ydist2 * zdist2;\n"
            "    tmpshell += const_wave_f[ifunc++] * zdist2 * zdist * xdist;\n"
            "    tmpshell += const_wave_f[ifunc++] * zdist2 * zdist * ydist;\n"
            "    tmpshell += const_wave_f[ifunc++] * zdist2 * zdist2;\n"
            "    value += tmpshell * contracted_gto;\n"
          );
          break;

      } // end switch
      fprintf(ofp, "\n");

      shell_counter++;
    } // end shell
  } // end atom

  fprintf(ofp, 
    "  }\n"
    "\n"
    "  orbitalgrid[outaddr] = value;\n"
    "}\n"
  );

  if (ofp != stdout)
    fclose(ofp);

  return 0;
}

#endif // VMDMOJIT



//
// CUDA loading from global memory into shared memory for all arrays
//

// Shared memory tiling parameters:
//   We need a tile size of at least 16 elements for coalesced memory access,
// and it must be the next power of two larger than the largest number of 
// wavefunction coefficients that will be consumed at once for a given 
// shell type.  The current code is designed to handle up to "G" shells 
// (15 coefficients), since that's the largest shell type supported by
// GAMESS, which is the only format we can presently read.

// The maximum shell coefficient count specifies the largest number
// of coefficients that might have to be read for a single shell,
// e.g. for the highest supported shell type.
// This is a "G" shell coefficient count
#define MAXSHELLCOUNT 15   

// Mask out the lower N bits of the array index
// to compute which tile to start loading shared memory with.
// This must be large enough to gaurantee coalesced addressing, but
// be half or less of the minimum tile size
#define MEMCOAMASK  (~15)

// The shared memory basis set and wavefunction coefficient arrays 
// must contain a minimum of at least two tiles, but can be any 
// larger multiple thereof which is also a multiple of the thread block size.
// Larger arrays reduce duplicative data loads/fragmentation 
// at the beginning and end of the array
#define SHAREDSIZE 256


__global__ static void cuorbitaltiledshared(int numatoms, 
                          const float *wave_f,
                          const float *basis_array,
                          const flint *atominfo, 
                          const int *shellinfo, 
                          float voxelsize, 
                          float originx,
                          float originy,
                          float grid_z, 
                          float * orbitalgrid) {
  unsigned int xindex  = __umul24(blockIdx.x, blockDim.x) * UNROLLX
                         + threadIdx.x;
  unsigned int yindex  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int outaddr = (__umul24(gridDim.x, blockDim.x) * UNROLLX) * yindex
                         + xindex;
  float grid_x = originx + voxelsize * xindex;
  float grid_y = originy + voxelsize * yindex;

  int sidx = __mul24(threadIdx.y, blockDim.x) + threadIdx.x;

  __shared__ float s_wave_f[SHAREDSIZE];
  int sblock_wave_f = 0;
  s_wave_f[sidx      ] = wave_f[sidx      ];
  s_wave_f[sidx +  64] = wave_f[sidx +  64];
  s_wave_f[sidx + 128] = wave_f[sidx + 128];
  s_wave_f[sidx + 192] = wave_f[sidx + 192];
  __syncthreads();

  // similar to C version
  int at;
  int prim, shell;

  // initialize value of orbital at gridpoint
  float value = 0.0f;

  // initialize the wavefunction and primitive counters
  int ifunc = 0;
  int shell_counter = 0;
  int sblock_prim_counter = -1; // sentinel value to indicate no data loaded
  // loop over all the QM atoms
  for (at = 0; at < numatoms; at++) {
    __shared__ flint s_atominfo[5];
    __shared__ float s_basis_array[SHAREDSIZE];

    __syncthreads();
    if (sidx < 5)
      s_atominfo[sidx].i = atominfo[(at<<4) + sidx].i;
    __syncthreads();

    int prim_counter = s_atominfo[3].i;
    int maxshell     = s_atominfo[4].i;
    int new_sblock_prim_counter = prim_counter & MEMCOAMASK;
    if (sblock_prim_counter != new_sblock_prim_counter) {
      sblock_prim_counter = new_sblock_prim_counter;
      s_basis_array[sidx      ] = basis_array[sblock_prim_counter + sidx      ];
      s_basis_array[sidx +  64] = basis_array[sblock_prim_counter + sidx +  64];
      s_basis_array[sidx + 128] = basis_array[sblock_prim_counter + sidx + 128];
      s_basis_array[sidx + 192] = basis_array[sblock_prim_counter + sidx + 192];
      prim_counter -= sblock_prim_counter;
      __syncthreads();
    }

    // calculate distance between grid point and center of atom
    float xdist = (grid_x - s_atominfo[0].f)*ANGS_TO_BOHR;
    float ydist = (grid_y - s_atominfo[1].f)*ANGS_TO_BOHR;
    float zdist = (grid_z - s_atominfo[2].f)*ANGS_TO_BOHR;

    float xdist2 = xdist*xdist;
    float ydist2 = ydist*ydist;
    float zdist2 = zdist*zdist;

    float dist2 = xdist2 + ydist2 + zdist2;

    // loop over the shells belonging to this atom (or basis function)
    for (shell=0; shell < maxshell; shell++) {
      float contracted_gto = 0.0f;

      // Loop over the Gaussian primitives of this contracted
      // basis function to build the atomic orbital
      __shared__ int s_shellinfo[2];
     
      __syncthreads();
      if (sidx < 2)
        s_shellinfo[sidx] = shellinfo[(shell_counter<<4) + sidx];
      __syncthreads();

      int maxprim = s_shellinfo[0];
      int shell_type = s_shellinfo[1];

      if ((prim_counter + (maxprim<<1)) >= SHAREDSIZE) {
        prim_counter += sblock_prim_counter;
        sblock_prim_counter = prim_counter & MEMCOAMASK;
        s_basis_array[sidx      ] = basis_array[sblock_prim_counter + sidx      ];
        s_basis_array[sidx +  64] = basis_array[sblock_prim_counter + sidx +  64];
        s_basis_array[sidx + 128] = basis_array[sblock_prim_counter + sidx + 128];
        s_basis_array[sidx + 192] = basis_array[sblock_prim_counter + sidx + 192];
        prim_counter -= sblock_prim_counter;
        __syncthreads();
      } 
      for (prim=0; prim < maxprim;  prim++) {
        float exponent       = s_basis_array[prim_counter    ];
        float contract_coeff = s_basis_array[prim_counter + 1];

        // By premultiplying the stored exponent factors etc,
        // we can use exp2f() rather than exp(), giving us full
        // precision, but with the speed of __expf()
        contracted_gto += contract_coeff * exp2f(-exponent*dist2);

        prim_counter += 2;
      }

      // XXX should use a constant memory lookup table to store
      // shared mem refill constants, and dynamically lookup the
      // number of elements referenced in the next iteration.
      if ((ifunc + MAXSHELLCOUNT) >= SHAREDSIZE) { 
        ifunc += sblock_wave_f;
        sblock_wave_f = ifunc & MEMCOAMASK;
        __syncthreads();
        s_wave_f[sidx      ] = wave_f[sblock_wave_f + sidx      ];
        s_wave_f[sidx +  64] = wave_f[sblock_wave_f + sidx +  64];
        s_wave_f[sidx + 128] = wave_f[sblock_wave_f + sidx + 128];
        s_wave_f[sidx + 192] = wave_f[sblock_wave_f + sidx + 192];
        __syncthreads();
        ifunc -= sblock_wave_f;
      }

      /* multiply with the appropriate wavefunction coefficient */
      float tmpshell=0;
      switch (shell_type) {
        case S_SHELL:
          value += s_wave_f[ifunc++] * contracted_gto;
          break;

        case P_SHELL:
          tmpshell += s_wave_f[ifunc++] * xdist;
          tmpshell += s_wave_f[ifunc++] * ydist;
          tmpshell += s_wave_f[ifunc++] * zdist;
          value += tmpshell * contracted_gto;
          break;

        case D_SHELL:
          tmpshell += s_wave_f[ifunc++] * xdist2;
          tmpshell += s_wave_f[ifunc++] * xdist * ydist;
          tmpshell += s_wave_f[ifunc++] * ydist2;
          tmpshell += s_wave_f[ifunc++] * xdist * zdist;
          tmpshell += s_wave_f[ifunc++] * ydist * zdist;
          tmpshell += s_wave_f[ifunc++] * zdist2;
          value += tmpshell * contracted_gto;
          break;

        case F_SHELL:
          tmpshell += s_wave_f[ifunc++] * xdist2 * xdist;
          tmpshell += s_wave_f[ifunc++] * xdist2 * ydist;
          tmpshell += s_wave_f[ifunc++] * ydist2 * xdist;
          tmpshell += s_wave_f[ifunc++] * ydist2 * ydist;
          tmpshell += s_wave_f[ifunc++] * xdist2 * zdist;
          tmpshell += s_wave_f[ifunc++] * xdist * ydist * zdist;
          tmpshell += s_wave_f[ifunc++] * ydist2 * zdist;
          tmpshell += s_wave_f[ifunc++] * zdist2 * xdist;
          tmpshell += s_wave_f[ifunc++] * zdist2 * ydist;
          tmpshell += s_wave_f[ifunc++] * zdist2 * zdist;
          value += tmpshell * contracted_gto;
          break;

        case G_SHELL:
          tmpshell += s_wave_f[ifunc++] * xdist2 * xdist2; // xxxx
          tmpshell += s_wave_f[ifunc++] * xdist2 * xdist * ydist;  // xxxy
          tmpshell += s_wave_f[ifunc++] * xdist2 * ydist2; // xxyy
          tmpshell += s_wave_f[ifunc++] * ydist2 * ydist * xdist;  // xyyy
          tmpshell += s_wave_f[ifunc++] * ydist2 * ydist2; // yyyy
          tmpshell += s_wave_f[ifunc++] * xdist2 * xdist * zdist; // xxxz
          tmpshell += s_wave_f[ifunc++] * xdist2 * ydist * zdist; // xxyz
          tmpshell += s_wave_f[ifunc++] * ydist2 * xdist * zdist; // xyyz
          tmpshell += s_wave_f[ifunc++] * ydist2 * ydist * zdist; // yyyz
          tmpshell += s_wave_f[ifunc++] * xdist2 * zdist2; // xxzz
          tmpshell += s_wave_f[ifunc++] * zdist2 * xdist * ydist; // xyzz
          tmpshell += s_wave_f[ifunc++] * ydist2 * zdist2; // yyzz
          tmpshell += s_wave_f[ifunc++] * zdist2 * zdist * xdist; // zzzx
          tmpshell += s_wave_f[ifunc++] * zdist2 * zdist * ydist; // zzzy
          tmpshell += s_wave_f[ifunc++] * zdist2 * zdist2; // zzzz
          value += tmpshell * contracted_gto;
          break;
      } // end switch

      shell_counter++;
    }
  }

  orbitalgrid[outaddr] = value;
}



static int computepaddedsize(int orig, int tilesize) {
  int alignmask = tilesize - 1;
  int paddedsz = (orig + alignmask) & ~alignmask;  
//printf("orig: %d  padded: %d  tile: %d\n", orig, paddedsz, tilesize);
  return paddedsz;
}

static void * cudaorbitalthread(void *voidparms) {
  dim3 volsize, Gsz, Bsz;
  float *d_wave_f = NULL;
  float *d_basis_array = NULL;
  flint *d_atominfo = NULL;
  int *d_shellinfo = NULL;
  float *d_origin = NULL;
  int *d_numvoxels = NULL;
  float *d_orbitalgrid = NULL;
  float *h_orbitalgrid = NULL;
  float *h_basis_array_exp2f = NULL;
  int numvoxels[3];
  float origin[3];
  orbthrparms *parms = NULL;
  int usefastconstkernel=0;
  int threadid=0;
#if defined(USE_PINNED_MEMORY)
  int h_orbitalgrid_pinnedalloc=0;
#endif
  int tilesize = 1; // default tile size to use in absence of other info

#if defined(USE_CUDADEVPOOL)
  vmd_threadpool_worker_getid(voidparms, &threadid, NULL);
  vmd_threadpool_worker_getdata(voidparms, (void **) &parms);

  // scale tile size by device performance
  tilesize=4; // GTX 280, Tesla C1060 starting point tile size
  vmd_threadpool_worker_devscaletile(voidparms, &tilesize);
#else
  vmd_threadlaunch_getid(voidparms, &threadid, NULL);
  vmd_threadlaunch_getdata(voidparms, (void **) &parms);

  // attach to the GPU and check for errors
  cudaError_t rc;
  rc = cudaSetDevice(threadid);
  if (rc != cudaSuccess) { 
#if CUDART_VERSION >= 2010
    rc = cudaGetLastError(); // query last error and reset error state
    if (rc != cudaErrorSetOnActiveProcess)
      return NULL; // abort and return an error
#else
    cudaGetLastError(); // just ignore and reset error state, since older CUDA
                        // revs don't have a cudaErrorSetOnActiveProcess enum
#endif
  }
  // done attaching newly spawned thread to GPU..
#endif


  numvoxels[0] = parms->numvoxels[0];
  numvoxels[1] = parms->numvoxels[1];
  numvoxels[2] = 1;

  origin[0] = parms->origin[0];
  origin[1] = parms->origin[1];

  // setup energy grid size, padding out arrays for peak GPU memory performance
  volsize.x = (parms->numvoxels[0] + GPU_X_ALIGNMASK) & ~(GPU_X_ALIGNMASK);
  volsize.y = (parms->numvoxels[1] + GPU_Y_ALIGNMASK) & ~(GPU_Y_ALIGNMASK);
  volsize.z = 1;      // we only do one plane at a time
 
  // setup CUDA grid and block sizes
  Bsz.x = BLOCKSIZEX;
  Bsz.y = BLOCKSIZEY;
  Bsz.z = 1;
  Gsz.x = volsize.x / (Bsz.x * UNROLLX);
  Gsz.y = volsize.y / (Bsz.y * UNROLLY);
  Gsz.z = 1;
  int volmemsz = sizeof(float) * volsize.x * volsize.y * volsize.z;

  // determine which runtime strategy is workable
  // given the data sizes involved
  if ((parms->num_wave_f < MAX_WAVEF_SZ) &&
      (parms->numatoms < MAX_ATOM_SZ) &&
      (parms->numatoms < MAX_ATOMSHELL_SZ) &&
      (2*parms->num_basis < MAX_BASIS_SZ) &&
      (parms->num_shells < MAX_SHELL_SZ)) {
    usefastconstkernel=1;
  }

  // allow overrides for testing purposes
  if (getenv("VMDFORCEMOTILEDSHARED") != NULL) {
    usefastconstkernel=0; 
  }

  // allocate and copy input data to GPU arrays
  int padsz;
  padsz = computepaddedsize(2 * parms->num_basis, MEMCOALESCE);
  h_basis_array_exp2f = (float *) malloc(padsz * sizeof(float));
  float log2e = log2(2.718281828);
  for (int ll=0; ll<(2*parms->num_basis); ll+=2) {
#if 1
    // use exp2f() rather than expf()
    h_basis_array_exp2f[ll  ] = parms->basis_array[ll  ] * log2e;
#else
    h_basis_array_exp2f[ll  ] = parms->basis_array[ll  ];
#endif
    h_basis_array_exp2f[ll+1] = parms->basis_array[ll+1];
  }

  if (usefastconstkernel) {
    cudaMemcpyToSymbol(const_wave_f, parms->wave_f, parms->num_wave_f * sizeof(float), 0);
    cudaMemcpyToSymbol(const_atompos, parms->atompos, 3 * parms->numatoms * sizeof(float), 0);
    cudaMemcpyToSymbol(const_atom_basis, parms->atom_basis, parms->numatoms * sizeof(int), 0);
    cudaMemcpyToSymbol(const_num_shells_per_atom, parms->num_shells_per_atom, parms->numatoms * sizeof(int), 0);
    cudaMemcpyToSymbol(const_basis_array, h_basis_array_exp2f, 2 * parms->num_basis * sizeof(float), 0);
    cudaMemcpyToSymbol(const_num_prim_per_shell, parms->num_prim_per_shell, parms->num_shells * sizeof(int), 0);
    cudaMemcpyToSymbol(const_shell_symmetry, parms->shell_symmetry, parms->num_shells * sizeof(int), 0);
  } else {
    padsz = computepaddedsize(parms->num_wave_f, MEMCOALESCE);
    cudaMalloc((void**)&d_wave_f, padsz * sizeof(float));
    cudaMemcpy(d_wave_f, parms->wave_f, parms->num_wave_f * sizeof(float), cudaMemcpyHostToDevice);

    // pack atom data into a tiled array
    padsz = computepaddedsize(16 * parms->numatoms, MEMCOALESCE);
    flint * h_atominfo = (flint *) calloc(1, padsz * sizeof(flint));
    cudaMalloc((void**)&d_atominfo, padsz * sizeof(flint));
    for (int ll=0; ll<parms->numatoms; ll++) {
      int addr = ll * 16;
      h_atominfo[addr    ].f = parms->atompos[ll*3    ];
      h_atominfo[addr + 1].f = parms->atompos[ll*3 + 1];
      h_atominfo[addr + 2].f = parms->atompos[ll*3 + 2];
      h_atominfo[addr + 3].i = parms->atom_basis[ll];
      h_atominfo[addr + 4].i = parms->num_shells_per_atom[ll];
    }
    cudaMemcpy(d_atominfo, h_atominfo, padsz * sizeof(flint), cudaMemcpyHostToDevice);
    free(h_atominfo);

    padsz = computepaddedsize(16 * parms->num_shells, MEMCOALESCE);
    int * h_shellinfo = (int *) calloc(1, padsz * sizeof(int));
    cudaMalloc((void**)&d_shellinfo, padsz * sizeof(int));
    for (int ll=0; ll<parms->num_shells; ll++) {
      h_shellinfo[ll*16    ] = parms->num_prim_per_shell[ll];
      h_shellinfo[ll*16 + 1] = parms->shell_symmetry[ll];
    }
    cudaMemcpy(d_shellinfo, h_shellinfo, padsz * sizeof(int), cudaMemcpyHostToDevice);
    free(h_shellinfo);

    cudaMalloc((void**)&d_basis_array, padsz * sizeof(float));
    cudaMemcpy(d_basis_array, h_basis_array_exp2f, 2 * parms->num_basis * sizeof(float), cudaMemcpyHostToDevice);

    padsz = computepaddedsize(3, MEMCOALESCE);
    cudaMalloc((void**)&d_origin, padsz * sizeof(float));
    cudaMemcpy(d_origin, origin, 3 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_numvoxels, padsz * sizeof(int));
    cudaMemcpy(d_numvoxels, numvoxels, 3 * sizeof(int), cudaMemcpyHostToDevice);
  }

  // allocate and initialize the GPU output array
  cudaMalloc((void**)&d_orbitalgrid, volmemsz);

  CUERR // check and clear any existing errors

#if defined(USE_PINNED_MEMORY)
  // try to allocate working buffer in pinned host memory
  if ((getenv("VMDCUDANOPINNEDMEMORY") == NULL) && 
      (cudaMallocHost((void **) &h_orbitalgrid, volmemsz) == cudaSuccess)) {
    h_orbitalgrid_pinnedalloc=1;
  } else {
//    printf("WARNING: CUDA pinned memory allocation failed!\n"); 
    h_orbitalgrid_pinnedalloc=0;
    h_orbitalgrid = (float *) malloc(volmemsz);
  } 
#else
  // allocate working buffer
  h_orbitalgrid = (float *) malloc(volmemsz);
#endif

#if 0
  if (threadid == 0) {
    // inform on which kernel we're actually going to run
    printf("atoms[%d] ", parms->numatoms);
    printf("wavef[%d] ", parms->num_wave_f);
    printf("basis[%d] ", parms->num_basis);
    printf("shell[%d] ", parms->num_shells);
    if (usefastconstkernel) {
#if defined(VMDMOJITSRC) && defined(VMDMOJIT)
      printf("GPU constant memory (JIT)");
#else
      printf("GPU constant memory");
#endif
    } else {
      printf("GPU tiled shared memory:");
    }
    printf(" Gsz:%dx%d\n", Gsz.x, Gsz.y);
  }
#endif

  // loop over orbital planes
  vmd_tasktile_t tile;
  int planesize = numvoxels[0] * numvoxels[1];

#if defined(USE_CUDADEVPOOL)
  while (vmd_threadpool_next_tile(voidparms, tilesize, &tile) != VMD_SCHED_DONE) {
#else
  while (vmd_threadlaunch_next_tile(voidparms, tilesize, &tile) != VMD_SCHED_DONE) {
#endif
    int k;
    for (k=tile.start; k<tile.end; k++) {
      origin[2] = parms->origin[2] + parms->voxelsize * k;


      // RUN the kernel...
      if (usefastconstkernel) {
#if defined(VMDMOJITSRC) && defined(VMDMOJIT)
        cuorbitalconstmem_jit<<<Gsz, Bsz, 0>>>(parms->numatoms, 
                              parms->voxelsize,
                              origin[0],
                              origin[1],
                              origin[2],
                              d_orbitalgrid);
#else
        cuorbitalconstmem<<<Gsz, Bsz, 0>>>(parms->numatoms, 
                          parms->voxelsize,
                          origin[0],
                          origin[1],
                          origin[2],
                          d_orbitalgrid);
#endif
      } else {
        cuorbitaltiledshared<<<Gsz, Bsz, 0>>>(parms->numatoms, 
                             d_wave_f,
                             d_basis_array,
                             d_atominfo,
                             d_shellinfo,
                             parms->voxelsize,
                             origin[0],
                             origin[1],
                             origin[2],
                             d_orbitalgrid);
      }
      CUERR // check and clear any existing errors

      // Copy the GPU output data back to the host and use/store it..
      cudaMemcpy(h_orbitalgrid, d_orbitalgrid, volmemsz, cudaMemcpyDeviceToHost);
      CUERR // check and clear any existing errors

      // Copy GPU blocksize padded array back down to the original size
      int y;
      for (y=0; y<numvoxels[1]; y++) {
        long orbaddr = k*planesize + y*numvoxels[0];
        memcpy(&parms->orbitalgrid[orbaddr], &h_orbitalgrid[y*volsize.x], numvoxels[0] * sizeof(float));
      }
    }
    cudaThreadSynchronize();
  }

  free(h_basis_array_exp2f);

#if defined(USE_PINNED_MEMORY)
  if (h_orbitalgrid_pinnedalloc)
    cudaFreeHost(h_orbitalgrid);
  else
    free(h_orbitalgrid);
#else
  free(h_orbitalgrid);
#endif

  cudaFree(d_orbitalgrid);
  cudaFree(d_wave_f);
  cudaFree(d_basis_array);
  cudaFree(d_atominfo);
  cudaFree(d_shellinfo);
  cudaFree(d_numvoxels);
  cudaFree(d_origin);
  CUERR // check and clear any existing errors

  return NULL;
}


int vmd_cuda_evaluate_orbital_grid(vmd_threadpool_t *devpool,
                       int numatoms,
                       const float *wave_f, int num_wave_f,
                       const float *basis_array, int num_basis,
                       const float *atompos,
                       const int *atom_basis,
                       const int *num_shells_per_atom,
                       const int *num_prim_per_shell,
                       const int *shell_symmetry,
                       int num_shells,
                       const int *numvoxels,
                       float voxelsize,
                       const float *origin,
                       float *orbitalgrid) {
  int rc=0;
  orbthrparms parms;

  int deviceCount = 0;
  if (vmd_cuda_num_devices(&deviceCount))
    return -1;
  if (deviceCount == 0)
    return -1;

#if defined(VMDMOJITSRC)
  if (getenv("VMDMOJIT") != NULL) {
    cuda_jit_generate(getenv("VMDMOJITSRCFILE"), 
                      numatoms, wave_f, basis_array, atom_basis, 
                      num_shells_per_atom, num_prim_per_shell, shell_symmetry);
    return 0;
  }
#endif

#if defined(BUILD_STANDALONE)
  int numprocs = 1;
  if (getenv("VMDDUMPORBITALS")) {
    write_orbital_data(getenv("VMDDUMPORBITALS"), numatoms,
                       wave_f, num_wave_f, basis_array, num_basis,
                       atompos, atom_basis, num_shells_per_atom,
                       num_prim_per_shell, shell_symmetry,
                       num_shells, numvoxels, voxelsize, origin);

    read_calc_orbitals(getenv("VMDDUMPORBITALS"));
  }
#endif

  parms.numatoms = numatoms;
  parms.wave_f = wave_f;
  parms.num_wave_f = num_wave_f;
  parms.basis_array = basis_array;
  parms.num_basis = num_basis;
  parms.atompos = atompos;
  parms.atom_basis = atom_basis;
  parms.num_shells_per_atom = num_shells_per_atom;
  parms.num_prim_per_shell = num_prim_per_shell;
  parms.shell_symmetry = shell_symmetry;
  parms.num_shells = num_shells;
  parms.numvoxels = numvoxels;
  parms.voxelsize = voxelsize;
  parms.origin = origin;
  parms.orbitalgrid = orbitalgrid;

  vmd_tasktile_t tile;
  tile.start=0;
  tile.end=numvoxels[2];

#if defined(USE_CUDADEVPOOL)
  vmd_threadpool_sched_dynamic(devpool, &tile);
  vmd_threadpool_launch(devpool, cudaorbitalthread, &parms, 1); 
#else
  /* spawn child threads to do the work */
  rc = vmd_threadlaunch(1, &parms, cudaorbitalthread, &tile);
#endif

  return rc;
}


