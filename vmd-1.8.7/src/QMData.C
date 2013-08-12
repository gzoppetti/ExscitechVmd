/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2009 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: QMData.C,v $
 *	$Author: saam $	$Locker:  $		$State: Exp $
 *	$Revision: 1.79 $	$Date: 2009/07/28 22:54:49 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The QMData class, which stores all QM simulation data that
 * are not dependent on the timestep.
 *
 ***************************************************************************/

#include <math.h>
#include <stdio.h>
#include "QMData.h"
#include "QMTimestep.h"
#include "Orbital.h"
#include "Molecule.h"
#include "molfile_plugin.h"

// #define DEBUGGING 1

//! translation table for numerical runtype constants.
static const char *runtypes[] = { 
  "(unknown)", "ENERGY", "OPTIMIZE", "SADPOINT", "HESSIAN", 
  "SURFACE", "DYNAMICS", "PROPERTIES" };

//! translation table for numerical scftype constants.
static const char *scftypes[] = { 
  "(unknown)", "RHF", "UHF", "ROHF", "GVB", "MCSCF", "FF" };

///  constructor  
QMData::QMData(int natoms, int nbasis, int nshells, int nwave) :
  num_wave_f(nwave),
  num_basis(nbasis),
  num_atoms(natoms),
  num_shells(nshells)
{
  num_wavef_signa = 0;
  wavef_signa = NULL;
  num_shells_per_atom = NULL;
  num_prim_per_shell = NULL;
  wave_offset = NULL;
  atom_types = NULL;
  atom_basis = NULL;
  basis_array = NULL;
  basis_set = NULL;
  shell_symmetry = NULL;
  angular_momentum = NULL;
  norm_factors = NULL;
  carthessian = NULL;
  inthessian  = NULL;
  wavenumbers = NULL;
  intensities = NULL;
  normalmodes = NULL;
  imagmodes   = NULL;
};


QMData::~QMData() {
  int i;
  for (i=0; i<num_wavef_signa; i++) {
    free(wavef_signa[i].orbids);
    free(wavef_signa[i].orbocc);
  }
  free(wavef_signa);
  delete [] basis_array;
  delete [] shell_symmetry;
  delete [] num_shells_per_atom;
  delete [] num_prim_per_shell;
  delete [] atom_types;
  delete [] wave_offset;
  delete [] angular_momentum;
  delete [] carthessian;
  delete [] inthessian;
  delete [] wavenumbers;
  delete [] intensities;
  delete [] normalmodes;
  delete [] imagmodes;
  delete [] basis_string;
  delete [] atom_basis;
  if (norm_factors) {
    for (i=0; i<=highest_shell; i++) {
      if (norm_factors[i]) delete [] norm_factors[i];
    }
    delete [] norm_factors;
  }
  if (basis_set)
    delete_basis_set();
}

// Free memory of the basis set
void QMData::delete_basis_set() {
  int i, j;
  for (i=0; i<num_types; i++) {
    for (j=0; j<basis_set[i].numshells; j++) {
      delete [] basis_set[i].shell[j].prim;
    }
    delete [] basis_set[i].shell;
  }
  delete [] basis_set;
}



//! Set the total molecular charge, multiplicity and compute
//! the corresponding number of alpha/beta and total electrons.
//! XXX: this may be rather deduced from the occupations if available.
void QMData::init_electrons(Molecule *mol, int totcharge) {

  int i, nuclear_charge = 0;
  for (i=0; i<num_atoms; i++) {
    nuclear_charge += mol->atom(i)->atomicnumber;
  }
  
  totalcharge   = totcharge;
  num_electrons = nuclear_charge - totalcharge;
  //multiplicity  = mult;

#if 0
  if (scftype == SCFTYPE_RHF) {
    if (mult!=1) {
      msgErr << "For RHF calculations the multiplicity has to be 1, but it is "
             << multiplicity << "!"
             << sendmsg;
    }
    if (num_electrons%2) {
      msgErr << "Unpaired electron(s) in RHF calculation!"
             << sendmsg;
    }
    num_orbitals_A = num_orbitals_B = num_electrons/2;
  }
  else if ( (scftype == SCFTYPE_ROHF) ||
            (scftype == SCFTYPE_UHF) ) {
    num_orbitals_B = (num_electrons-multiplicity+1)/2;
    num_orbitals_A = num_electrons-num_orbitals_B;
  }
#endif
}



//   ====================================
//   Functions for basis set initializion
//   ====================================


// Populate basis set data and organize them into
// hierarcical data structures.
int QMData::init_basis(Molecule *mol, int num_basis_atoms,
                       const char *bstring,
                       const float *basis,
                       const int *atomic_numbers,
                       const int *nshells,
                       const int *nprims,
                       const int *symm) {
  num_types = num_basis_atoms;

  basis_string = new char[1+strlen(bstring)];
  strcpy(basis_string, bstring);

  if (!basis && (!strcmp(basis_string, "MNDO") ||
                 !strcmp(basis_string, "AM1")  ||
                 !strcmp(basis_string, "PM3"))) {
    // Semiempirical methods are based on STOs.
    // The only parameter we need for orbital rendering
    // are the exponents zeta for S, P, D,... shells for
    // each atom. Since most QM packages don't print these
    // values we have to generate the basis set here using
    // hardcoded table values.

    // generate_sto_basis(basis_string);

    return 1;
  }

  int i, j;


  // Copy the basis set arrays over.
  if (!basis || !num_basis) return 1;
  basis_array = new float[2*num_basis];
  memcpy(basis_array, basis, 2*num_basis*sizeof(float));

  if (!nshells || !num_basis_atoms) return 0;
  num_shells_per_atom = new int[num_basis_atoms];
  memcpy(num_shells_per_atom, nshells, num_basis_atoms*sizeof(int));

  if (!nprims || !num_shells) return 0;
  num_prim_per_shell = new int[num_shells];
  memcpy(num_prim_per_shell, nprims, num_shells*sizeof(int));

  if (!symm || !num_shells) return 0;
  shell_symmetry = new int[num_shells];
  highest_shell = 0;
  for (i=0; i<num_shells; i++) {
    int shelltype = symm[i];
    shell_symmetry[i] = shelltype;
    switch (shelltype) {
      case SP_S_SHELL:  shelltype = S_SHELL; break;
      case SP_P_SHELL:  shelltype = P_SHELL; break;
      case SPD_S_SHELL: shelltype = S_SHELL; break;
      case SPD_P_SHELL: shelltype = P_SHELL; break;
      case SPD_D_SHELL: shelltype = D_SHELL; break;
    }
    if (shelltype>highest_shell) highest_shell = shelltype;
  }
#ifdef DEBUGGING
  printf("highest shell = %d\n", highest_shell);
#endif

  // Create table of angular normalization constants
  init_angular_norm_factors();

  // Organize basis set data hierarchically
  int boffset = 0;
  int shell_counter = 0;
  int numcartpershell[14] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 93, 107}; 
  basis_set  = new basis_atom_t[num_basis_atoms];

  for (i=0; i<num_basis_atoms; i++) {
    basis_set[i].atomicnum = atomic_numbers[i];
    basis_set[i].numshells = num_shells_per_atom[i];
    basis_set[i].shell = new shell_t[basis_set[i].numshells];

    for (j=0; j<basis_set[i].numshells; j++) {
      // We keep the info about SP-shells in an extra flag in
      // the basis_set structure, while in shell_symmetry we
      // just store S or P.
      if      (shell_symmetry[shell_counter]==SP_S_SHELL) {
        shell_symmetry[shell_counter] = S_SHELL;
        basis_set[i].shell[j].sp_shell = 1;
      }
      else if (shell_symmetry[shell_counter]==SP_P_SHELL) {
        shell_symmetry[shell_counter] = P_SHELL;
        basis_set[i].shell[j].sp_shell = 1;
      }
      basis_set[i].shell[j].symmetry = shell_symmetry[shell_counter];

      int shelltype = shell_symmetry[shell_counter];
      basis_set[i].shell[j].num_cart_func = numcartpershell[shelltype];
      basis_set[i].shell[j].basis = basis_array+2*boffset;
      basis_set[i].shell[j].norm_fac = norm_factors[shelltype];
      basis_set[i].shell[j].numprims = num_prim_per_shell[shell_counter];

      basis_set[i].shell[j].prim = new prim_t[basis_set[i].shell[j].numprims];
#ifdef DEBUGGING
      //printf("atom %i shell %i %s\n", i, j, get_shell_type_str(&basis_set[i].shell[j]));
#endif

      int k;
      for (k=0; k<basis_set[i].shell[j].numprims; k++) {
        float expon = basis_array[2*(boffset+k)  ];
        float coeff = basis_array[2*(boffset+k)+1];
        basis_set[i].shell[j].prim[k].expon = expon;
        basis_set[i].shell[j].prim[k].coeff = coeff;
     }
  
      // Offsets to get to this shell in the basis array.
      boffset += basis_set[i].shell[j].numprims;

      shell_counter++;
    }
  }



  // Collapse basis set so that we have one basis set
  // per atom type.
  if (!create_unique_basis(mol, num_basis_atoms)) {
    return 0;
  }

  // Multiply the contraction coefficients with
  // the shell dependent part of the normalization factor.
  normalize_basis();

  return 1;
}


// =================================================
// Helper functions for building the list of unique
// basis set atoms
// =================================================

// Return 1 if the two given shell basis sets are identical,
// otherwise return 0.
static int compare_shells(const shell_t *s1, const shell_t *s2) {
  if (s1->symmetry != s2->symmetry) return 0;
  if (s1->numprims != s2->numprims) return 0;
  int i;
  for (i=0; i<s1->numprims; i++) {
    if (s1->prim[i].expon != s2->prim[i].expon) return 0;
    if (s1->prim[i].coeff != s2->prim[i].coeff) return 0;
  }
  return 1;
}

// Return 1 if the two given atomic basis sets are identical,
// otherwise return 0.
static int compare_atomic_basis(const basis_atom_t *a1, const basis_atom_t *a2) {
  if (a2->atomicnum != a1->atomicnum) return 0;
  if (a1->numshells != a2->numshells) return 0;
  int i;
  for (i=0; i<a1->numshells; i++) {
    if (!compare_shells(&a1->shell[i], &a2->shell[i])) return 0;
  }
  return 1;
}

static void copy_shell_basis(const shell_t *s1, shell_t *s2) {
  s2->numprims = s1->numprims;
  s2->symmetry = s1->symmetry;
  s2->sp_shell = s1->sp_shell;
  s2->norm_fac = s1->norm_fac;
  s2->num_cart_func = s1->num_cart_func;
  s2->prim = new prim_t[s2->numprims];
  int i;
  for (i=0; i<s2->numprims; i++) {
    s2->prim[i].expon = s1->prim[i].expon;
    s2->prim[i].coeff = s1->prim[i].coeff;
  }
}

static void copy_atomic_basis(const basis_atom_t *a1, basis_atom_t *a2) {
  a2->atomicnum = a1->atomicnum;
  a2->numshells = a1->numshells;
  a2->shell = new shell_t[a2->numshells];
  int i;
  for (i=0; i<a2->numshells; i++) {
    copy_shell_basis(&a1->shell[i], &a2->shell[i]);
  }
}

// Collapse basis set so that we have one basis set per
// atom type rather that per atom. In most cases an atom
// type is a chemical element. Create an array that maps
// individual atoms to their corresponding atomic basis.
int QMData::create_unique_basis(Molecule *mol, int num_basis_atoms) {
  basis_atom_t *unique_basis = new basis_atom_t[num_basis_atoms];
  copy_atomic_basis(&basis_set[0], &unique_basis[0]);
  int num_unique_atoms = 1;
  int i, j, k;
  for (i=1; i<num_basis_atoms; i++) {
    int found = 0;
    for (j=0; j<num_unique_atoms; j++) {
      if (compare_atomic_basis(&basis_set[i], &unique_basis[j])) {
        found = 1;
        break;
      }
    }
    if (!found) {
      copy_atomic_basis(&basis_set[i], &unique_basis[j]);
      num_unique_atoms++;
    }
  }

  msgInfo << "Number of unique atomic basis sets = "
          << num_unique_atoms <<"/"<< num_atoms << sendmsg;


  // Free memory of the basis set
  delete_basis_set();
  delete [] basis_array;

  num_types = num_unique_atoms;
  basis_set = unique_basis;

  // Count the new number of basis functions
  num_basis = 0;
  for (i=0; i<num_types; i++) {
    for (j=0; j<basis_set[i].numshells; j++) {
      num_basis += basis_set[i].shell[j].numprims;
    }
  }

  basis_array       = new float[2*num_basis];
  int *basis_offset = new int[num_types];

  int ishell = 0;
  int iprim  = 0;
  for (i=0; i<num_types; i++) {
     basis_offset[i] = iprim;

    for (j=0; j<basis_set[i].numshells; j++) {
      basis_set[i].shell[j].basis = basis_array+iprim;
#ifdef DEBUGGING
      printf("atom type %i shell %i %s\n", i, j, get_shell_type_str(&basis_set[i].shell[j]));
#endif
      for (k=0; k<basis_set[i].shell[j].numprims; k++) {
        basis_array[iprim  ] = basis_set[i].shell[j].prim[k].expon;
        basis_array[iprim+1] = basis_set[i].shell[j].prim[k].coeff;
#ifdef DEBUGGING 
        printf("prim %i: % 9.2f % 9.6f \n", k, basis_array[iprim], basis_array[iprim+1]);
#endif
        iprim += 2;
      }
      ishell++;
    }
  }

  atom_types = new int[num_atoms];

  // Assign basis set type to each atom and
  // create array of offsets into basis_array.
  for (i=0; i<num_atoms; i++) {
    int found = 0;
    for (j=0; j<num_types; j++) {
      if (basis_set[j].atomicnum == mol->atom(i)->atomicnumber) {
        found = 1;
        break;
      }
    }
    if (!found) {
      msgErr << "Error reading QM data: Could not assign basis set type to atom "
             << i << "." << sendmsg;
      delete [] basis_offset;
      return 0;
    }
    atom_types[i] = j;
#ifdef DEBUGGING 
    printf("atom_types[%d]=%d\n", i, j);
#endif
  }

  // Count the new number of shells
  num_shells = 0;
  for (i=0; i<num_atoms; i++) {
    num_shells += basis_set[atom_types[i]].numshells;
  }

  // Reallocate symmetry expanded arrays
  delete [] shell_symmetry;
  delete [] num_prim_per_shell;
  delete [] num_shells_per_atom;
  shell_symmetry      = new int[num_shells];
  num_prim_per_shell  = new int[num_shells];
  num_shells_per_atom = new int[num_atoms];
  atom_basis          = new int[num_atoms];
  wave_offset         = new int[num_atoms];
  int shell_counter = 0;
  int woffset = 0;

  // Populate the arrays again.
  for (i=0; i<num_atoms; i++) {
    int type = atom_types[i];

    // Offsets into wavefunction array
    wave_offset[i] = woffset;

    for (j=0; j<basis_set[type].numshells; j++) {
      shell_t *shell = &basis_set[type].shell[j];

      woffset += shell->num_cart_func;

      shell_symmetry[shell_counter]     = shell->symmetry;
      num_prim_per_shell[shell_counter] = shell->numprims;
      shell_counter++;
    }

    num_shells_per_atom[i] = basis_set[type].numshells;

    // Offsets into basis_array
    atom_basis[i] = basis_offset[type];
  }

  delete [] basis_offset;

  return 1;
}



// Multiply the contraction coefficients with
// the shell dependent part of the normalization factor.
void QMData::normalize_basis() {
  int i, j, k;
  for (i=0; i<num_types; i++) {
    for (j=0; j<basis_set[i].numshells; j++) {
      shell_t *shell = &basis_set[i].shell[j];
      int shelltype = shell->symmetry;
      for (k=0; k<shell->numprims; k++) {
        float expon = shell->prim[k].expon;
        float norm = pow(2.0*expon/VMD_PI, 0.75)
                        *sqrt(pow(8*expon, shelltype));
#ifdef DEBUGGING
        //printf("prim %i: % 9.2f % 9.6f  norm=%f\n", k, expon, coeff, norm);
#endif
        shell->basis[2*k+1] = norm*shell->prim[k].coeff;
      }
    }
  }
}

// Computes the factorial of n
static int fac(int n) {
  if (n==0) return 1;
  int i, x=1;
  for (i=1; i<=n; i++) x*=i;
  return x;
}

// Initialize table of angular momentum dependent normalization
// factors containing different factors for each shell and its
// cartesian functions.
void QMData::init_angular_norm_factors() {
  int shell;
  norm_factors = new float*[highest_shell+1];
  for (shell=0; shell<=highest_shell; shell++) {
    int i, j, k;
    int numcart = 0;
    for (i=0; i<=shell; i++) numcart += i+1;

    norm_factors[shell] = new float[numcart];
    int count = 0;
    for (k=0; k<=shell; k++) {
      for (j=0; j<=shell; j++) {
        for (i=0; i<=shell; i++) {
          if (i+j+k==shell) {
#ifdef DEBUGGING
            printf("count=%i (%i%i%i) %f\n", count, i, j, k, sqrt(((float)(fac(i)*fac(j)*fac(k))) / (fac(2*i)*fac(2*j)*fac(2*k))));
#endif
            norm_factors[shell][count++] = sqrt(((float)(fac(i)*fac(j)*fac(k))) / (fac(2*i)*fac(2*j)*fac(2*k)));
          }
        }
      }
    }
  } 
}



//   =================
//   Basis set acccess
//   =================


// Get basis set for an atom
const basis_atom_t* QMData::get_basis(int atom) const {
  if (!basis_set || !num_types || atom<0 || atom>=num_atoms)
    return NULL;
  return &(basis_set[atom_types[atom]]);
}


// Get basis set for a shell
const shell_t* QMData::get_basis(int atom, int shell) const {
  if (!basis_set || !num_types || atom<0 || atom>=num_atoms ||
      shell<0 || shell>=basis_set[atom_types[atom]].numshells)
    return NULL;
  return &(basis_set[atom_types[atom]].shell[shell]);
}


// Get the offset in the wavefunction array for a specified
// shell in an atom.
int QMData::get_wave_offset(int atom, int shell) const {
  if (atom<0 || atom>num_atoms) {
    msgErr << "Atom "<<atom<<" does not exist!"<<sendmsg;
    return -1;
  }
  if (shell<0 || shell>=basis_set[atom_types[atom]].numshells) {
    msgErr << "Shell "<<shell<<" in atom "<<atom
           << " does not exist!"<<sendmsg;
    return -1;
  }
  int i;
  int numcart = 0;
  for (i=0; i<shell; i++) {
    numcart += basis_set[atom_types[atom]].shell[i].num_cart_func;
  }
  return wave_offset[atom]+numcart;
}


/// Get shell type letter (S, P, D, F, ...) followed by '\0'
const char* QMData::get_shell_type_str(const shell_t *shell) {
  const char* map[14] = {"S\0", "P\0", "D\0", "F\0", "G\0", "H\0",
                  "I\0", "K\0", "L\0", "M\0", "N\0", "O\0", "Q\0", "R\0"};

  return map[shell->symmetry];
}



int QMData::set_angular_momenta(const int *angmom) {
  if (!angmom || !num_wave_f) return 0;
  angular_momentum = new int[3*num_wave_f];
  memcpy(angular_momentum, angmom, 3*num_wave_f*sizeof(int));
  return 1;
}

void QMData::set_angular_momentum(int atom, int shell, int mom,
                                  int *array) {
  if (!array || !angular_momentum) return;
  int offset = get_wave_offset(atom, shell);
  if (offset<0) return;
  memcpy(&angular_momentum[3*(offset+mom)], array, 3*sizeof(int));
}


// For a certain atom and shell return the exponent of the
// requested cartesian component of the angular momentum
// (specified by comp=0,1,2 for x,y,z resp.).
// Example:
// For XYYYZZ the exponents of the angular momentum are
// X (comp 0): 1
// Y (comp 1): 3
// Y (comp 2): 2
int QMData::get_angular_momentum(int atom, int shell, int mom, int comp) {
  if (!angular_momentum) return -1;
  int offset = get_wave_offset(atom, shell);
  if (offset<0 ||
      mom>=get_basis(atom, shell)->num_cart_func) return -1;
  //printf("atom=%d, shell=%d, mom=%d, comp=%d\n", atom, shell, mom, comp);
  return angular_momentum[3*(offset+mom)+comp];
}


// Set the angular momentum from a string
void QMData::set_angular_momentum_str(int atom, int shell, int mom,
                                  const char *tag) {
  unsigned int j;
  int offset = get_wave_offset(atom, shell);
  if (offset<0) return;

  int xexp=0, yexp=0, zexp=0;

  for (j=0; j<strlen(tag); j++) {
    switch (tag[j]) {
      case 'X':
        xexp++;
        break;
      case 'Y':
        yexp++;
        break;
      case 'Z':
        zexp++;
        break;
    }
  }
  angular_momentum[3*(offset+mom)  ] = xexp;
  angular_momentum[3*(offset+mom)+1] = yexp;
  angular_momentum[3*(offset+mom)+2] = zexp;
}


// Returns a pointer to a string representing the angular
// momentum of a certain cartesian basis function.
// The strings for an F-shell would be for instance
// XX YY ZZ XY XZ YZ.
// The necessary memory is automatically allocated.
// Caller is responsible delete the string!
char* QMData::get_angular_momentum_str(int atom, int shell, int mom) const {
  int offset = get_wave_offset(atom, shell);
  if (offset<0) return NULL;

  char *s = new char[2+basis_set[atom_types[atom]].shell[shell].symmetry];
  int i, j=0;
  for (i=0; i<angular_momentum[3*(offset+mom)  ]; i++) s[j++]='X';
  for (i=0; i<angular_momentum[3*(offset+mom)+1]; i++) s[j++]='Y';
  for (i=0; i<angular_momentum[3*(offset+mom)+2]; i++) s[j++]='Z';
  s[j] = '\0';
  if (!strlen(s)) strcpy(s, "S");

  return s;
}



//   ========================
//   Hessian and normal modes
//   ========================


void QMData::set_carthessian(int numcart, double *array) {
  if (!array || !numcart || numcart!=3*num_atoms) return;
  carthessian = new double[numcart*numcart];
  memcpy(carthessian, array, numcart*numcart*sizeof(double));
}

void QMData::set_inthessian(int numint, double *array) {
  if (!array || !numint) return;
  nintcoords = numint;
  inthessian = new double[numint*numint];
  memcpy(inthessian, array, numint*numint*sizeof(double));
}

void QMData::set_normalmodes(int numcart, float *array) {
  if (!array || !numcart || numcart!=3*num_atoms) return;
  normalmodes = new float[numcart*numcart];
  memcpy(normalmodes, array, numcart*numcart*sizeof(float));
}

void QMData::set_wavenumbers(int numcart, float *array) {
  if (!array || !numcart || numcart!=3*num_atoms) return;
  wavenumbers = new float[numcart];
  memcpy(wavenumbers, array, numcart*sizeof(float));
}

void QMData::set_intensities(int numcart, float *array) {
  if (!array || !numcart || numcart!=3*num_atoms) return;
  intensities = new float[numcart];
  memcpy(intensities, array, numcart*sizeof(float));
}

void QMData::set_imagmodes(int numimag, int *array) {
  if (!array || !numimag) return;
  nimag = numimag;
  imagmodes = new int[nimag];
  memcpy(imagmodes, array, nimag*sizeof(int));
}



//   =====================
//   Calculation meta data
//   =====================


// Translate the runtype constant into a string
const char *QMData::get_runtype_string(void) const
{
  // runtype is public, so we have to check its consistency.
  if (runtype < 0 || runtype >= RUNTYPE_TOTAL )
    return "";

  return runtypes[runtype];
}


// Translate the scftype constant into a string
const char *QMData::get_scftype_string(void) const
{
  // scftype is public, so we have to check its consistency.
  if (scftype < 0 || scftype >= SCFTYPE_TOTAL )
    return "";
      
  return scftypes[scftype];
}


// Get status of SCF and optimization convergence
const char* QMData::get_status_string() {
  if (status==MOLFILE_QM_OPT_CONVERGED)
    return "Optimization converged";
  else if (status==MOLFILE_QM_OPT_NOT_CONV)
    return "Optimization not converged";
  else if (status==MOLFILE_QM_SCF_NOT_CONV)
    return "SCF not converged";
  else if (status==MOLFILE_QM_FILE_TRUNCATED)
    return "File truncated";
  else
    return "Unknown";
}



//   =======================
//   Wavefunction signatures
//   =======================


/// Determine a unique ID for each wavefuntion based on it's signature
/// (type, spin, excitation, info)
/// If for a given timestep there are more than one wavefunctions
/// with the same signature the we assume these are different and
/// we assign a different IDs. This can happen if the wavefunctions
/// cannot be sufficiently distinguished by the existing descriptors
/// and the plugin didn't make use of the info string to set them
/// apart.
int QMData::assign_wavef_id(int type, int spin, int exci, char *info,
                            wavef_signa_t *(&signa_ts), int &numsig) {
  int j, idtag=-1;

  for (j=0; j<num_wavef_signa; j++) {
    if (wavef_signa[j].type==type &&
        wavef_signa[j].spin==spin &&
        wavef_signa[j].exci==exci &&
        (info && !strncmp(wavef_signa[j].info, info, QMDATA_BUFSIZ))) {
      idtag = j;
    }
  }
  // Check if we have the same signature in the current timestep
  int duplicate = 0;
  for (j=0; j<numsig; j++) {
    if (signa_ts[j].type==type &&
        signa_ts[j].spin==spin &&
        signa_ts[j].exci==exci &&
        (info && !strncmp(signa_ts[j].info, info, QMDATA_BUFSIZ))) {
      duplicate = 1;
    }
  }
  
  // Add a new signature for the current timestep
  if (!signa_ts) {
    signa_ts = (wavef_signa_t *)calloc(1, sizeof(wavef_signa_t));
  } else {
    signa_ts = (wavef_signa_t *)realloc(signa_ts,
                                        (numsig+1)*sizeof(wavef_signa_t));
  }
  signa_ts[numsig].type = type;
  signa_ts[numsig].spin = spin;
  signa_ts[numsig].exci = exci;
  if (!info)
    signa_ts[numsig].info[0] = '\0';
  else
    strncpy(signa_ts[numsig].info, info, QMDATA_BUFSIZ);
  numsig++;

  // Add new wavefunction ID tag in case this signature wasn't
  // found at all or we have a duplicate for this timestep
  if (idtag<0 || duplicate) {
    if (!wavef_signa) {
      wavef_signa = (wavef_signa_t *)calloc(1, sizeof(wavef_signa_t));
    } else {
      wavef_signa = (wavef_signa_t *)realloc(wavef_signa,
                              (num_wavef_signa+1)*sizeof(wavef_signa_t));
    }
    wavef_signa[num_wavef_signa].type = type;
    wavef_signa[num_wavef_signa].spin = spin;
    wavef_signa[num_wavef_signa].exci = exci;
    wavef_signa[num_wavef_signa].max_avail_orbs = 0;
    wavef_signa[num_wavef_signa].orbids = NULL;
    if (!info)
      wavef_signa[num_wavef_signa].info[0] = '\0';
    else
      strncpy(wavef_signa[num_wavef_signa].info, info, QMDATA_BUFSIZ);
    idtag = num_wavef_signa;
    num_wavef_signa++;
  }

  //printf("idtag=%d (%d, %d, %d, %s)\n", idtag, type, spin, exci, info);

  return idtag;  
}


// Find the wavefunction ID tag by comparing
// type, spin, and excitation with the signatures
// of existing wavefunctions
// Returns -1 if no such wavefunction exists.
int QMData::find_wavef_id_from_gui_specs(int type, int spin, int exci) {
  int i, idtag = -1;
  for (i=0; i<num_wavef_signa; i++) {
    if (spin==wavef_signa[i].spin &&
        exci==wavef_signa[i].exci) {
      if ((type==GUI_WAVEF_TYPE_CANON &&
           wavef_signa[i].type==MOLFILE_WAVE_CANON)    ||
          (type==GUI_WAVEF_TYPE_GEMINAL &&
           wavef_signa[i].type==MOLFILE_WAVE_GEMINAL)  ||
          (type==GUI_WAVEF_TYPE_MCSCFNAT &&
           wavef_signa[i].type==MOLFILE_WAVE_MCSCFNAT) ||
          (type==GUI_WAVEF_TYPE_MCSCFOPT &&
           wavef_signa[i].type==MOLFILE_WAVE_MCSCFOPT) ||
          (type==GUI_WAVEF_TYPE_CINAT &&
           wavef_signa[i].type==MOLFILE_WAVE_CINATUR)  ||
          (type==GUI_WAVEF_TYPE_LOCAL &&
           (wavef_signa[i].type==MOLFILE_WAVE_BOYS ||
            wavef_signa[i].type==MOLFILE_WAVE_RUEDEN ||
            wavef_signa[i].type==MOLFILE_WAVE_PIPEK))  ||
          (type==GUI_WAVEF_TYPE_OTHER &&
           wavef_signa[i].type==MOLFILE_WAVE_UNKNOWN)) {
        idtag = i;
      }
    }
  }
  return idtag;
}


/// Return 1 if we have any wavefunction with the given type
int QMData::has_wavef_type(int type) {
  int i;
  for (i=0; i<num_wavef_signa; i++) {
    if (wavef_signa[i].type==type) return 1;
  }
  return 0;
}


/// Return 1 if we have any wavefunction with the given spin
int QMData::has_wavef_spin(int spin) {
  int i;
  for (i=0; i<num_wavef_signa; i++) {
    if (wavef_signa[i].spin==spin) return 1;
  }
  return 0;
}


/// Return 1 if we have any wavefunction with the given spin
int QMData::has_wavef_exci(int exci) {
  int i;
  for (i=0; i<num_wavef_signa; i++) {
    if (wavef_signa[i].exci==exci) return 1;
  }
  return 0;
}


/// Return 1 if we have any wavefunction with the given
/// signature (type, spin, and excitation).
int QMData::has_wavef_signa(int type, int spin, int exci) {
  int i;
  for (i=0; i<num_wavef_signa; i++) {
    if (wavef_signa[i].type==type &&
        wavef_signa[i].exci==exci &&
        wavef_signa[i].spin==spin) return 1;
  }
  return 0;
}


/// Get the highest excitation for any wavefunction 
/// with the given type.
int QMData::get_highest_excitation(int type) {
  int i, highest=0;
  for (i=0; i<num_wavef_signa; i++) {
    if (wavef_signa[i].type==type &&
        wavef_signa[i].exci>highest)
      highest = wavef_signa[i].exci;
  }
  return highest;
}


//   =====================================================
//   Functions dealing with the list of orbitals available 
//   for a given wavefunction. Needed for the GUI.
//   =====================================================


// Merge the provided list of orbital IDs with the existing
// list of available orbitals. Available orbitals are the union
// of all orbital IDs for the wavefunction with ID iwavesig
// occuring throughout the trajectory.
void QMData::update_avail_orbs(int iwavesig, int norbitals,
                               const int *orbids, const float *orbocc) {
  int i, j;

  // Signature of wavefunction
  wavef_signa_t *cursig = &wavef_signa[iwavesig];

  for (i=0; i<norbitals; i++) {
    int found = 0;
    for (j=0; j<cursig->max_avail_orbs; j++) {
      if (cursig->orbids[j]==orbids[i]) {
        found = 1;
        break;
      }
    }
    if (!found) {
      if (!cursig->orbids) {
        cursig->orbids = (int  *)calloc(1, sizeof(int));
        cursig->orbocc = (float*)calloc(1, sizeof(float));
      } else {
        cursig->orbids = (int  *)realloc(cursig->orbids,
                                  (cursig->max_avail_orbs+1)*sizeof(int));
        cursig->orbocc = (float*)realloc(cursig->orbocc,
                                  (cursig->max_avail_orbs+1)*sizeof(float));
      }
      cursig->orbids[cursig->max_avail_orbs] = orbids[i];
      cursig->orbocc[cursig->max_avail_orbs] = orbocc[i];
      cursig->max_avail_orbs++;
    }
  }
//   printf("iwavesig=%d, ", iwavesig);
//   for (j=0; j<cursig->max_avail_orbs; j++) {
//     printf("%d %.2f\n",cursig->orbids[j], cursig->orbocc[j]);
//   }
//   printf("\n");
}


/// Return the maximum number of available orbitals
/// for the given wavefunction over all frames
/// Can be used to determine the number of orbitals
/// to be displayed in the GUI.
/// Returns -1 if requested wavefunction doesn't exist.
int QMData::get_max_avail_orbitals(int iwavesig) {
  if (iwavesig<0 || iwavesig>=num_wavef_signa) return -1;
  return wavef_signa[iwavesig].max_avail_orbs;
}


/// Get IDs of all orbitals available for the given wavefunction.
/// iwavesig is the index of the wavefunction signature.
/// Returns 1 upon success, 0 otherwise.
int QMData::get_avail_orbitals(int iwavesig, int *(&orbids)) {
  if (iwavesig<0 || iwavesig>=num_wavef_signa) return 0;

  int i;
  for (i=0; i<wavef_signa[iwavesig].max_avail_orbs; i++) {
    orbids[i] = wavef_signa[iwavesig].orbids[i];
  }
  return 1;
}


/// Get occupancies of all orbitals available for the given wavefunction.
/// iwavesig is the index of the wavefunction signature.
/// Returns 1 upon success, 0 otherwise.
int QMData::get_avail_occupancies(int iwavesig, float *(&orbocc)) {
  if (iwavesig<0 || iwavesig>=num_wavef_signa) return 0;

  int i;
  for (i=0; i<wavef_signa[iwavesig].max_avail_orbs; i++) {
    orbocc[i] = wavef_signa[iwavesig].orbocc[i];
  }
  return 1;
}


/// For the given wavefunction signature return
/// the iorb-th orbital ID. Used to translate from the
/// GUI list of available orbitals the unique orbital label.
/// Returns -1 if requested wavefunction doesn't exist or
/// the orbital index is out of range.
int QMData::get_orbital_label_from_gui_index(int iwavesig, int iorb) {
  if (iwavesig<0 || iwavesig>=num_wavef_signa ||
      iorb<0 ||iorb>=wavef_signa[iwavesig].max_avail_orbs)
    return -1;
  return wavef_signa[iwavesig].orbids[iorb];
}

/// Return 1 if the given wavefunction has an orbital with
/// ID orbid in any frame.
int QMData::has_orbital(int iwavesig, int orbid) {
  if (iwavesig<0 || iwavesig>=num_wavef_signa) return 0;

  int i;
  for (i=0; i<wavef_signa[iwavesig].max_avail_orbs; i++) {
    if (orbid==wavef_signa[iwavesig].orbids[i]) return 1;
  }
  return 0;

}


// Create a new Orbital and return the pointer.
// User is responsible for deleting.
Orbital* QMData::create_orbital(int iwave, int orbid, float *pos,
                         QMTimestep *qmts) {
  Orbital *orbital = new Orbital(pos,
                  qmts->get_wavecoeffs(iwave),
                  basis_array, basis_set, atom_types,
                  atom_sort, atom_basis,
                  (const float**)norm_factors,
                  num_shells_per_atom,
                  num_prim_per_shell, shell_symmetry,
                  num_atoms, num_types, num_wave_f, num_basis,
                  orbid);
  return orbital;
}



// =========================================
// Currently unused stuff I might need later
// =========================================


#if 0                           // XXX: unused
// XXX these quicksort routines are duplicates of the ones
// in QMTimestep.
static void quicksort(const int *tag, int *A, int p, int r);
static int  quickpart(const int *tag, int *A, int p, int r);

// Create an index array *atom_sort that sorts the atoms
// by basis set type (usually that means by atomic number).
void QMData::sort_atoms_by_type() {
  int i;
  if (atom_sort) delete [] atom_sort;

  // Initialize the index array;
  atom_sort = new int[num_atoms];
  for (i=0; i<num_atoms; i++) {
    atom_sort[i] = i;
  }

  // Sort index array according to the atom_types
  quicksort(atom_types, atom_sort, 0, num_atoms-1);

  //int *sorted_types = new int[num_atoms];

  // Copy data over into sorted arrays
  //for (i=0; i<num_atoms; i++) {
  //  sorted_types[i] = atom_types[atom_sort[i]];
  //}
}

// The standard quicksort algorithm except for it doesn't
// sort the data itself but rather sorts array of ints *A
// in the same order as it would sort the integers in array
// *tag. Array *A can then be used to reorder any array
// according to the string tags.
// Example:
// tag:   BC DD BB AA  -->  AA BB BC DD
// index:  0  1  2  3  -->   3  2  0  1
//
static void quicksort(const int* tag, int *idx, int p, int r) {
  int q;
  if (p < r) {
    q=quickpart(tag, idx, p, r);
    quicksort(tag, idx, p, q);
    quicksort(tag, idx, q+1, r);
  }
}

// Partitioning for quicksort.
static int quickpart(const int *tag, int *idx, int p, int r) {
  int i, j;
  int tmp;
  int x = tag[idx[p]];
  i = p-1;
  j = r+1;

  while (1) {
    // Find highest element smaller than idx[p]
    do j--; while (tag[idx[j]] > x);

    // Find lowest element larger than idx[p]
    do i++; while (tag[idx[i]] < x);

    if (i < j) {
      tmp    = idx[i];
      idx[i] = idx[j];
      idx[j] = tmp;
    }
    else {
      return j;
    }
  }
}
#endif
