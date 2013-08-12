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
 *	$RCSfile: QMTimestep.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.56 $	$Date: 2009/08/01 04:10:30 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The QMTimestep class, which stores orbitals, SCF energies, etc. for a
 * single timestep.
 *
 ***************************************************************************/

#include <math.h>
#include <stdio.h>
#include "QMTimestep.h"
#include "QMData.h"
#include "Molecule.h"
#include "molfile_plugin.h"

#define ANGMOM_X  0
#define ANGMOM_Y  1
#define ANGMOM_Z  2

//#define DEBUG 1

Wavefunction::Wavefunction()
{
  idtag      = 0;
  type       = 0;
  spin       = 0;
  excitation = 0;
  multiplicity = 0;
  num_orbitals = 0;
  num_coeffs = 0;
  energy = 0.0;
  wave_coeffs  = NULL;
  orb_energies = NULL;
  occupancies  = NULL;
  orb_ids      = NULL;
  orb_id2index = NULL;
}

Wavefunction::Wavefunction(const Wavefunction& wf)
{
  wave_coeffs  = NULL;
  orb_energies = NULL;
  occupancies  = NULL;
  orb_ids      = NULL;
  orb_id2index = NULL;
  *this = wf;
}

Wavefunction::Wavefunction(int ncoeffs,
                           int norbitals, 
                           const float *coeffs,
                           const float *energies,
                           const float *occ,
                           const int   *orbids,
                           double _energy,
                           int _idtag,
                           int _type,
                           int _spin,
                           int _excitation,
                           int _multiplicity,
                           char *infostr) :
  idtag     (_idtag),
  type      (_type),
  excitation(_excitation),
  multiplicity(_multiplicity),
  num_orbitals(norbitals),
  num_coeffs  (ncoeffs),
  energy      (_energy),
  wave_coeffs (NULL),
  orb_energies(NULL),
  occupancies (NULL),
  orb_ids     (NULL),
  orb_id2index(NULL)
{
  strncpy(info, infostr, QMDATA_BUFSIZ);

  set_coeffs(coeffs, norbitals, ncoeffs);
  set_orbenergies(energies, norbitals);
  set_occupancies(occ, norbitals);
  set_orbids(orbids, norbitals);
}

Wavefunction& Wavefunction::operator=(const Wavefunction& wf) {
  idtag        = wf.idtag;
  type         = wf.type;
  spin         = wf.spin;
  excitation   = wf.excitation;
  multiplicity = wf.multiplicity;
  strncpy(info, wf.info, QMDATA_BUFSIZ);

  num_orbitals = wf.num_orbitals;
  num_coeffs   = wf.num_coeffs;
  energy       = wf.energy;

  if (orb_energies) delete [] orb_energies;
  if (orb_ids)      delete [] orb_ids;
  if (orb_id2index) delete [] orb_id2index;
  if (wave_coeffs)  delete [] wave_coeffs;
  if (occupancies)  delete [] occupancies;
  wave_coeffs  = NULL;
  orb_energies = NULL;
  occupancies  = NULL;
  orb_ids      = NULL;
  orb_id2index = NULL;

  if (wf.orb_energies) {
    orb_energies = new float[num_orbitals];
    memcpy(orb_energies, wf.orb_energies, num_orbitals * sizeof(float));
  }

  if (wf.wave_coeffs) {
    wave_coeffs = new float[num_orbitals * num_coeffs];
    memcpy(wave_coeffs, wf.wave_coeffs, num_orbitals * num_coeffs * sizeof(float));
  }

  if (wf.occupancies) {
    occupancies = new float[num_orbitals];
    memcpy(occupancies, wf.occupancies, num_orbitals * sizeof(int));
  }

  if (wf.orb_ids) {
    orb_ids = new int[num_orbitals];
    memcpy(orb_ids, wf.orb_ids, num_orbitals * sizeof(int));
  }

  if (wf.orb_id2index) {
    orb_id2index = new int[num_coeffs];
    memcpy(orb_id2index, wf.orb_id2index, num_coeffs * sizeof(int));
  }

  return *this;
}

// Move the data over from the given wavefunction wf
// and set the pointers in wf to NULL.
// This avoids copying the arrays.
void Wavefunction::movefrom(Wavefunction& wf) {
  idtag        = wf.idtag;
  type         = wf.type;
  spin         = wf.spin;
  excitation   = wf.excitation;
  multiplicity = wf.multiplicity;
  strncpy(info, wf.info, QMDATA_BUFSIZ);

  num_orbitals = wf.num_orbitals;
  num_coeffs   = wf.num_coeffs;
  energy       = wf.energy;

  wave_coeffs  = wf.wave_coeffs;
  orb_energies = wf.orb_energies;
  occupancies  = wf.occupancies;
  orb_ids      = wf.orb_ids;
  orb_id2index = wf.orb_id2index;
  wf.wave_coeffs  = NULL;
  wf.orb_energies = NULL;
  wf.occupancies  = NULL;
  wf.orb_ids      = NULL;
  wf.orb_id2index = NULL;
}

Wavefunction::~Wavefunction()
{
  if (wave_coeffs)  delete [] wave_coeffs;
  if (orb_energies) delete [] orb_energies;
  if (occupancies)  delete [] occupancies;
  if (orb_ids)      delete [] orb_ids;
  if (orb_id2index) delete [] orb_id2index;
}

#if 0
const float* Wavefunction::get_coeffs(int orb)
{
  if (!wave_coeffs || orb<0 || orb>=num_orbitals) return NULL;
  return wave_coeffs + orb*num_coeffs;
}

float Wavefunction::get_coeff(int orb, int i)
{
  if (orb>=num_orbitals || i<0 || i>=num_coeffs || !wave_coeffs)
    return 0.f;

  return wave_coeffs[orb*num_coeffs + i];
}
#endif

float Wavefunction::get_orbitalenergy(int orb)
{
  if (orb_energies && orb>=0 && orb<num_orbitals)
    return orb_energies[orb];
  else
    return 0.f;
}


void Wavefunction::set_coeffs(const float *wfn, int norbitals, int wavef_size) 
{
  if (!wfn || !norbitals || !wavef_size) return;
  num_orbitals = norbitals;
  num_coeffs = wavef_size;
  
  wave_coeffs = new float[num_orbitals*num_coeffs];
  memcpy(wave_coeffs, wfn, num_orbitals*num_coeffs*sizeof(float));
}


void Wavefunction::set_orbenergies(const float *energies, int norbitals) 
{
  if (!energies || !norbitals) return;

  if (num_orbitals < 1) 
    num_orbitals = norbitals;

  if (num_orbitals != norbitals) 
    msgWarn << "Mismatched number of orbitals in " << "QMTimestep::set_orbenergies()" << ": " 
            << norbitals << " != " << num_orbitals << sendmsg;
  
  orb_energies = new float[norbitals];
  memcpy(orb_energies, energies, norbitals*sizeof(float));
}

void Wavefunction::set_occupancies(const float *occ, int norbitals) 
{
  if (!occ || !norbitals) return;

  if (num_orbitals < 1) 
    num_orbitals = norbitals;

  if (num_orbitals != norbitals) 
    msgWarn << "Mismatched number of orbitals in " << "QMTimestep::set_occupancies()" << ": " 
            << norbitals << " != " << num_orbitals << sendmsg;
  
  occupancies = new float[norbitals];
  memcpy(occupancies, occ, norbitals*sizeof(float));
}

// Set orbital ID number array.
// Assumed 1,2,3,...num_orbitals if orbids==NULL.      
void Wavefunction::set_orbids(const int *orbids, int norbitals) 
{
  if (!norbitals) return;

  if (num_orbitals < 1) 
    num_orbitals = norbitals;

  if (num_orbitals != norbitals) 
    msgWarn << "Mismatched number of orbitals in " << "QMTimestep::set_orbids()" << ": " 
            << norbitals << " != " << num_orbitals << sendmsg;
  
  int i;
  orb_ids = new int[norbitals];
  if (orbids) {
    memcpy(orb_ids, orbids, norbitals*sizeof(int));
  } else {
    for (i=0; i<num_orbitals; i++) {
      orb_ids[i] = i+1;
    }
  }

  orb_id2index = new int[num_coeffs+1];
  for (i=0; i<num_coeffs+1; i++) {
    orb_id2index[i] = -1;
  }
  for (i=0; i<num_orbitals; i++) {
    orb_id2index[orb_ids[i]] = i;
//    printf("orb_id2index[%d]=%d\n", orb_ids[i], orb_id2index[orb_ids[i]]);
  }
}


void Wavefunction::get_typestr(char *&typestr) {
  typestr = new char[64];
  switch (type) {
  case MOLFILE_WAVE_CANON:
    strcpy(typestr, "Canonical");
    break;
  case MOLFILE_WAVE_CINATUR:
    strcpy(typestr, "CI natural");
    break;
  case MOLFILE_WAVE_GEMINAL:
    strcpy(typestr, "GVB geminal pairs");
    break;
  case MOLFILE_WAVE_BOYS:
    strcpy(typestr, "Boys localized");
    break;
  case MOLFILE_WAVE_RUEDEN:
    strcpy(typestr, "Ruedenberg localized");
    break;
  case MOLFILE_WAVE_PIPEK:
    strcpy(typestr, "Pipek-Mezey localized");
    break;
  case MOLFILE_WAVE_MCSCFOPT:
    strcpy(typestr, "MCSCF optimized");
    break;
  case MOLFILE_WAVE_MCSCFNAT:
    strcpy(typestr, "MCSCF natural");
    break;
  default:
    strcpy(typestr, "Unknown");
  }
}


// The array angular_momentum consists of 3*num_wave_f flags
// determining which cartesian component of the angular
// momentum each wavefunction coefficient corresponds to. 
// Each triplet represents the exponents of the x, y, and z
// components. I.e. (1, 0, 2) means xzz for an F shell.
// Our inner loop in the orbital computation assumes an order
// with the z-component changing fastest and x slowest, i.e
// for a D- shell the order would be xx xy xz yy yz zz
// (I hope I did that right :-)
// This is a routine to sort the wavefunction coefficients
// of each shell according to that scheme.
void Wavefunction::sort_wave_coefficients(QMData *qmdata) {
  if (!wave_coeffs || !num_orbitals ||
      !qmdata->num_wave_f || !qmdata->num_basis) {
    return;
  }

  int atom, j;
  for (atom=0; atom<qmdata->get_num_atoms(); atom++) {
    for (j=0; j<qmdata->get_num_shells_per_atom()[atom]; j++) {

      const shell_t *shell = qmdata->get_basis(atom, j);
      if (!shell) {
        printf("sort_wave_coefficients(): NO SHELL %d %d\n", atom, j);
      }
      int shelltype = shell->symmetry;

      // Sort the wavefunction coefficients of this shell
      // according to the angular momentum.
      // First we must sort by increasing exponent of the
      // z-component:
      sort_incr(qmdata, atom, j, ANGMOM_Z, 0, shell->num_cart_func);

      // Then we sort the coefficients for each z-component
      // by increasing exponent of the y-component.
      // The x-component needs no sorting since it is
      // dependent on y and z.
      int k, first=0;
      for (k=0; k<=shelltype; k++) {
        sort_incr(qmdata, atom, j, ANGMOM_Y, first, shelltype-k+1);
        first += shelltype-k+1;
      }
    }
  }
}



// Sort array of indexes *idx according to the order obtained 
// by sorting the integers in array *tag. Array *idx can then
// be used to reorder any array according to the string tags.
static void quicksort(const int *tag, int *A, int p, int r);
static int  quickpart(const int *tag, int *A, int p, int r);


// Sort the wavefunction coefficients of the specified shell
// according to the increasing exponent of requested angular
// momentum component.
// Parameters:
// comp:  0,1,2; sort according to the x,y,z component 
// first: the array element where the sorting shall begin
//        counted from the beginning of the shell
// num:   the number of consecutive array elements to be
//        sorted
void Wavefunction::sort_incr(QMData *qmdata, int atom, int ishell,
                           int comp, int first, int num) {
  int i, orb;
  int wave_offset = qmdata->get_wave_offset(atom, ishell);

  // Initialize the index array;
  int *index = new int[num];
  int *powz  = new int[num];
  for (i=0; i<num; i++) {
    index[i] = i;
    powz[i] = qmdata->get_angular_momentum(atom, ishell, first+i, comp);
  }

  float *wave_f = wave_coeffs + wave_offset + first;
#if DEBUG
  for (i=0; i<num; i++) {
    printf("unsorted %i: %i %f\n", i, powz[i], wave_f[i]);
  }
#endif

  // Sort index array according to power of z
  quicksort(powz, index, 0, num-1);

  // Copy angular moments over into sorted array
  int   *sorted_ang    = new int[3*num];
  for (i=0; i<num; i++) {
    sorted_ang[3*i+(comp+1)%3] = qmdata->get_angular_momentum(atom, ishell, first+index[i], (comp+1)%3);
    sorted_ang[3*i+(comp+2)%3] = qmdata->get_angular_momentum(atom, ishell, first+index[i], (comp+2)%3);
    sorted_ang[3*i+comp] = powz[index[i]];
  }

  // Copy sorted angular moments back into original arrays
  for (i=0; i<num; i++) {
    qmdata->set_angular_momentum(atom, ishell, first+i, &sorted_ang[3*i]);
  }

  float *sorted_wave_f = new float[num];

  // Sort the wavefunction coefficients for each orbital
  for (orb=0; orb<num_orbitals; orb++) {
    wave_f = wave_coeffs + (num_coeffs*orb) + wave_offset + first;

    // Create sorted array
    for (i=0; i<num; i++) {
      sorted_wave_f[i] = wave_f[index[i]];
    }

    //Copy sorted coeffs back to original array
    for (i=0; i<num; i++) {
      wave_f[i] = sorted_wave_f[i];
    }
  }

#if DEBUG
  orb=0;
  for (i=0; i<qmdata->get_basis(atom, ishell)->num_cart_func; i++) {
    printf("sorted %i: %i %i %i %f\n", i, 
           qmdata->get_angular_momentum(atom, ishell, i, 0),
           qmdata->get_angular_momentum(atom, ishell, i, 1),
           qmdata->get_angular_momentum(atom, ishell, i, 2),
           wave_coeffs[(num_coeffs*orb) + wave_offset + i]);
  }
#endif

  delete [] powz;
  delete [] sorted_wave_f;
  delete [] sorted_ang;
  delete [] index;
}


// **************************************************
// ************** QMTimestep class ******************
// **************************************************

///  constructor  
QMTimestep::QMTimestep(int numatoms) :
  num_scfiter(0),
  num_atoms  (numatoms),
  num_wavef  (0),
  num_idtags (0),
  num_charge_sets(0)
{
  wavef_id_map     = NULL;
  wavef            = NULL;
  scfenergies      = NULL;
  gradients        = NULL;
  charges          = NULL;
  chargetypes      = NULL;
}


/// copy constructor
QMTimestep::QMTimestep(const QMTimestep& ts) 
{
  num_scfiter = ts.num_scfiter;
  num_atoms   = ts.num_atoms;
  num_wavef   = ts.num_wavef;
  //wavef_size = ts.wavef_size;
  num_idtags = ts.num_idtags;
  num_charge_sets = ts.num_charge_sets;

  wavef            = NULL;
  wavef_id_map     = NULL;
  scfenergies      = NULL;
  gradients        = NULL;
  charges          = NULL;
  chargetypes      = NULL;

  if (ts.wavef) {
    int i;
    wavef = new Wavefunction[num_wavef];
    for (i=0; i<num_wavef; i++) {
      wavef[i] = ts.wavef[i];
    }
  }

  if (ts.wavef_id_map) {
    wavef_id_map = (int *)calloc(num_idtags, sizeof(int));
    memcpy(wavef_id_map, ts.wavef_id_map, num_idtags*sizeof(int));
  }

  if (ts.scfenergies) {
    scfenergies = new double[num_scfiter];
    memcpy(scfenergies, ts.scfenergies, num_scfiter*sizeof(double));
  }

  if (ts.gradients) {
    gradients = new float[3*num_atoms];
    memcpy(gradients, ts.gradients, 3*num_atoms*sizeof(float));
  }

  if (ts.charges) {
    charges = new double[num_atoms*num_charge_sets];
    memcpy(charges, ts.charges, num_atoms*num_charge_sets*sizeof(double));
  }

  if (ts.chargetypes) {
    chargetypes = new int[num_charge_sets];
    memcpy(chargetypes, ts.chargetypes, num_charge_sets*sizeof(int));
  }
}


/// destructor  
QMTimestep::~QMTimestep() 
{
  free(wavef_id_map);
  delete [] wavef;
  delete [] gradients;
  delete [] scfenergies;
  delete [] charges;
  delete [] chargetypes;
}

/// Get pointer to a wavefunction object
const Wavefunction* QMTimestep::get_wavefunction(int iwave)
{
  if (iwave<0 || iwave>=num_wavef) return NULL;
  return &wavef[iwave];
}

/// Get array of wavefunction coefficients
const float* QMTimestep::get_wavecoeffs(int iwave)
{
  if (iwave<0 || iwave>=num_wavef) return NULL;
  return wavef[iwave].get_coeffs();
}

/// Get array of orbital energies
const float* QMTimestep::get_orbitalenergy(int iwave)
{
  if (iwave<0 || iwave>=num_wavef) return NULL;
  return wavef[iwave].get_orbenergies();
}
 
/// Get array of orbital occupancies
const float* QMTimestep::get_occupancies(int iwave)
{
  if (iwave<0 || iwave>=num_wavef) return NULL;
  return wavef[iwave].occupancies;
}

/// Get array of orbital IDs
const int* QMTimestep::get_orbitalids(int iwave)
{
  if (iwave<0 || iwave>=num_wavef) return NULL;
  return wavef[iwave].orb_ids;
}

/// Get array of charges from specified charge set
const double* QMTimestep::get_charge_set(int iset)
{
  if (iset<0 || iset>=num_charge_sets) return NULL;
  return &charges[iset*num_atoms];
}

/// Get charge type of given charge set
int QMTimestep::get_charge_type(int iset)
{
  if (iset<0 || iset>=num_charge_sets) return -1;
  return chargetypes[iset];
}

/// Get charge type of given charge set
const char* QMTimestep::get_charge_type_str(int iset)
{
  if (iset<0 || iset>=num_charge_sets) return "";

  switch (chargetypes[iset]) {
  case MOLFILE_QMCHARGE_MULLIKEN:
    return "Mulliken";
    break;
  case MOLFILE_QMCHARGE_LOWDIN:
    return "Loewdin";
    break;
  case MOLFILE_QMCHARGE_ESP:
    return "ESP";
    break;
  case MOLFILE_QMCHARGE_NPA:
    return "NPA";
    break;
  default:
    return "Unknown";
  }
}

/// Get # coefficients for a wavefunction
int QMTimestep::get_num_coeffs(int iwave) {
  if (iwave<0 || iwave>=num_wavef) return -1;
  return wavef[iwave].num_coeffs;
}

/// Get # orbitals for a wavefunction
int QMTimestep::get_num_orbitals(int iwave) {
  if (iwave<0 || iwave>=num_wavef) return -1;
  return wavef[iwave].num_orbitals;
}

/// Get wavefunction ID from wavefunction index
int QMTimestep::get_waveid(int iwave) {
  if (iwave<0 || iwave>=num_wavef) return -1;
  return wavef[iwave].idtag;
}

/// Get spin for a wavefunction (alpha=0, beta=1)
int QMTimestep::get_spin(int iwave) {
  if (iwave<0 || iwave>=num_wavef) return -1;
  return wavef[iwave].spin;
}

/// Get electronic excitation level for a wavefunction (groundstate=0)
int QMTimestep::get_excitation(int iwave) {
  if (iwave<0 || iwave>=num_wavef) return -1;
  return wavef[iwave].excitation;
}

/// Get spin multiplicity for a wavefunction
int QMTimestep::get_multiplicity(int iwave) {
  if (iwave<0 || iwave>=num_wavef) return -1;
  return wavef[iwave].multiplicity;
}


/// Get the total energy of the electronic configuration
/// of this wavefunction.
double QMTimestep::get_wave_energy(int iwave) {
  if (iwave<0 || iwave>=num_wavef) return -1;
  return wavef[iwave].energy;
}

// Translate the wavefunction ID tag into the index the
// wavefunction has in this timestep
int QMTimestep::get_wavef_index(int idtag) {
  if (idtag<0 || idtag>=num_idtags) return -1;
  return wavef_id_map[idtag];
}

/// Add a new wavefunction object to the timestep
int QMTimestep::add_wavefunction(QMData *qmdata,
                                 int numcoeffs,
                                 int numorbitals, 
                                 const float *coeffs,
                                 const float *orbenergies,
                                 float *occupancies,
                                 const int *orbids,
                                 double energy,
                                 int type,
                                 int spin,
                                 int excitation,
                                 int multiplicity,
                                 char *info) 
{
  if (!numcoeffs || !numorbitals) return 0;
  int iwave = num_wavef;

  Wavefunction *newwavef = new Wavefunction[num_wavef+1];
  memset(newwavef, 0, (num_wavef+1)*sizeof(Wavefunction));
  int i;
  for (i=0; i<num_wavef; i++) {
    newwavef[i].movefrom(wavef[i]);
  }
  delete [] wavef;
  wavef = newwavef;
  num_wavef++;

  wavef[iwave].energy       = energy;
  wavef[iwave].type         = type;
  wavef[iwave].spin         = spin;
  wavef[iwave].excitation   = excitation;
  wavef[iwave].multiplicity = multiplicity;
  strncpy(wavef[iwave].info, info, QMDATA_BUFSIZ);
  
  wavef[iwave].set_coeffs(coeffs, numorbitals, numcoeffs);
  wavef[iwave].set_orbenergies(orbenergies, numorbitals);
  wavef[iwave].set_orbids(orbids, numorbitals);

  if (occupancies) {
    wavef[iwave].set_occupancies(occupancies, numorbitals);
  } else {
    // Assign the MO occupancies depending on RHF, ROHF, UHF
    vmd_set_default_occ(wavef[iwave].occupancies,
                        qmdata->scftype,
                        qmdata->num_electrons,
                        numorbitals,
                        wavef[iwave].multiplicity);
  }

  // Sort the wavefunction coefficients of each shell in
  // each orbital according to the angular momenta
  wavef[iwave].sort_wave_coefficients(qmdata);

  //msgInfo << "HOMO = " << get_homo(iwave) << sendmsg;
  return 1;
}

// Set timestep independent IDtag for a wavefunction
void QMTimestep::set_wavef_idtag(int iwave, int idtag) {
  if (iwave<0 || iwave>=num_wavef) return;
  wavef[iwave].idtag = idtag;
  if (idtag>=num_idtags) {
    if (!wavef_id_map) {
      wavef_id_map = (int *)calloc(1, sizeof(int));
    } else {
      wavef_id_map = (int *)realloc(wavef_id_map,
                                    (num_idtags+1)*sizeof(int));
    }
    num_idtags++;
  }
  wavef_id_map[idtag] = iwave;
}

void QMTimestep::set_scfenergies(const double *energies, int numscfiter) 
{
  if (!energies || !numscfiter) return;

  num_scfiter = numscfiter;
  scfenergies = new double[numscfiter];
  memcpy(scfenergies, energies, numscfiter*sizeof(double));
}


void QMTimestep::set_gradients(const float *grad, int natoms) {
  if (!grad || !natoms || natoms!=num_atoms) return;

  gradients = new float[3*num_atoms];
  memcpy(gradients, grad, 3*num_atoms*sizeof(float));
}

void QMTimestep::set_charges(const double *q, const int *qtype,
                             int natoms, int numqsets) {
  if (!q || !natoms || natoms!=num_atoms || !numqsets || !qtype)
    return;
  num_charge_sets = numqsets;
  charges = new double[num_atoms*num_charge_sets];
  memcpy(charges, q, num_atoms*num_charge_sets*sizeof(double));

  chargetypes = new int[num_charge_sets];
  memcpy(chargetypes, qtype, num_charge_sets*sizeof(int));
}


/// Find the HOMO, based on the orbital occupancies.
/// iwave is the wavefunction index, not the ID.
int QMTimestep::get_homo(int iwave) {
  if (iwave<0 || iwave>=num_wavef) return -1;
  const float *orbocc = wavef[iwave].get_occupancies();
  if (!orbocc) return -1;

  int i;
  int homo = -1;
  for (i=0; i<wavef[iwave].get_num_orbitals(); i++) {
    int intocc = (int)floor(0.5f+orbocc[i]);
    if (intocc>0) {
      homo = i;
    }
  }

  return homo;
}


/// Find the LUMO, i.e. the orbital with the smallest positive
/// or zero energy value. Note that for pathological systems
/// this way of identifying the the LUMO might give wrong
/// results, since higher energy orbitals could be occupied as well.
int QMTimestep::get_lumo(int iwave) {
  if (iwave<0 || iwave>=num_wavef) return -1;
  if (!wavef[iwave].get_occupancies()) return -1;
  return get_homo(iwave)+1;
}

/// Return occupancy and energy eigenvalue for a given orbital
void QMTimestep::get_orbital_occ_energy(int iwave, int orb, float &occ, float &energy) {
  if (wavef[iwave].get_occupancies() && orb>=0 && orb<wavef[iwave].get_num_orbitals())
    occ = wavef[iwave].get_occupancies()[orb];
  else
    occ = -1.f;      // undefined, return a sentinel value for now

  if (wavef[iwave].get_orbenergies())
    energy = wavef[iwave].get_orbenergies()[orb];
  else
    energy = -666.f; // undefined, return a sentinel value for now
}


/// Return the orbital ID for the orbital with the given
/// index in the specified wavefunction.
/// Returns -1 if requested wavefunction or orbital
/// doesn't exist.
int QMTimestep::get_orbital_id_from_index(int iwave, int index) {
  if (iwave<0 || iwave>=num_wavef) return -1;
  if (index<0 || index>=wavef[iwave].num_orbitals) return -1;
  return wavef[iwave].orb_ids[index];
}


/// Return the orbital index for the orbital with the given
/// 1-based ID in the specified wavefunction.
/// Returns -1 if requested wavefunction or orbital
/// doesn't exist.
int QMTimestep::get_orbital_index_from_id(int iwave, int id) {
  if (iwave<0 || iwave>=num_wavef) return -1;
  if (id<1 || id>wavef[iwave].num_orbitals) return -1;
  return wavef[iwave].orb_id2index[id];
}


// Generate mapping that sorts the orbitals by similarity
// throughout the trajectory (rather than by energy).
void QMTimestep::sort_orbitals(Timestep *previous_ts) {
  // XXX unfinished thus commented out.
#if 0
  QMTimestep *prev_qmts = previous_ts->qm_timestep;
  float *scores = new float[orbital_counter];
  int i, j, k;
  for (i=0; i<orbital_counter; i++) {
    printf("Orbital %d\n", i);
    for (j=0; j<orbital_counter; j++) {
      float dot = 0.f;
      // Compute the square sum
      for (k=0; k<wavef_size; k++) {
        float d = prev_qmts->wave_function[i*orbital_counter+k] 
          * wave_function[j*orbital_counter+k];
        dot += d;
      }
      scores[j] = dot/wavef_size;
      printf("score[%d] = % .3f\n", j, scores[j]);
    }
  }
  delete [] scores;
#endif
}



// The standard quicksort algorithm except for it doesn't
// sort the data itself but rather sorts array of ints *idx
// in the same order as it would sort the integers in array
// *tag. Array *idx can then be used to reorder any array
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
    } else {
      return j;
    }
  }
}


#if 0
static void quicksort(const char **tag, int *A, int p, int r);
static int  quickpart(const char **tag, int *A, int p, int r);

// Sorts the indexlist such that the indexes refer to elements in
// tag in an increasing order.
void QMTimestep::sort_shell(QMData *qmdata, int atom, int ishell) {
  int i;
  const shell_t *shell = qmdata->get_basis(atom, ishell);

  // Initialize the index array;
  int *index = new int[shell->num_cart_func];
  for (i=0; i<shell->num_cart_func; i++) index[i] = i;

  // Initialize array of sortable tag strings
  const char **tag = new const char*[shell->num_cart_func];
  for (i=0; i<shell->num_cart_func; i++) {
    tag[i] = qmdata->get_angular_momentum(atom, ishell, i);
  }

  float *wave_f = wave_function + shell->wave_offset;
#if DEBUG
  for (i=0; i<shell->num_cart_func; i++) {
    printf("unsorted %i: %s %f\n", i, tag[i], wave_f[i]);
  }
#endif

  // Sort index array according to tags.
  quicksort(tag, index, 0, shell->num_cart_func-1);

#if DEBUG
  for (i=0; i<shell->num_cart_func; i++) {
    printf("sorted %i: %s %f\n", i, tag[index[i]], wave_f[i]);
  }
#endif

  // Copy data over into sorted arrays
  float *sorted_wave_f = new float[shell->num_cart_func];
  char **sorted_tag = new char*[shell->num_cart_func];
  for (i=0; i<shell->num_cart_func; i++) {
    sorted_wave_f[i] = wave_f[index[i]];
    sorted_tag[i] = new char[strlen(tag[index[i]])+1];
    strcpy(sorted_tag[i], tag[index[i]]);
  }

  // Copy sorted data back into original arrays
  for (i=0; i<shell->num_cart_func; i++) {
    wave_f[i] = sorted_wave_f[i];
    qmdata->set_angular_momentum_str(atom, ishell, i, sorted_tag[i]);
  }

  for (i=0; i<shell->num_cart_func; i++) {
    delete [] tag[i];
    delete [] sorted_tag[i];
  }
  delete [] sorted_wave_f;
  delete [] sorted_tag;
  delete [] index;
}


// The standard quicksort algorithm except for it doesn't
// sort the data itself but rather sorts array of ints *A
// in the same order as it would sort the strings in array
// **tag. Array *A can then be used to reorder any array
// according to the string tags.
// Example:
// tag:   BC DD BB AA  -->  AA BB BC DD
// index:  0  1  2  3  -->   3  2  0  1
//
static void quicksort(const char **tag, int *idx, int p, int r) {
  int q;
  if (p < r) {
    q=quickpart(tag, idx, p, r);
    quicksort(tag, idx, p, q);
    quicksort(tag, idx, q+1, r);
  }
}


// Partitioning for quicksort.
static int quickpart(const char **tag, int *idx,
                             int p, int r) {
  int i, j;
  int tmp;
  const char *x = tag[idx[p]];
  i = p-1;
  j = r+1;

  while (1) {
    // Find highest element smaller than idx[p]
    do j--; while (strcmp(tag[idx[j]], x) > 0);

    // Find lowest element larger than idx[p]
    do i++; while (strcmp(tag[idx[i]], x) < 0);

    if (i < j) {
      tmp    = idx[i];
      idx[i] = idx[j];
      idx[j] = tmp;

    } else {
      return j;
    }
  }
}

#endif


// Assign default occupancies depending on calculation method,
// number of electrons and multiplicity.
// Memory for array *occupancies will be allocated.
void vmd_set_default_occ(float *(&occupancies), int scftype, int numelec, 
                         int numorbitals, int multiplicity) 
{
  if (!numorbitals) return;
  if (occupancies) {
    delete [] occupancies;
  }
  occupancies = new float[numorbitals];

  int i;
  for (i=0; i<numorbitals; i++) {
    switch(scftype) {
      case SCFTYPE_UNKNOWN:
        occupancies[i] = -1.f;
        break;

      case SCFTYPE_RHF:
        if (i<numelec/2) occupancies[i] = 2;
        else             occupancies[i] = 0;
        break;
        
      case SCFTYPE_UHF:
        if (i<(numelec/2+multiplicity/2)) {
          occupancies[i] = 1.f;
        } else if ((i >= numorbitals/2) && 
                   (i < (numorbitals/2 + numelec/2+multiplicity/2))) {
          occupancies[i] = 1.f;
        } else {
          occupancies[i] = 0.f;
        }
        break;

      case SCFTYPE_ROHF:
        if (i<(numelec/2-multiplicity/2))      occupancies[i] = 2.f;
        else if (i<(numelec/2+multiplicity/2)) occupancies[i] = 1.f;
        else                                   occupancies[i] = 0.f;
        break;

      default:
        occupancies[i] = -1.f;
        break;
    }
  }
}
