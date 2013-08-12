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
 *	$RCSfile: QMTimestep.h,v $
 *	$Author: saam $	$Locker:  $		$State: Exp $
 *	$Revision: 1.32 $	$Date: 2009/07/28 21:58:50 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The QMTimestep class, which stores orbitals, SCF energies, etc. for a
 * single timestep.
 *
 ***************************************************************************/
#ifndef QMTIMESTEP_H
#define QMTIMESTEP_H

#include "QMData.h"

class Timestep;

class Wavefunction {
  friend class QMTimestep;
  int   idtag;              /**< unique tag to identify this wavefunction over the trajectory */
  int   type;               /**< CANONICAL, LOCALIZED, OTHER */
  int   spin;               /**< 0 for alpha, 1 for beta */
  int   excitation;         /**< 0 for ground state, 1,2,3,... for excited states */
  int   multiplicity;       /**< spin multiplicity of the state,
                             *   zero if unknown */
  int   num_orbitals;       /**< number of orbitals that was really 
                             *   present in the output for this step */
  int   num_coeffs;         /**< number of coefficients per orbital */
  //int   have_energies;      /**< number of orbital energies */
  //int   have_occup;         /**< number of occupancies */
  char info[QMDATA_BUFSIZ]; /**< string for additional type info */

  double energy;            /**< energy of the electronic state.
                             *   i.e. HF-SCF energy, CI state energy,
                             *   MCSCF energy, etc. */

  float *wave_coeffs;       /**< expansion coefficients for wavefunction in the
                             *   form {orbital1(c1),orbital1(c2),.....,orbitalM(cN)} */
  float *orb_energies;      /**< list of orbital energies for wavefunction */
  float *occupancies;       /**< orbital occupancies */
  int   *orb_ids;           /**< orbital IDs provided by plugin
                             *   default is 1,2,3,...,num_orbitals according
                             *   to order by increasing energy */
  int   *orb_id2index;      /**< array of length num_coeffs+1 storing the 
                             *   corresponding index for each orb_id. */

 public:
  Wavefunction();
  Wavefunction(const Wavefunction& wf);
  Wavefunction(int numcoeffs,
               int numorbitals, 
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
               char *infostr);
  ~Wavefunction();

  Wavefunction& operator=(const Wavefunction& wf);

  // Move the data over from the given wavefunction wf
  // and set the pointers in wf to NULL.
  // This avoids copying the arrays.
  void movefrom(Wavefunction& wf);

  int get_num_orbitals()    { return num_orbitals; }
  int get_num_coeffs()      { return num_coeffs; }

  /// Get pointer to the wavefunction coefficients
  const float* get_coeffs()      { return wave_coeffs; }

  /// Get pointer to the orbitals energies
  const float* get_orbenergies() { return orb_energies; }

  /// Get pointer to the orbital occupancies
  const float* get_occupancies() { return occupancies; }

  /// Get pointer to the orbital indices
  const int*   get_orbids() { return orb_ids; }

  //  const float* get_coeffs(int orb);

  void set_coeffs(const float *wfn, int norbitals, int wavef_size);
  void set_orbenergies(const float *energies, int norbitals);
  void set_occupancies(const float *occupancies, int norbitals);
  void set_orbids(const int *orbids, int norbitals);

  //float get_coeff(int orb, int i);
  float get_orbitalenergy(int orb);


  int  get_type() { return type; }
  void get_typestr(char *&typestr);

  void sort_wave_coefficients(QMData *qmdata);

 private:
  void sort_incr(QMData *qmdata, int atom, int ishell, int comp, int first, int num);

};

/// Timesteps store coordinates, energies, etc. for one trajectory timestep
class QMTimestep {
private:
  int   num_scfiter;        ///< # SCF iterations
  int   num_atoms;          ///< # atoms (size of gradient array)
#if 0
  int   wavef_size;         ///< # basisfunctions, i.e. the number of 
                            ///< coefficients the wavefunction is expanded in.  
#endif
  int num_wavef;            ///< # wavefunction for this ts
  Wavefunction *wavef;      ///< array of wavefunctions

  int num_idtags;           ///< # wavefunction IDtags
                            // XXX (should be same as num_wavef)

  int *wavef_id_map;        ///< maps timestep independent wavefunction
                            ///< IDtags to wavefunction indices used in
                            ///< this timestep.

  double *scfenergies;      ///< SCF energy for each iteration
  float  *gradients;        ///< energy gradient for each atom

  int num_charge_sets;      ///< number of different charge fields
  double *charges;          ///< per-atom charge sets
  int    *chargetypes;      ///< specifies type for each charge set
                            ///< (e.g. MOLFILE_QMCHARGE_MULLIKEN)

public:
  QMTimestep(int natoms);           ///< constructor:
  QMTimestep(const QMTimestep& ts); ///< copy constructor
  ~QMTimestep(void);                ///< destructor

  /// Add a new wavefunction object to the timestep
  int add_wavefunction(QMData *qmdata,
                       int numcoeffs,
                       int numorbitals, 
                       const float *coeffs,
                       const float *energies,
                       float *occupancies,
                       const int   *orbids,
                       double energy,
                       int type,
                       int spin,
                       int excitation,
                       int multiplicity,
                       char *info);

  // Initialization functions
  void set_scfenergies(const double *energies, int numscfiter);
  void set_wavefunction(const float *wfn, int numorbitals, int num_gauss_basis_funcs);
  void set_orbitalenergies(const float *energies, int numorbitals);
  void set_gradients(const float *grad, int numatoms);
  void set_charges(const double *q, const int *qtype, 
                   int numatoms, int numqsets);

  /// Set timestep independent IDtag for a wavefunction
  void set_wavef_idtag(int iwave, int idtag);

  /// Get pointer to a wavefunction object
  const Wavefunction* get_wavefunction(int iwave);

  /// Get array of wavefunction coefficients
  const float*  get_wavecoeffs(int iwave);
 
  /// Get array of orbital energies
  const float*  get_orbitalenergy(int iwave);

  /// Get array of orbital occupancies
  const float*  get_occupancies(int iwave);

  /// Get array of orbital IDs
  const int*    get_orbitalids(int iwave);

  /// Get energy gradient for each atom
  const float*  get_gradients()     { return gradients;   }

  /// Get array of SCF energies
  const double* get_scfenergies()   { return scfenergies; }

  /// Get array of charges from specified charge set
  const double* get_charge_set(int set);

  /// Get charge type of given charge set
  int get_charge_type(int iset);

  /// Get charge type of given charge set
  const char* get_charge_type_str(int iset);

  /// Get # SCF iterations
  int get_num_scfiter() { return num_scfiter; }

  /// Get # SCF iterations
  int get_num_charge_sets() { return num_charge_sets; }

  /// Get # wavefunctions
  int get_num_wavef()   { return num_wavef;   }

  /// Get # coefficients for a wavefunction
  int get_num_coeffs(int iwave);

  /// Get # orbitals for a wavefunction
  int get_num_orbitals(int iwave);

  /// Get wavefunction index used in this timstep from
  /// timestep independent IDtag
  int get_wavef_index(int idtag);

  /// Get timestep independent wavefunction ID from
  /// wavefunction index used in this timestep
  int get_waveid(int iwave);

  /// Get spin for a wavefunction (alpha=0, beta=1)
  int get_spin(int iwave);

  /// Get electronic excitation level for a wavefunction (groundstate=0)
  int get_excitation(int iwave);

  /// Get spin multiplicity for a wavefunction
  int get_multiplicity(int iwave);

  /// Get the total energy of the electronic configuration
  /// of this wavefunction.
  double get_wave_energy(int iwave);

  // Get orbital index for HOMO
  int get_homo(int iwave);

  // Get orbital index for LUMO
  int get_lumo(int iwave);

  void get_orbital_occ_energy(int iwave, int orb, float &occ, float &energy);

  // Get string describing the wavefunction type
  void get_wavef_typestr(int iwave, char *&typestr) {
    wavef[iwave].get_typestr(typestr);
  }

  // Generate mapping that sorts the orbitals by similarity
  // throughout the trajectory (rather than by energy).
  // XXX Still unfinished.
  void sort_orbitals(Timestep *previous_ts);

  /// Return the orbital ID for the orbital with the given
  /// index in the specified wavefunction.
  /// Returns -1 if requested wavefunction or orbital
  /// doesn't exist.
  int get_orbital_id_from_index(int iwave, int index);

  /// Return the orbital index for the orbital with the given
  /// ID in the specified wavefunction.
  /// Returns -1 if requested wavefunction or orbital
  /// doesn't exist.
  int get_orbital_index_from_id(int iwave, int id);

private:
  void sort_shell(QMData *qmdata, int atom, int ishell);
  void sort_incr(QMData *qmdata, int atom, int ishell, int comp, int first, int num);
};

/// Assign default occupancies depending on calculation method,
/// number of electrons and multiplicity.
/// Memory for array *occupancies will be allocated.
void vmd_set_default_occ(float *(&occupancies), int scftyp, int numelec, int numorbitals, int multiplicity);


#endif

