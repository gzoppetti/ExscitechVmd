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
 *	$RCSfile: QMData.h,v $
 *	$Author: saam $	$Locker:  $		$State: Exp $
 *	$Revision: 1.68 $	$Date: 2009/07/28 21:58:50 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The QMData class, which stores all QM simulation data that
 * are not dependent on the timestep.
 *
 ***************************************************************************/
#ifndef QMDATA_H
#define QMDATA_H


#define GUI_WAVEF_TYPE_CANON    0
#define GUI_WAVEF_TYPE_GEMINAL  1
#define GUI_WAVEF_TYPE_MCSCFNAT 2
#define GUI_WAVEF_TYPE_MCSCFOPT 3
#define GUI_WAVEF_TYPE_CINAT    4
#define GUI_WAVEF_TYPE_LOCAL    5
#define GUI_WAVEF_TYPE_OTHER    6

#define GUI_WAVEF_SPIN_ALPHA    0
#define GUI_WAVEF_SPIN_BETA     1

#define GUI_WAVEF_EXCI_GROUND   0

// NOTE: this has to be kept in sync with the corresponding table
//       in the QM molfile plugin reader. 
// XXX   These should all be added to the molfile_plugin header
//       When copied from the plugin to the internal data structure,
//       we can either arrange that the molfile plugin constants and these
//       are always identical, or we can use a switch() block to translate 
//       between them if they ever diverge.
#define SPD_D_SHELL -5
#define SPD_P_SHELL -4
#define SPD_S_SHELL -3
#define SP_S_SHELL  -2
#define SP_P_SHELL  -1
#define S_SHELL      0
#define P_SHELL      1
#define D_SHELL      2
#define F_SHELL      3
#define G_SHELL      4
#define H_SHELL      5
#define I_SHELL      6

#define QMDATA_BUFSIZ       81   ///< maximum chars in string data
#define QMDATA_BIGBUFSIZ  8192   ///< maximum chars in runtitle string data

// forward declarations.
class Orbital;
class QMTimestep;
class Molecule;

//! scf types
enum { SCFTYPE_UNKNOWN=0, SCFTYPE_RHF=1, SCFTYPE_UHF=2, SCFTYPE_ROHF=3,
       SCFTYPE_GVB=4, SCFTYPE_MCSCF=5, SCFTYPE_FF=6, 
       SCFTYPE_TOTAL=7 };

//! run types
//! XXX don't know what 8,9,10 are.
//!     I suppose Axel added them for Gaussian.
enum { RUNTYPE_UNKNOWN=0,  RUNTYPE_ENERGY=1,  RUNTYPE_OPTIMIZE=2, 
       RUNTYPE_SADPOINT=3, RUNTYPE_HESSIAN=4, RUNTYPE_SURFACE=5,
       RUNTYPE_GRADIENT=6, RUNTYPE_MEX=7,     RUNTYPE_DYNAMICS=8,
       RUNTYPE_PROPERTIES=9, RUNTYPE_TOTAL=10 };


// Signature of a wavefunction, used to identify
// a wavefunction thoughout the trajectory.
// The orbital metadata are needed by the GUI to
// determine what is the union of orbitals showing up
// over the trajetory. It is displayed as the available
// orbital list in the representations window.
typedef struct {
  int type;
  int spin;
  int exci;
  char info[QMDATA_BUFSIZ];

  int max_avail_orbs; // maximum number of available orbitals
                      // over all frames.
  int *orbids;        // list of existing orbital IDs
  float *orbocc;      // occupancies of all available orbitals
} wavef_signa_t;


// Basis set data structures
// -------------------------

//! A Gaussian primitive
typedef struct {
  float expon;       ///< gaussian exponent
  float coeff;       ///< contraction coefficient
} prim_t;

//! A Shell (Gaussian type orbital)
typedef struct {
  int numprims;      ///< number of gaussian primitives 
  int symmetry;      ///< 0, 1, 2, ... for S, P, D, ...
  int sp_shell;      ///< true for SP-shells;

  int num_cart_func; ///< Number of cartesian functions for this shell
                     ///< symmetry, i.e. size of the wave_coeff array.
                     ///< This is determined by the shell type:
                     ///< S=1, P=3, D=6, F=10, ...
  float *norm_fac;   ///< pointer to the table of angular normalization factors
  float *basis;      ///< pointer to basisset array
  prim_t *prim;      ///< array of primitives
} shell_t;

//! Basis set definition for one atom
typedef struct {
  int atomicnum;     ///< atomic number (chemical element)
  int numshells;     ///< number of shells for atom
  shell_t *shell;    ///< pointer to list of shells
} basis_atom_t;

//! QM data management class.
//! XXX: a lot of the arrays that are internal are exposed as 'public'
//!      might be better to write access methods (if needed) or declare
//!      classes that need access as "friend".
class QMData {
 public:
  // Orbital data
  int num_wave_f;    ///< max. size of the wave_function array (stored
                     ///< in QMTimestep). The actual number of MOs
                     ///< present can be different for each frame but
                     ///< this is the maximum number of possible occupied
                     ///< and virtual orbitals, i.e. this is the number
                     ///< of contracted cartesian gaussian basis functions
                     ///< or the size of the secular equation.

  int num_basis;     ///< Number of the {exp, coeff) pairs in basis array.
                     ///< This is NOT the same as above since each
                     ///< (P, D, F, ...) shell consists of (3, 6, 10, ...)
                     ///< cartesian functions!
  int num_wavef_signa; ///< total number of wavefunctions with different
                       ///< signatures (over all frames)

 private:
  int num_types;     ///< number of unique atoms in basis set
  int num_atoms;     ///< number of atoms (size of the next 3 arrays)

  int *atom_types;          ///< maps each atom to an atomic basis set;
  int *atom_sort;           ///< atom indexes sorted by atom type
  int *num_shells_per_atom; ///< number of shells per atom

  int *wave_offset;         ///< offset to get to the beginning of each atom 
                            ///< in the wave_function array. This takes the  
                            ///< number of shells per atom, the number of
                            ///< prims per shell and the number of angular
                            ///< momenta per shell into account.

  int num_shells;           ///< total number of shells in basis set
  int *num_prim_per_shell;  ///< number of gaussian primitives per shell i

  int *angular_momentum;    ///< 3 ints per wave function coeff. describe the 
                            ///< cartesian components of the angular momentum.
                            ///< E.g. S={0 0 0}, Px={1 0 0}, 
                            ///< Dxy={1 1 0}, or Fyyz={0 2 1}.

  int highest_shell;        ///< highest shell that occurs in the basis set,
                            ///< i.e. size-1 of **norm_factors.

  float **norm_factors;     ///< Normalization factors for different shell types
                            ///< and their cartesian functions.
                            ///< Each shell splits into n cartesian functions
                            ///< with different normalization factors which can
                            ///< be applied on the fly or premultiplied
                            ///< with the wave function coefficients.

  float *basis_array;       ///< Contraction coeffients and exponents for 
                            ///< the basis functions in the form 
                            ///< {exp1, coeff1, exp2, coeff2, ...}.
                            ///< Contraction coefficients multiplied by the
                            ///< shell-specific normalization constants.
                            ///< SP-shells (L in GAMESS) are expanded 
                            ///< into S and P.
                            ///< Array has a length of 2*num_basis_entries.
                            ///< It is used for computing orbitals using 
                            ///< SSE or on the GPU.

  int *atom_basis;          ///< Offset into basis_array for each atom

  int *shell_symmetry;      ///< Symmetry type per (exp(),c-coeff()) pair 
                            ///< in basis_array (0, 1, 2,...) corresponds
                            ///< to (S, P, D,...).
                            ///< Convenient access use the basis_set 
                            ///< data structure.

  basis_atom_t *basis_set;  ///< Hierarchical representation of basis set 
                            ///< for easier access: Array containing 
                            ///< basis set definitions for each atom each of 
                            ///< which contain basis set definitions for
                            ///< all the shells in that atom.
                            ///< Represents the unnormalized basis set 
                            ///< as read from file.
  

  wavef_signa_t *wavef_signa; ///< Array wavefunction signature containing
                              ///< type, spin and excitation. Accessible
                              ///< via the wavefunction idtags.

  //public:
  int have_esp;
  int have_npa;
  int have_carthessian;
  int have_internals;
  int have_normalmodes;
private:
  // Hessian and frequency data
  int nimag;                ///< number of imaginary modes
  int nintcoords;           ///< number of internal coordinates

  double *carthessian;     ///< hessian matrix in cartesian coordinates 
                           ///< (3*natoms)*(3*natoms) as a single array of 
                           ///< doubles (row(1), ...,row(natoms))
  double *inthessian;      ///< hessian matrix in internal coordinates 
                           ///< (nintcoords*nintcoords) as a single array 
                           ///< of doubles (row(1), ...,row(nintcoords))
  float *wavenumbers;      ///< array(3*natoms) of wavenumbers of normal modes
  float *intensities;      ///< array(3*natoms) of intensities of normal modes
  float *normalmodes;      ///< matrix(3*natoms*3*natoms) of normal modes
  int   *imagmodes;        ///< array(nimag) of normalmode indices
                           ///< specifying the imaginary modes.

public:
  // QM run info
  int runtype;             ///< runtype for internal use
  int scftype;             ///< scftype for internal use
  int status;              ///< SCF and optimization status
  int nproc;               ///< number of processors used
  int memory;              ///< amount of memory used in MByte
  int num_electrons;       ///< number of electrons
  int totalcharge;         ///< total charge of system
  //int num_orbitals_A;      ///< number of alpha orbitals
  //int num_orbitals_B;      ///< number of beta orbitals

  char *basis_string;                  ///< basis name as "nice" string
  char runtitle[QMDATA_BIGBUFSIZ];     ///< title of run. (large for Gaussian)
  char geometry[QMDATA_BUFSIZ];        ///< type of provided geometry,
                                       ///< e.g. UNIQUE, ZMT, CART, ...
  char version_string[QMDATA_BUFSIZ];  ///< QM code version information

  QMData(int natoms, int nbasis, int shells, int nwave); ///< constructor:
  ~QMData(void);                       ///< destructor


  /// Initialize total charge, multiplicity and everything that can
  /// be derived from that.
  void init_electrons(Molecule *mol, int totcharge);


  // ====================================
  // Functions for basis set initializion
  // ====================================

  /// Free memory of the basis set.
  void delete_basis_set();

  /// Populate basis set data and organize them into
  /// hierarcical data structures.
  int init_basis(Molecule *mol, int num_basis_atoms, const char *string,
                 const float *basis, const int *atomic_numbers,
                 const int *nshells, const int *nprim,
                 const int *symm);

  /// Collapse basis set so that we have one atomic basis set
  /// per atom type.
  int create_unique_basis(Molecule *mol, int num_basis_atoms);

  /// Multiply contraction coefficients with the shell
  /// dependent part of the normalization factor.
  void normalize_basis();

#if 0                           // XXX: unused
  /// Create an index array that sorts the atoms by basis set type
  void sort_atoms_by_type();
#endif

  /// Initialize table of normalization factors containing different factors for each 
  /// shell and its cartesian functions.
  void init_angular_norm_factors();


  // =================
  // Basis set acccess
  // =================

  /// Get basis set for an atom
  const basis_atom_t* get_basis(int atom=0) const;

  /// Get basis set for a shell
  const shell_t*      get_basis(int atom, int shell) const;

  const int* get_atom_types()          const { return atom_types; }
  const int* get_num_shells_per_atom() const { return num_shells_per_atom; }
  const int* get_num_prim_per_shell()  const { return num_prim_per_shell; }

  int get_num_atoms()                { return num_atoms; }
  int get_num_types()                { return num_types; }
  int get_num_shells()               { return num_shells; }

  /// Get the offset in the wavefunction array for given shell
  int get_wave_offset(int atom, int shell=0) const;

  /// Get shell type letter (S, P, D, F, ...) followed by '\0'
  const char* get_shell_type_str(const shell_t *shell);

  char* get_angular_momentum_str(int atom, int shell, int mom) const;
  void  set_angular_momentum_str(int atom, int shell, int mom,
                                 const char *tag);
  void  set_angular_momentum(int atom, int shell, int mom,
                             int *pow);
  int   get_angular_momentum(int atom, int shell, int mom, int comp);
  int   set_angular_momenta(const int *angmom);


  // ========================
  // Hessian and normal modes
  // ========================

  int get_num_imag()       { return nimag; }
  int get_num_intcoords()  { return nintcoords; }
  void set_carthessian(int numcartcoords, double *array);
  void set_inthessian(int numintcoords, double *array);
  void set_normalmodes(int numcart, float *array);
  void set_wavenumbers(int numcart, float *array);
  void set_intensities(int numcart, float *array);
  void set_imagmodes(int numimag, int *array);
  const double* get_carthessian() const { return carthessian; }
  const double* get_inthessian()  const { return inthessian;  }
  const float*  get_normalmodes() const { return normalmodes; }
  const float*  get_wavenumbers() const { return wavenumbers; }
  const float*  get_intensities() const { return intensities; }
  const int*    get_imagmodes()   const { return imagmodes;   }


  // =====================
  // Calculation meta data
  // =====================

  /// Translate the scftype constant into a string
  const char *get_scftype_string(void) const;

  /// Translate the runtype constant into a string
  const char *get_runtype_string(void) const;
  
  /// Get status of SCF and optimization convergence
  const char* get_status_string();


  // =======================
  // Wavefunction signatures
  // =======================

  /// Determines and returns a unique ID for each wavefuntion
  /// based on it's signature (type, spin, excitation, info).
  int assign_wavef_id(int type, int spin, int excit, char *info,
                      wavef_signa_t *(&signa_ts), int &numsig);

  /// Matches the wavefunction selected from the GUI buttons
  /// (type, spin, excitation) with a wavefunction IDtag.
  /// Returns -1 if no such wavefunction exists.
  int find_wavef_id_from_gui_specs(int type, int spin, int exci);


  /// Return 1 if we have a wavefunction with the given type
  int has_wavef_type(int type);

  /// Return 1 if we have any wavefunction with the given spin
  int has_wavef_spin(int spin);

  /// Return 1 if we have any wavefunction with the given excitation
  int has_wavef_exci(int exci);

  /// Return 1 if we have any wavefunction with the given
  /// signature (type, spin, and excitation).
  int has_wavef_signa(int type, int spin, int exci);

  /// Get the highest excitation for any wavefunction 
  /// with the given type.
  int get_highest_excitation(int type);


  // ========================================================
  // Functions dealing with the list of orbitals available 
  // for a given wavefunction. Needed for the GUI.
  // ========================================================

  /// Update the list of available orbitals
  void update_avail_orbs(int iwavesig, int norbitals,
                         const int *orbids, const float *orbocc);

  /// Return the maximum number of available orbitals
  /// for the given wavefunction over all frames
  /// Can be used to determine the number of orbitals
  /// to be displayed in the GUI.
  /// Returns -1 if requested wavefunction doesn't exist.
  int get_max_avail_orbitals(int iwavesig);

  /// Get IDs of all orbitals available for the given wavefunction.
  /// iwavesig is the index of the wavefunction signature.
  /// Returns 1 upon success, 0 otherwise.
  int get_avail_orbitals(int iwavesig, int *(&orbids));

  /// Get the occupancies of all available orbitals
  /// for the given wavefunction.
  int get_avail_occupancies(int iwavesig, float *(&orbocc));

  /// For the given wavefunction signature return
  /// the iorb-th orbital label. Used to translate from the
  /// GUI list of available orbitals the unique orbital label.
  int get_orbital_label_from_gui_index(int iwavesig, int iorb);

  /// Return 1 if the given wavefunction has an orbital with
  /// ID orbid in any frame.
  int has_orbital(int iwavesig, int orbid);

  /// Create a new Orbital and return the pointer.
  /// User is responsible for deleting.
  Orbital* create_orbital(int iwave, int orbid, float *pos,
                          QMTimestep *qmts);

};



#endif

