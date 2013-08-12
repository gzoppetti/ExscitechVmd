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
 *	$RCSfile: Timestep.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.46 $	$Date: 2009/04/29 15:43:28 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The Timestep class, which stores coordinates, energies, etc. for a
 * single timestep.
 *
 * Note: As more data is stored for each step, it should go in here.  For
 * example, H-Bonds could be calculated each step.
 ***************************************************************************/
#ifndef TIMESTEP_H
#define TIMESTEP_H

#include "ResizeArray.h"
#include "Matrix4.h"

#if defined(VMDWITHORBITALS)
#include "QMTimestep.h"
#endif

// Energy terms and temperature stored for each timestep
// TSENERGIES must be the last element.  It indicates the number
// energies.  (TSE_TOTAL is the total energy).  If you add fields here
// you should also add the lines in MolInfo.C so you can get access to
// the fields from Tcl.
enum { TSE_BOND, TSE_ANGLE, TSE_DIHE, TSE_IMPR, TSE_VDW, TSE_COUL,
       TSE_HBOND, TSE_KE, TSE_PE, TSE_TEMP, TSE_TOTAL, TSE_VOLUME,
       TSE_PRESSURE, TSE_EFIELD, TSE_UREY_BRADLEY, TSE_RESTRAINT,
       TSENERGIES};

/// Timesteps store coordinates, energies, etc. for one trajectory timestep
class Timestep {
public:
  int num;                  ///< number of atoms this timestep is for
  float *pos;               ///< coords for all atoms, as (x,y,z), (x,y,z), ...
  float *vel;               ///< velocites for all atoms as (vx,vy,vz), ...
  float *force;             ///< forces for all atoms.
  float *user;              ///< Demand-allocated 1-float-per-atom 'User' data
  float *user2;             ///< Demand-allocated 1-float-per-atom 'User' data
  float *user3;             ///< Demand-allocated 1-float-per-atom 'User' data
  float *user4;             ///< Demand-allocated 1-float-per-atom 'User' data
#if defined(VMDWITHORBITALS)
  QMTimestep *qm_timestep;
#endif
  float energy[TSENERGIES]; ///< energy for this step; by default, all 0
  int timesteps;            ///< timesteps elapsed so far (if known)
  double physical_time;     ///< Physical time corresponding to this timestep

  /// Size and shape of unit cell 
  float a_length, b_length, c_length, alpha, beta, gamma;

  /// Get vectors corresponding to periodic image vectors
  void get_transform_vectors(float v1[3], float v2[3], float v3[3]) const;
 
  /// Compute transformations from current unit cell dimensions
  void get_transforms(Matrix4 &a, Matrix4 &b, Matrix4 &c) const;

  /// Convert (na, nb, nc) tuple to a transformation based on the current
  /// unit cell.
  void get_transform_from_cell(const int *cell, Matrix4 &trans) const;

  Timestep(int n);              ///< constructor: # atoms
  Timestep(const Timestep& ts); ///< copy constructor
  ~Timestep(void);              ///< destructor
  
  void zero_values();           ///< set the coords to 0
};

#endif

