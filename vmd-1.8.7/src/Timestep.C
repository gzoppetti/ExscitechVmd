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
 *	$RCSfile: Timestep.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.60 $	$Date: 2009/04/29 15:43:27 $
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

#include <math.h>
#include "Timestep.h"
#include "Inform.h"
#include "utilities.h"

///  constructor  
Timestep::Timestep(int n) {
  for(int i=0; i < TSENERGIES; energy[i++] = 0.0);
  num = n;
  pos = new float[3 * num];
  vel = NULL;
  force = NULL;
  user  = NULL;
  user2 = NULL;
  user3 = NULL;
  user4 = NULL;
#if defined(VMDWITHORBITALS)
  qm_timestep = NULL;
#endif
  a_length = b_length = c_length = 0;
  alpha = beta = gamma = 90;
  timesteps=0;
  physical_time=0;
}


/// copy constructor
Timestep::Timestep(const Timestep& ts) {
  num = ts.num;

  pos = new float[3 * num];
  memcpy(pos, ts.pos, 3 * num * sizeof(float));

  if (ts.force) {
    force = new float[3 * num];
    memcpy(force, ts.force, 3 * num * sizeof(float));
  } else {
    force = NULL;
  }

  if (ts.vel) {
    vel = new float[3 * num];
    memcpy(vel, ts.vel, 3 * num * sizeof(float));
  } else {
    vel = NULL;
  }

  if (ts.user) {
    user = new float[num];
    memcpy(user, ts.user, num*sizeof(float));
  } else {
    user = NULL;
  }

  if (ts.user2) {
    user2 = new float[num];
    memcpy(user2, ts.user2, num*sizeof(float));
  } else {
    user2 = NULL;
  }

  if (ts.user3) {
    user3 = new float[num];
    memcpy(user3, ts.user3, num*sizeof(float));
  } else {
    user3 = NULL;
  }

  if (ts.user4) {
    user4 = new float[num];
    memcpy(user4, ts.user4, num*sizeof(float));
  } else {
    user4 = NULL;
  }

#if defined(VMDWITHORBITALS)
  if (ts.qm_timestep) {
    qm_timestep = new QMTimestep(*(ts.qm_timestep));
  } else {
    qm_timestep = NULL;
  }
#endif

  memcpy(energy, ts.energy, sizeof(ts.energy));
  a_length = ts.a_length;
  b_length = ts.b_length;
  c_length = ts.c_length;
  alpha = ts.alpha;
  beta = ts.beta;
  gamma = ts.gamma;
  timesteps=ts.timesteps;
  physical_time=ts.physical_time;
}


/// destructor  
Timestep::~Timestep() {
  delete [] force;
  delete [] vel;
  delete [] pos;
  delete [] user;
  delete [] user2;
  delete [] user3;
  delete [] user4;
#if defined(VMDWITHORBITALS)
  if (qm_timestep) delete qm_timestep;
#endif
}

// reset coords and related items to 0
void Timestep::zero_values() {
  if (num <= 0) 
    return;
    
  memset(pos,0,3*num*sizeof(float));
   
  for(int i=0; i < TSENERGIES; energy[i++] = 0.0);
  timesteps=0;
}

void Timestep::get_transform_vectors(float A[3], float B[3], float C[3]) const
{
  // notes: a, b, c are side lengths of the unit cell
  // alpha = angle between b and c
  //  beta = angle between a and c
  // gamma = angle between a and b

  // convert from degrees to radians
  double cosBC = cos(DEGTORAD(alpha));
  double cosAC = cos(DEGTORAD(beta));
  double cosAB = cos(DEGTORAD(gamma));
  double sinAB = sin(DEGTORAD(gamma));

  // A will lie along the positive x axis.
  // B will lie in the x-y plane
  // The origin will be (0,0,0).
  float Ax = (float) (a_length);
  float Bx = (float) (b_length * cosAB);
  float By = (float) (b_length * sinAB);

  float Cx=0, Cy=0, Cz=0;
  // If sinAB is zero, then we can't determine C uniquely since it's defined
  // in terms of the angle between A and B.
  if (sinAB > 0) {
    Cx = (float) cosAC;
    Cy = (float) ((cosBC - cosAC * cosAB) / sinAB);
    Cz = sqrtf(1.0f - Cx*Cx - Cy*Cy);
  }
  Cx *= c_length;
  Cy *= c_length;
  Cz *= c_length;
  vec_zero(A); A[0] = Ax;
  vec_zero(B); B[0] = Bx; B[1] = By;
  vec_zero(C); C[0] = Cx; C[1] = Cy; C[2] = Cz;
}

void Timestep::get_transforms(Matrix4 &a, Matrix4 &b, Matrix4 &c) const {
  float A[3], B[3], C[3];
  get_transform_vectors(A, B, C);
  a.translate(A);
  b.translate(B);
  c.translate(C);
}

void Timestep::get_transform_from_cell(const int *cell, Matrix4 &mat) const {
  float A[3], B[3], C[3];
  get_transform_vectors(A, B, C);
  float Ax=A[0];
  float Bx=B[0];
  float By=B[1];
  float Cx=C[0];
  float Cy=C[1];
  float Cz=C[2];
  mat.identity();
  mat.mat[12] = cell[0]*Ax + cell[1]*Bx + cell[2]*Cx;
  mat.mat[13] =              cell[1]*By + cell[2]*Cy;            
  mat.mat[14] =                           cell[2]*Cz;
}

