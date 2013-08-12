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
 *      $RCSfile: VolCPotential.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.4 $      $Date: 2009/04/29 15:43:31 $
 *
 ***************************************************************************/
#ifndef VOLCPOTENTIAL_H
#define VOLCPOTENTIAL_H

int vol_cpotential(long int natoms, float* atoms, float* grideners, long int numplane, long int numcol, long int numpt, float gridspacing); 

#endif
