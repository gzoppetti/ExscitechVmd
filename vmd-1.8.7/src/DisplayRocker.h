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
 *	$RCSfile: DisplayRocker.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.6 $	$Date: 2009/04/29 15:42:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************/
#ifndef DISPLAYROCKER_H
#define DISPLAYROCKER_H

#include "Displayable.h"

class DisplayRocker : public Displayable {
private:
  /// are we rocking the objects back and forth?  If so, what, where, and how?
  int Rocking, rockSteps, currRockStep, rockOnce;
  char rockAxis;
  float rockAmount;

public:
  DisplayRocker(Displayable *);   ///< constructor

  void start_rocking(float a, char ax, int steps, int doOnce = FALSE);
  void stop_rocking()   { Rocking = FALSE; }  

  virtual void prepare();
};

#endif

