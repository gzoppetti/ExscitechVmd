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
 *	$RCSfile: P_FreeVRButtons.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.14 $	$Date: 2009/04/29 15:43:15 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 * This is a Buttons that gets its info from the FreeVR wand.
 ***************************************************************************/

#include "P_Buttons.h"

/// Buttons subclass that gets its info from the FreeVR wand.
class FreeVRButtons : public Buttons {
public:
  FreeVRButtons() {}
  virtual const char *device_name() const { return "freevrbuttons"; }
  virtual Buttons *clone() { return new FreeVRButtons; }
  virtual void update();
  inline virtual int alive() { return 1; }

protected:
  /// Check that we are running in a FreeVR environment.
  virtual int do_start(const SensorConfig *);
};

