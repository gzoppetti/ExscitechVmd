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
 *	$RCSfile: P_FreeVRTracker.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.14 $	$Date: 2009/04/29 15:43:15 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 * -dispdev freevr
 ***************************************************************************/

#include "Matrix4.h"
#include "P_Tracker.h"
#include "P_FreeVRTracker.h"
#include <freevr.h>

void FreeVRTracker::update() {

  #define WAND_SENSOR     1
  vrPoint wand_location;
  // old FreeVR api
  //vrGet6sensorLocationRW(WAND_SENSOR, &wand_location);   
  // CAVE version
  // GetWand(pos[0], pos[1], pos[2]);
  vrPointGetRWFrom6sensor(&wand_location, WAND_SENSOR);
  pos[0] = wand_location.v[0];
  pos[1] = wand_location.v[1];
  pos[2] = wand_location.v[2];

  /* "classical" Euler angles */
  float azi, elev, roll;

  // XXX hack to get us by for now until FreeVR can do this, or 
  // something like this.
  azi=0.0;  
  elev=0.0;
  roll=0.0;
  // CAVE version
  // CAVEGetWandOrientation(azi, elev, roll);

  orient->identity();
  orient->rot(azi,'y');
  orient->rot(elev,'x');
  orient->rot(roll,'z');
  orient->rot(90,'y'); // to face forward (-z)
}

int FreeVRTracker::do_start(const SensorConfig *config) {
  // Must check that we are actually running in FreeVR here; if not, 
  // return 0.
  if (!config->require_freevr_name()) return 0;
  if (!config->have_one_sensor()) return 0;
  return 1;
}
