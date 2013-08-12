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
 *	$RCSfile: Stage.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.33 $	$Date: 2009/04/29 15:43:25 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   A Stage is a displayable object that acts as floor for other objects.
 * It is intended to be a point of reference.
 *
 ***************************************************************************/
#ifndef STAGE_H
#define STAGE_H

#include "Displayable.h"
#include "DispCmds.h"

#define STAGE_PANELS	8

/// Displayable subclass implementing a checkered "stage"
class Stage : public Displayable {
public:
  /// enumerated locations for the stage
  enum StagePos { NO_STAGE = 0, STAGE_ORIGIN, STAGE_LOWER, STAGE_UPPER,
  	STAGE_LEFT, STAGE_RIGHT, STAGE_BEHIND, STAGEPOS_TOTAL };

private:
  // corners defining the stage
  float corner1[3], corner2[3], corner3[3], corner4[3];
  float xw, zw;
  int usecolors[2];

  /// number of panels each side is divided into, and width of panel
  int Panels;
  float inc;

  /// current stage position
  int stagePos;
  
  /// do we need an update
  int need_update;

  /// useful display command objects
  DispCmdColorIndex cmdColor;
  DispCmdSquare cmdSquare;

  /// color category index with the colors to use.  If < 0, use default colors
  int colorCat;

  /// regenerate the command listo
  void create_cmdlist(void);

protected:
  virtual void do_color_changed(int);

public:
  /// constructor: the parent displayable 
  Stage(Displayable *);

  /// set stage display mode; return success
  int location(int);

  /// return stage display mode
  int location(void) { return stagePos; }

  /// return descripton of location
  char *loc_description(int);

  /// return total number of locations
  int locations(void) { return STAGEPOS_TOTAL; }

  /// get/set number of panels (must be >= 1)
  int panels(void) { return Panels; }
  int panels(int newp) {
    if (newp == Panels) return TRUE;
    if(newp >= 1 && newp <= 30) {
      Panels = newp;
      inc = 2.0f / (float) Panels;
      need_update = TRUE;
      return TRUE; // success
    }
    return FALSE; // failed
  }

  //
  // public virtual routines
  //
  
  /// prepare for drawing ... do any updates needed right before draw.
  virtual void prepare();
};

#endif

