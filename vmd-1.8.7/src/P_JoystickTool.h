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
 *	$RCSfile: P_JoystickTool.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.29 $	$Date: 2009/04/29 15:43:16 $
 *
 ***************************************************************************/

/// The 3d analog of a joystick.
/** This is a tool which creates the 3d analog of a joystick - pressing
    the button puts this tool into a relative mode, so it is kind of
    like flying a plane.  Force-feedback compatible. */

#include "P_Tool.h"
class JoystickTool : public Tool {
 public:
  JoystickTool(int id, VMDApp *, Displayable *);
  virtual void do_event();
  const float *position() const;

  const char *type_name() const { return "joystick"; }
 private:
  int constrained;
  float pos[3];
  float constraint[3];
};

