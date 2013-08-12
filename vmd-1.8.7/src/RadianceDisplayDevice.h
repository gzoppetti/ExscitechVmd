#ifndef RADIANCEDISPLAYDEVICE_H
#define RADIANCEDISPLAYDEVICE_H
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
 *	$RCSfile: RadianceDisplayDevice.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.28 $	$Date: 2009/04/29 15:43:22 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Writes to the format for Radiance.  For more information about that
 * package, see http://radsite.lbl.gov/radiance/HOME.html .
 *
 ***************************************************************************/

#include <stdio.h>
#include "FileRenderer.h"

/// FileRenderer subclass to export VMD scenes to a Radiance scene file
class RadianceDisplayDevice : public FileRenderer {
private:
  /// manage the colors
  ResizeArray<float> red;
  ResizeArray<float> green;
  ResizeArray<float> blue;
  ResizeArray<float> trans;
  int cur_color;             ///< active colorID
  void reset_vars(void);     ///< reset internal state variables

protected:
  /// assorted graphics functions
  void point(float *);
  void sphere(float *);
  void line(float *, float *);
  void cylinder(float *, float *, float,int);
  void cone(float *, float *, float); 
  void cone(float *, float *, float, float); 
  void triangle(const float *, const float *, const float *,
                const float *, const float *, const float *);
  void square(float *, float *, float *, float *, float *);
  void comment(const char *);   
  void set_color(int); ///< set current colorID
   
public: 
  RadianceDisplayDevice();
  virtual ~RadianceDisplayDevice(void);
  void write_header(void);
  void write_trailer(void);
}; 

#endif

