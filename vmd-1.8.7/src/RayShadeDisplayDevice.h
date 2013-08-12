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
 *      $RCSfile: RayShadeDisplayDevice.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.24 $      $Date: 2009/04/29 15:43:22 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * FileRenderer type for the RayShade raytracer
 *
 ***************************************************************************/
#ifndef RAYSHADEDISPLAYDEVICE
#define RAYSHADEDISPLAYDEVICE

#include <stdio.h>
#include "FileRenderer.h"

/// FileRenderer subclass to exports VMD scenes to Rayshade scene format
class RayShadeDisplayDevice : public FileRenderer {
private:
  char *ray_filename;                     ///< output file name
  void write_cindexmaterial(int, int);    ///< write colors, materials etc.
  void write_colormaterial(float *, int); ///< write colors, materials etc.
  float scale_fix(float);                 /// fix scaling in a hackish manner
  
protected:
  /// assorted graphics functions
  void point(float *);
  void sphere(float *);
  void line(float *, float *);
  void cylinder(float *, float *, float,int filled);
  void cone(float *, float *, float); 
  void triangle(const float *, const float *, const float *,
                const float *, const float *, const float *);
  void comment(const char *);
   
public: 
  RayShadeDisplayDevice();
  virtual ~RayShadeDisplayDevice(void);
  void write_header(void);
  void write_trailer(void);
}; 

#endif

