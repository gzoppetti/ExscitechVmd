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
*      $RCSfile: RenderManDisplayDevice.h
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.19 $         $Date: 2009/04/29 15:43:22 $
*
***************************************************************************
* DESCRIPTION:
*
* FileRenderer type for the RenderMan interface.
*
***************************************************************************/

#ifndef RENDERMANDISPLAYDEVICE
#define RENDERMANDISPLAYDEVICE

#include <stdio.h>
#include "FileRenderer.h"

/// FileRenderer subclass to exports VMD scenes to Renderman RIB scene format
class RenderManDisplayDevice: public FileRenderer {
private:
  /// keep track of what the last written material properties are,
  /// that way we can avoid writing redundant definitions.
  float old_color[3];
  float old_opacity;
  float old_ambient;
  float old_diffuse;
  float old_specular;

  void reset_vars(void); ///< reset internal state variables
  void write_materials(int write_color);

protected:
  void point(float *xyz);
  void sphere(float *xyzr);
  void line(float *xyz1, float *xyz2);
  void cylinder(float *, float *, float rad, int filled);
  void cone(float * a, float * b, float rad); 
  void triangle(const float *, const float *, const float *,
                const float *, const float *, const float *);
  void tricolor(const float * xyz1, const float * xyz2, const float * xyz3,
                const float * n1,   const float * n2,   const float * n3,
                const float *c1,    const float *c2,    const float *c3);
  void square(float *, float *, float *, float *, float *);
  void comment(const char *);

public: 
  RenderManDisplayDevice(void);
  virtual ~RenderManDisplayDevice(void);
  void write_header(void); 
  void write_trailer(void);
}; 

#endif

