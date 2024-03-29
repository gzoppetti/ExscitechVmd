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
 *	$RCSfile: Vrml2DisplayDevice.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.16 $	$Date: 2009/04/29 15:43:32 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   VRML2 / VRML97 scene export code
 ***************************************************************************/

#ifndef VRML2DISPLAYDEVICE_H
#define VRML2DISPLAYDEVICE_H

#include <stdio.h>
#include "FileRenderer.h"

/// FileRenderer subclass to export VMD scenes to VRML2/VRML97 scene format
class Vrml2DisplayDevice : public FileRenderer {
private:
  struct triList {
    triList *next;
    int ptz[3];
  };
  triList *tList;
  void write_cindexmaterial(int, int); // write colors, materials etc.
  void write_colormaterial(float *, int); // write colors, materials etc.

protected:
  // assorted graphics functions
  void sphere(float *xyzr);
  void cylinder(float *a, float *b, float rad, int filled);
  void cone    (float *a, float *b, float rad);
  void line(float *xyz1, float *xyz2);
  void triangle(const float *, const float *, const float *,
                const float *, const float *, const float *);
  void tricolor(const float * xyz1, const float * xyz2, const float * xyz3,
                const float * n1,   const float * n2,   const float * n3,
                const float *c1,    const float *c2,    const float *c3);
  void comment(const char *);
  void load(const Matrix4& mat);       ///< load transofrmation matrix
  void multmatrix(const Matrix4& mat); ///< concatenate transformation
  void set_color(int color_index);     ///< set the colorID

public:
  Vrml2DisplayDevice(void);
  void write_header(void);
  void write_trailer(void);
}; 

#endif



