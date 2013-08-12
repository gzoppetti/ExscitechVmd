#ifndef ARTDISPLAYDEVICE_H
#define ARTDISPLAYDEVICE_H

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
 *	$RCSfile: ArtDisplayDevice.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.21 $	$Date: 2009/04/29 15:42:47 $
 *
 ***************************************************************************
 * DESCRIPTION: 
 *   Writes to the ART raytracer.  This is available from gondwana.ecr.mu.oz.au
 * as part of the vort package.
 *
 ***************************************************************************/


#include <stdio.h>
#include "FileRenderer.h"

/// FileRenderer subclass to export VMD scenes to ART ray tracer scene format
class ArtDisplayDevice : public FileRenderer {
private:
  char *art_filename; ///< output file name
  int Initialized;    ///< was the output file created?

protected:
  // assorted graphics functions
  void point(float *);
  void sphere(float *);
  void line(float *, float *);
  void cylinder(float *, float *, float,int filled);
  void cone(float *, float *, float); 
  void triangle(const float *, const float *, const float *,
                const float *, const float *, const float *);
  void square(float *, float *, float *, float *, float *);
  void comment(const char *);

public: 
  ArtDisplayDevice();
  virtual ~ArtDisplayDevice(void);
  void write_header(void);
  void write_trailer(void);
}; 

#endif

