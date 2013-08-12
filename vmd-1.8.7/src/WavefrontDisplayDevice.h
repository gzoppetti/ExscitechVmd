/***************************************************************************
 *cr
 *cr		(C) Copyright 1995-2009 The Board of Trustees of the
 *cr			    University of Illinois
 *cr			     All Rights Reserved
 *cr
 ***************************************************************************/
/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile: WavefrontDisplayDevice.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.6 $	$Date: 2009/04/29 15:43:33 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Use to make Wavefront "OBJ" files for importing into numerous animation
 *   systems.
 *
 ***************************************************************************/

#ifndef WavefrontDISPLAYDEVICE
#define WavefrontDISPLAYDEVICE

#include <stdio.h>
#include "FileRenderer.h"

/// FileRenderer subclass to export VMD scenes to Wavefront "OBJ" format
class WavefrontDisplayDevice : public FileRenderer {
protected:
  void point(float *xyz);
  void line(float *xyz1, float *xyz2);
  void triangle(const float *, const float *, const float *,
                const float *, const float *, const float *);

public:
  WavefrontDisplayDevice(void);            // constructor
  virtual ~WavefrontDisplayDevice(void);   // destructor
  void write_header (void);
  void write_trailer(void);
};

#endif

