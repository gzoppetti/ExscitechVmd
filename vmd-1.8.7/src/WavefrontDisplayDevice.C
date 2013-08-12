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
 *	$RCSfile: WavefrontDisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.10 $	$Date: 2009/04/29 15:43:33 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 *   Render to a Wavefront Object or "OBJ" file.
 *   This file format is one of the most universally supported 3-D model
 *   file formats, particular for animation software such as Maya, 
 *   3-D Studio, etc.  The file format is simple, but good enough for getting
 *   a lot of things done.
 *
 ***************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "WavefrontDisplayDevice.h"
#include "Matrix4.h"
#include "DispCmds.h"
#include "Inform.h"
#include "utilities.h"
#include "VMDDisplayList.h"

#define DASH_LENGTH 0.02

// constructor ... call the parent with the right values
WavefrontDisplayDevice::WavefrontDisplayDevice(void) 
: FileRenderer("Wavefront", "vmd.obj", "true") { }

// destructor
WavefrontDisplayDevice::~WavefrontDisplayDevice(void) { }

// draw a point
void WavefrontDisplayDevice::point(float * spdata) {
  float vec[3];
  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);

  // draw the sphere
  fprintf(outfile, "v %5f %5f %5f\n", vec[0], vec[1], -vec[2]);
  fprintf(outfile, "p -1\n");
}

// draw a line from a to b
void WavefrontDisplayDevice::line(float *a, float*b) {
  int i, j, test;
  float dirvec[3], unitdirvec[3];
  float from[3], to[3], tmp1[3], tmp2[3];
  float len;
   
  if (lineStyle == ::SOLIDLINE) {
    // transform the world coordinates
    (transMat.top()).multpoint3d(a, from);
    (transMat.top()).multpoint3d(b, to);

    // draw the solid line
    fprintf(outfile, "v %5f %5f %5f\n", from[0], from[1], -from[2]);
    fprintf(outfile, "v %5f %5f %5f\n", to[0], to[1], -to[2]);
    fprintf(outfile, "l -1 -2\n");
  } else if (lineStyle == ::DASHEDLINE ) {
     // transform the world coordinates
    (transMat.top()).multpoint3d(a, tmp1);
    (transMat.top()).multpoint3d(b, tmp2);

    // how to create a dashed line
    for(i=0; i<3; i++) {
      dirvec[i] = tmp2[i] - tmp1[i];  // vector from a to b
    }
    len = sqrtf(dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2])
;
    for(i=0;i<3;i++) {
      unitdirvec[i] = dirvec[i] / sqrtf(len); // unit vector from a to b
    }
          
    test = 1;
    i = 0;
    while(test == 1) {
      for(j=0;j<3;j++) {
        from[j] = (float) (tmp1[j] + (2*i    )*DASH_LENGTH*unitdirvec[j]);
          to[j] = (float) (tmp1[j] + (2*i + 1)*DASH_LENGTH*unitdirvec[j]);
      }

      if (fabsf(tmp1[0] - to[0]) >= fabsf(dirvec[0])) {
        for(j=0;j<3;j++)
          to[j] = tmp2[j];
        test = 0;
      }

      // draw the solid line dash
      fprintf(outfile, "v %5f %5f %5f\n", from[0], from[1], -from[2]);
      fprintf(outfile, "v %5f %5f %5f\n", to[0], to[1], -to[2]);
      fprintf(outfile, "l -1 -2\n");
      i++;
    }
  } else {
    msgErr << "WavefrontDisplayDevice: Unknown line style "
           << lineStyle << sendmsg;
  }
}




void WavefrontDisplayDevice::triangle(const float *v1, const float *v2, const float *v3, 
                                      const float *n1, const float *n2, const float *n3) {
  float a[3], b[3], c[3];
  float norm1[3], norm2[3], norm3[3];

  // transform the world coordinates
  (transMat.top()).multpoint3d(v1, a);
  (transMat.top()).multpoint3d(v2, b);
  (transMat.top()).multpoint3d(v3, c);

  // and the normals
  (transMat.top()).multnorm3d(n1, norm1);
  (transMat.top()).multnorm3d(n2, norm2);
  (transMat.top()).multnorm3d(n3, norm3);
                                                       
  // draw the triangle 
  fprintf(outfile,"v %f %f %f\n", a[0], a[1], a[2]);
  fprintf(outfile,"v %f %f %f\n", b[0], b[1], b[2]);
  fprintf(outfile,"v %f %f %f\n", c[0], c[1], c[2]);
  fprintf(outfile,"vn %f %f %f\n", norm1[0], norm1[1], norm1[2]);
  fprintf(outfile,"vn %f %f %f\n", norm2[0], norm2[1], norm2[2]);
  fprintf(outfile,"vn %f %f %f\n", norm3[0], norm3[1], norm3[2]);
  fprintf(outfile,"f -3//-3 -2//-2 -1//-1\n");
}

void WavefrontDisplayDevice::write_header (void) {
  fprintf (outfile, "# Wavefront OBJ file export by VMD\n");
}

void WavefrontDisplayDevice::write_trailer (void) {
  msgWarn << "Materials and colors are not exported to Wavefront files.\n";
}

