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
 *	$RCSfile: R3dDisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.77 $	$Date: 2009/05/04 22:22:29 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * The R3dDisplayDevice implements routines needed to render to a file 
 * in raster3d format
 *
 ***************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#define sqr(x) ((x) * (x))

#include "R3dDisplayDevice.h"
#include "Matrix4.h"
#include "DispCmds.h"
#include "Inform.h"
#include "utilities.h"

#define DEFAULT_RADIUS 0.002 // radius for faking lines with cylinders
#define DASH_LENGTH 0.02     // dash lengths

#define currentColor matData[colorIndex]

///////////////////////// constructor and destructor

// constructor ... initialize some variables
// Raster3D README suggests this as the default
// It assumes you have 'display' from the ImageMagick tools.
static char standard_r3d[] = " render < %s | display avs:-";

static char raster3d_filename[] = "plot.r3d";

R3dDisplayDevice::R3dDisplayDevice(void) : 
  FileRenderer((char *) "Raster3D", raster3d_filename, standard_r3d) {
  reset_vars(); // initialize internal state
}
               
//destructor
R3dDisplayDevice::~R3dDisplayDevice(void) { }

void R3dDisplayDevice::reset_vars(void) {
  // Object decl's won't be legal until the header is out.
  objLegal = 0;
  mat_on = 0;
  old_mat_shininess = -1;
  old_mat_specular = -1;
  old_mat_opacity = -1;
}


///////////////////////// protected nonvirtual routines

// draw a point
void R3dDisplayDevice::point(float * spdata) {
  float vec[3];

  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);

  write_materials();

  // draw the sphere
  fprintf(outfile, "2\n");  // sphere
  fprintf(outfile, "%7f %7f %7f ", vec[0], vec[1], vec[2]); // center of sphere
  fprintf(outfile, "%7f ", float(lineWidth)*DEFAULT_RADIUS ); // the radius of the sphere
  fprintf(outfile, "%3.2f %3.2f %3.2f\n", sqr(currentColor[0]), 
	  sqr(currentColor[1]),  sqr(currentColor[2]));
}

// draw a sphere
void R3dDisplayDevice::sphere(float * spdata) {
  float vec[3];
  float radius;
    
  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);
  radius = scale_radius(spdata[3]);
  
  // write out the current material properties
  write_materials();
 
  // draw the sphere
  fprintf(outfile, "2\n");  // sphere
  fprintf(outfile, "%7f %7f %7f ", vec[0], vec[1], vec[2]); // center of sphere
  fprintf(outfile, "%7f ", radius); // the radius of the sphere
  fprintf(outfile, "%3.2f %3.2f %3.2f\n", sqr(currentColor[0]), 
	  sqr(currentColor[1]), sqr(currentColor[2]));
}


// draw a line (cylinder) from a to b
void R3dDisplayDevice::line(float *a, float*b) {
    int i, j, test;
    float dirvec[3], unitdirvec[3];
    float from[3], to[3], tmp1[3], tmp2[3];
    float len;

    if(lineStyle == ::SOLIDLINE ) {
  
        // transform the world coordinates
        (transMat.top()).multpoint3d(a, from);
        (transMat.top()).multpoint3d(b, to);

        // draw the cylinder
        fprintf(outfile, "5\n"); // flat-ended cylinder
        fprintf(outfile, "%7f %7f %7f ", from[0], from[1], from[2]); // first point
        fprintf(outfile, "%7f ", float(lineWidth)*DEFAULT_RADIUS); // radius 1
        fprintf(outfile, "%7f %7f %7f ", to[0], to[1], to[2]); // second point
        fprintf(outfile, "%7f ", float(lineWidth)*DEFAULT_RADIUS); // radius 2
        fprintf(outfile, "%3.2f %3.2f %3.2f\n", sqr(currentColor[0]), 
		sqr(currentColor[1]), sqr(currentColor[2]));

    } else if (lineStyle == ::DASHEDLINE ) {
        
         // transform the world coordinates
        (transMat.top()).multpoint3d(a, tmp1);
        (transMat.top()).multpoint3d(b, tmp2);

        // how to create a dashed line
        for(i=0;i<3;i++) {
            dirvec[i] = tmp2[i] - tmp1[i];  // vector from a to b
        }
        len = sqrtf( dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2] );
        for(i=0;i<3;i++) {
            unitdirvec[i] = dirvec[i] / sqrtf(len);  // unit vector pointing from a to b
        }
           
        test = 1;
        i = 0;

        while( test == 1 ) {
            for(j=0;j<3;j++) {
              from[j] = (float) (tmp1[j] + (2*i)*DASH_LENGTH*unitdirvec[j]);
              to[j] =   (float) (tmp1[j] + (2*i + 1)*DASH_LENGTH*unitdirvec[j]);
            }
            if( fabsf(tmp1[0] - to[0]) >= fabsf(dirvec[0]) ) {
                for(j=0;j<3;j++) {
                    to[j] = tmp2[j];
                }
                test = 0;
            }

            // draw the cylinder
            fprintf(outfile, "5\n"); // flat-ended cylinder
            fprintf(outfile, "%7f %7f %7f ", from[0], from[1], from[2]); // first point
            fprintf(outfile, "%7f ", float(lineWidth)*DEFAULT_RADIUS); // radius 1
            fprintf(outfile, "%7f %7f %7f ", to[0], to[1], to[2]); // second point
            fprintf(outfile, "%7f ", float(lineWidth)*DEFAULT_RADIUS); // radius 2
            fprintf(outfile, "%3.2f %3.2f %3.2f\n", sqr(currentColor[0]), 
		    sqr(currentColor[1]), sqr(currentColor[2]));

            i++;
        }

    } else {
        msgErr << "R3dDisplayDevice: Unknown line style " << lineStyle << sendmsg;
    }

}

// draw a cylinder
void R3dDisplayDevice::cylinder(float *a, float *b, float r, int) {

  float vec1[3], vec2[3];
  float radius;
  
  // transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  radius = scale_radius(r);

  write_materials();

  //// draw the cylinder
  // ignore the 'filled' flag
  fprintf(outfile, "5\n"); // flat-ended cylinder
  fprintf(outfile, "%7f %7f %7f ", vec1[0], vec1[1], vec1[2]); // first point
  fprintf(outfile, "%7f ", radius); // radius 1
  fprintf(outfile, "%7f %7f %7f ", vec2[0], vec2[1], vec2[2]); // second point
  fprintf(outfile, "%7f ", radius); // radius 2
  fprintf(outfile, "%3.2f %3.2f %3.2f\n", sqr(currentColor[0]), 
	  sqr(currentColor[1]),  sqr(currentColor[2]));

}

// draw a triangle
void R3dDisplayDevice::triangle(const float *a, const float *b, const float *c, const float *n1, const float *n2, const float *n3) {

  float vec1[3], vec2[3], vec3[3];
  float norm1[3], norm2[3], norm3[3];
  
  // transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  (transMat.top()).multpoint3d(c, vec3);

  // transform the normals
  (transMat.top()).multnorm3d(n1, norm1);
  (transMat.top()).multnorm3d(n2, norm2);
  (transMat.top()).multnorm3d(n3, norm3);

  write_materials();

  // draw the triangle
  fprintf(outfile, "1\n"); // triangle
  fprintf(outfile, "%7f %7f %7f ", vec1[0], vec1[1], vec1[2]); 
  fprintf(outfile, "%7f %7f %7f ", vec2[0], vec2[1], vec2[2]); 
  fprintf(outfile, "%7f %7f %7f ", vec3[0], vec3[1], vec3[2]);
  fprintf(outfile, "%3.2f %3.2f %3.2f\n", sqr(currentColor[0]), 
	  sqr(currentColor[1]),  sqr(currentColor[2]));

  fprintf(outfile, "7\n"); // triangle normals
  fprintf(outfile, "%7f %7f %7f ",  norm1[0], norm1[1], norm1[2]);
  fprintf(outfile, "%7f %7f %7f ",  norm2[0], norm2[1], norm2[2]);
  fprintf(outfile, "%7f %7f %7f\n", norm3[0], norm3[1], norm3[2]);
}

// draw a three-color triangle
void R3dDisplayDevice::tricolor(const float *a, const float *b, const float *c, const float *n1, const float *n2, const float *n3, const float *c1, const float *c2, const float *c3) {

  float vec1[3], vec2[3], vec3[3];
  float norm1[3], norm2[3], norm3[3];

  // transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  (transMat.top()).multpoint3d(c, vec3);

  // transform the normals
  (transMat.top()).multnorm3d(n1, norm1);
  (transMat.top()).multnorm3d(n2, norm2);
  (transMat.top()).multnorm3d(n3, norm3);

  write_materials();

  // draw the triangle
  fprintf(outfile, "1\n");
  fprintf(outfile, "%7f %7f %7f ", vec1[0], vec1[1], vec1[2]);
  fprintf(outfile, "%7f %7f %7f ", vec2[0], vec2[1], vec2[2]);
  fprintf(outfile, "%7f %7f %7f ", vec3[0], vec3[1], vec3[2]);
  fprintf(outfile, "%3.2f %3.2f %3.2f\n", sqr(currentColor[0]),
          sqr(currentColor[1]), sqr(currentColor[2]));

  fprintf(outfile, "7\n"); // triangle normals
  fprintf(outfile, "%7f %7f %7f ",  norm1[0], norm1[1], norm1[2]);
  fprintf(outfile, "%7f %7f %7f ",  norm2[0], norm2[1], norm2[2]);
  fprintf(outfile, "%7f %7f %7f\n", norm3[0], norm3[1], norm3[2]);

  // now the colors at the three vertices
  fprintf(outfile, "17\n");
  fprintf(outfile, "%7f %7f %7f %7f %7f %7f %7f %7f %7f\n",
          c1[0], c1[1], c1[2], c2[0], c2[1], c2[2], c3[0], c3[1], c3[2]);
}

void R3dDisplayDevice::comment(const char *s) {
  int i=0, length;
  char buf[71];
  const char *index;

  if (!objLegal) return;

  length = strlen(s);
  index = s;

  while (i*70 < length) {
    strncpy(buf, index, 70);
    buf[70] = '\0';
    fprintf (outfile, "# %s\n", buf);
    index += 70;
    i++;
  }
}

///////////////////// public virtual routines

// initialize the file for output
void R3dDisplayDevice::write_header() {
    int tileX, tileY;
    int nTilesX, nTilesY;

    int i, nlights;
    float lightshare;
    float scale;

    fprintf(outfile, "r3d input script\n");

    // Raster3D does not allow you to specify an exact image size. Instead,
    // you specify a number of square tiles (maximum of 192) and then specify
    // a resolution (in pixels) per tile (maximum of 36). We want to choose
    // the tile size as small as possible and use as many tiles as possible
    // so that we get the best approximation of VMD's actual screen size.
    //
    // This is slightly complicated by the fact that due to antialiasing, the
    // tile size must be divisible by 3.

    tileX = 2;
    while (xSize / tileX > 192) {
        tileX += 2;
        if (tileX > 36) {
            tileX -= 2;
            msgInfo << "Warning: The Raster3D output image has too high a resolution" << sendmsg;
            msgInfo << "to be properly rendered. Writing the file anyway, but Raster3D" << sendmsg;
            msgInfo << "will probably give an error." << sendmsg;
            break;
        }
    }

    tileY = 2;
    while (ySize / tileY > 192) {
        tileY += 2;
        if (tileY > 36) {
            tileY -= 2;
            if (xSize / tileX > 192) {
                msgInfo << "Warning: The Raster3D output image has too high a resolution" << sendmsg;
                msgInfo << "to be properly rendered. Writing the file anyway, but Raster3D" << sendmsg;
                msgInfo << "will probably give an error." << sendmsg;
            }
            break;
        }
    }

    // Now that we've chosen a value for the tile size, we choose the number
    // of tiles to match, as closely as possible, VMD's screen size.

    nTilesX = xSize / tileX;
    nTilesY = ySize / tileY;
    if (xSize % tileX >= tileX / 2) nTilesX++;
    if (ySize % tileY >= tileY / 2) nTilesY++;

    fprintf(outfile, "%d %d          tiles in x,y\n", nTilesX, nTilesY);
    fprintf(outfile, "%d %d          computing pixels per tile\n", tileX, tileY);
    fprintf(outfile, "4              alti-aliasing scheme 4; 3x3 -> 2x2\n");
    fprintf(outfile, "%3.2f %3.2f %3.2f background color\n", 
            backColor[0], backColor[1], backColor[2]);
    fprintf(outfile, "T              shadows on\n");
    fprintf(outfile, "20             Phong power\n");
    fprintf(outfile, "1.00           secondary light contribution\n");
    fprintf(outfile, "0.10           ambient light contribution\n");
    fprintf(outfile, "0.50           specular reflection component\n");

    // Raster3D only allows us to tell it the ratio between the image's narrower
    // dimension and the distance from the eye to the viewing plane (in world
    // coordinates). Oh well.

    switch (projection()) {

        case DisplayDevice::ORTHOGRAPHIC:
            fprintf(outfile, "0              Eye position (orthographic mode)\n");
            break;

        case DisplayDevice::PERSPECTIVE:
        default:
            if (Aspect > 1) fprintf(outfile, "%6.2f         Eye position\n",
                (-zDist + eyePos[2]) / vSize);
            else fprintf(outfile, "%6.2f         Eye position\n",
                (-zDist + eyePos[2]) / vSize / Aspect);
            break;

    }

    // All light sources defined as Raster3d 2.3+ glow lights, not
    // in the header...
    fprintf(outfile, "1 0 0          main light source position\n");

    // We need to compute a scaling factor for the Raster3D objects. Depending on
    // whether our image is wider than it is tall, we give Raster3D either the
    // horizontal scaling factor or the vertical scaling factor (Raster3D doesn't
    // allow us to specify both).

    if (Aspect > 1) scale = vSize / 2;
    else scale = vSize * Aspect / 2;

    // Global transformation matrix for objects.
    fprintf(outfile, "1 0 0 0        global xform matrix\n");
    fprintf(outfile, "0 1 0 0\n");
    fprintf(outfile, "0 0 1 0\n");
    fprintf(outfile, "0 0 0 %.3f\n", scale);

    fprintf(outfile, "3\n");
    fprintf(outfile, "*\n*\n*\n");

    // Define additional light sources, if any. Raster3d uses a light-
    // sharing model; all light sources affect a percentage of the total
    // lighting system. This percentage must be determined in advance.
    nlights = 0;
    for (i = 0; i < DISP_LIGHTS; i++)
        if (lightState[i].on) nlights++;

    // Must use ?: operator to avoid divide by zero
    lightshare = nlights ? (1 / (float) nlights) : 0;

    // Now output all of the lights.
    for (i = 0; i < DISP_LIGHTS; i++) {
        if (lightState[i].on) {
            fprintf(outfile, "13\n%f %f %f 100 %f 0 20 1 1 1\n",
                    lightState[i].pos[0], lightState[i].pos[1], lightState[i].pos[2],
                    lightshare);
        }
    }
 
    // and that's it for the header.  next comes free format 
    // triangle, sphere, and cylinder descriptors
    objLegal = 1;
}

void R3dDisplayDevice::write_trailer(void) {
  // if we need to, turn material properties off
  close_materials();

  msgInfo << "Raster3D file generation finished" << sendmsg;

  reset_vars(); // reset internal state
}

// Writes out the current material properties as a
// material modifier (object type 8)
void R3dDisplayDevice::write_materials(void) {
   // Format of material definitions:
   //
   // 8
   // MPHONG MSPEC SR,SG,SB CLRITY OPTS(4)
   //    where MPHONG   - phong parameter for specular highlighting
   //          MSPEC    - specular scattering contribution
   //          SR,SG,SB - color of reflected light
   //          CLRITY   - opacity, 0.0=opaque, 1.0=transparent
   //          OPTS(4)  - zero, except for OPT(1), which controls
   //                     rendering of self-occluding objects
   // 9

   if (!mat_on) {
     fprintf(outfile, "8\n");
     //  Raster3D cannot tolerate inconsistent surface normals, so we
     //  have to tell it to flip them for itself.  This format string
     //  tell is to do that for us.  This is also necessary for things
     //  like surfaces which are inherently "two-sided", particularly
     //  if/when clipping planes are used.
#if 1
     // auto-flip normals 
     fprintf(outfile, "%.3f %.3f 1 1 1 %.3f 2 0 0 0\n",
#else
     // don't auto-flip normals 
     fprintf(outfile, "%.3f %.3f 1 1 1 %.3f 0 0 0 0\n",
#endif
        mat_shininess, mat_specular, 1 - mat_opacity);

     old_mat_shininess = mat_shininess;
     old_mat_specular = mat_specular;
     old_mat_opacity = mat_opacity;

     mat_on = 1;
   }
   else if (mat_shininess != old_mat_shininess ||
            mat_specular != old_mat_specular ||
            mat_opacity != old_mat_opacity) {
     fprintf(outfile, "9\n");
     fprintf(outfile, "8\n");
     fprintf(outfile, "%.3f %.3f 1 1 1 %.3f 0 0 0 0\n",
        mat_shininess, mat_specular, 1 - mat_opacity);

     old_mat_shininess = mat_shininess;
     old_mat_specular = mat_specular;
     old_mat_opacity = mat_opacity;
   }

   return;
}

void R3dDisplayDevice::close_materials(void) {
  if (mat_on) {
    fprintf(outfile, "9\n");
    mat_on = 0;
  }
  return;
}
