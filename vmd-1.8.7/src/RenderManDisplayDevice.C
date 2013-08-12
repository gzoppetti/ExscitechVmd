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
*      $RCSfile: RenderManDisplayDevice.C
*      $Author: johns $      $Locker:  $               $State: Exp $
*      $Revision: 1.40 $         $Date: 2009/04/29 15:43:22 $
*
***************************************************************************
* DESCRIPTION:
*
* FileRenderer type for the RenderMan interface.
*
***************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "RenderManDisplayDevice.h"

// The default radius for points and lines (which are displayed
// as small spheres or cylinders, respectively)
#define DEFAULT_RADIUS  0.0025

/// constructor ... initialize some variables
RenderManDisplayDevice::RenderManDisplayDevice() 
: FileRenderer("RenderMan","plot.rib", "rendrib -d 8 %s") {
  reset_vars(); // initialize material cache
}
        
/// destructor
RenderManDisplayDevice::~RenderManDisplayDevice(void) { }


/// (re)initialize cached state variables used to track material changes 
void RenderManDisplayDevice::reset_vars(void) {
  old_color[0] = -1;
  old_color[1] = -1;
  old_color[2] = -1;
  old_ambient = -1;
  old_specular = -1;
  old_opacity = -1;
  old_diffuse = -1;
}


/// draw a point
void RenderManDisplayDevice::point(float * spdata) {
  float vec[3];
  // Transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);

  fprintf(outfile, "TransformBegin\n");
  write_materials(1);
  fprintf(outfile, "  Translate %g %g %g\n", vec[0], vec[1], vec[2]);
  fprintf(outfile, "  Sphere %g %g %g 360\n",
    (float)  lineWidth * DEFAULT_RADIUS,
    (float) -lineWidth * DEFAULT_RADIUS,
    (float)  lineWidth * DEFAULT_RADIUS);
  fprintf(outfile, "TransformEnd\n");
}


/// draw a sphere
void RenderManDisplayDevice::sphere(float * spdata) {
  float vec[3];
  float radius;

  // Transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);
  radius = scale_radius(spdata[3]);
  if (radius < DEFAULT_RADIUS) {
    radius = (float) DEFAULT_RADIUS;
  }

  // Draw the sphere
  fprintf(outfile, "TransformBegin\n");
  write_materials(1);
  fprintf(outfile, "  Translate %g %g %g\n", vec[0], vec[1], vec[2]);
  fprintf(outfile, "  Sphere %g %g %g 360\n", radius, -radius, radius);
  fprintf(outfile, "TransformEnd\n");
}


/// draw a line (cylinder) from a to b
void RenderManDisplayDevice::line(float *a, float *b) {
  cylinder(a, b, (float) (lineWidth * DEFAULT_RADIUS), 0);
}


/// draw a cylinder
void RenderManDisplayDevice::cylinder(float *a, float *b, float r, 
                              int /* filled */ ) {
  float axis[3], vec1[3], vec2[3];
  float R, phi, rxy, theta;
  float radius;

  // Transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  radius = scale_radius(r);
  if (radius < DEFAULT_RADIUS) {
    radius = (float) DEFAULT_RADIUS;
  }

  // RenderMan's cylinders always run along the z axis, and must
  // be transformed to the proper position and rotation. This
  // code is taken from OpenGLRenderer.C.
  axis[0] = vec2[0] - vec1[0];
  axis[1] = vec2[1] - vec1[1];
  axis[2] = vec2[2] - vec1[2];

  R = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
  if (R <= 0) return;

  R = sqrtf(R); // evaluation of sqrt() _after_ early exit

  // determine phi rotation angle, amount to rotate about y
  phi = acosf(axis[2] / R);

  // determine theta rotation, amount to rotate about z
  rxy = sqrtf(axis[0] * axis[0] + axis[1] * axis[1]);
  if (rxy <= 0) {
    theta = 0;
  } else {
    theta = acosf(axis[0] / rxy);
    if (axis[1] < 0) theta = (float) (2.0 * VMD_PI) - theta;
  }

  // Write the cylinder
  fprintf(outfile, "TransformBegin\n");
  write_materials(1);
  fprintf(outfile, "  Translate %g %g %g\n", vec1[0], vec1[1], vec1[2]);
  if (theta) 
    fprintf(outfile, "  Rotate %g 0 0 1\n", (theta / VMD_PI) * 180);
  if (phi) 
    fprintf(outfile, "  Rotate %g 0 1 0\n", (phi / VMD_PI) * 180);
  fprintf(outfile, "  Cylinder %g 0 %g 360\n", radius, R);
  fprintf(outfile, "TransformEnd\n");
}


/// draw a cone
void RenderManDisplayDevice::cone(float *a, float *b, float r) {
  float axis[3], vec1[3], vec2[3];
  float R, phi, rxy, theta;
  float radius;

  // Transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  radius = scale_radius(r);
  if (radius < DEFAULT_RADIUS) {
    radius = (float) DEFAULT_RADIUS;
  }

  // RenderMan's cylinders always run along the z axis, and must
  // be transformed to the proper position and rotation. This
  // code is taken from OpenGLRenderer.C.
  axis[0] = vec2[0] - vec1[0];
  axis[1] = vec2[1] - vec1[1];
  axis[2] = vec2[2] - vec1[2];

  R = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
  if (R <= 0) return;

  R = sqrtf(R); // evaluation of sqrt() _after_ early exit

  // determine phi rotation angle, amount to rotate about y
  phi = acosf(axis[2] / R);

  // determine theta rotation, amount to rotate about z
  rxy = sqrtf(axis[0] * axis[0] + axis[1] * axis[1]);
  if (rxy <= 0) {
    theta = 0;
  } else {
    theta = acosf(axis[0] / rxy);
    if (axis[1] < 0) theta = (float) (2.0 * VMD_PI) - theta;
  }

  // Write the cone
  fprintf(outfile, "TransformBegin\n");
  write_materials(1);
  fprintf(outfile, "  Translate %g %g %g\n", vec1[0], vec1[1], vec1[2]);
  if (theta) 
    fprintf(outfile, "  Rotate %g 0 0 1\n", (theta / VMD_PI) * 180);
  if (phi) 
    fprintf(outfile, "  Rotate %g 0 1 0\n", (phi / VMD_PI) * 180);
  fprintf(outfile, "  Cone %g %g 360\n", R, radius);
  fprintf(outfile, "TransformEnd\n");
}


// draw a triangle
void RenderManDisplayDevice::triangle(const float *a, const float *b, const float *c, const float *n1, const float *n2, const float *n3) {
  float vec1[3], vec2[3], vec3[3];
  float norm1[3], norm2[3], norm3[3];

  // Transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  (transMat.top()).multpoint3d(c, vec3);
  (transMat.top()).multnorm3d(n1, norm1);
  (transMat.top()).multnorm3d(n2, norm2);
  (transMat.top()).multnorm3d(n3, norm3);

  // Write the triangle
  write_materials(1);
  fprintf(outfile, "Polygon \"P\" [ %g %g %g %g %g %g %g %g %g ] ",
          vec1[0], vec1[1], vec1[2],
          vec2[0], vec2[1], vec2[2],
          vec3[0], vec3[1], vec3[2]);
  fprintf(outfile, "\"N\" [ %g %g %g %g %g %g %g %g %g ]\n",
          norm1[0], norm1[1], norm1[2],
          norm2[0], norm2[1], norm2[2],
          norm3[0], norm3[1], norm3[2]);
}


// draw a tricolor
void RenderManDisplayDevice::tricolor(const float *a, const float *b, const float *c,
                      const float *n1, const float *n2, const float *n3,
                      const float *c1, const float *c2, const float *c3) {
  float vec1[3], vec2[3], vec3[3];
  float norm1[3], norm2[3], norm3[3];

  // Transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  (transMat.top()).multpoint3d(c, vec3);
  (transMat.top()).multnorm3d(n1, norm1);
  (transMat.top()).multnorm3d(n2, norm2);
  (transMat.top()).multnorm3d(n3, norm3);

  // Write the triangle
  write_materials(0);
  fprintf(outfile, "Polygon \"P\" [ %g %g %g %g %g %g %g %g %g ] ",
          vec1[0], vec1[1], vec1[2],
          vec2[0], vec2[1], vec2[2],
          vec3[0], vec3[1], vec3[2]);
  fprintf(outfile, "\"N\" [ %g %g %g %g %g %g %g %g %g ] ",
          norm1[0], norm1[1], norm1[2],
          norm2[0], norm2[1], norm2[2],
          norm3[0], norm3[1], norm3[2]);
  fprintf(outfile, "\"Cs\" [ %g %g %g %g %g %g %g %g %g ]\n",
          c1[0], c1[1], c1[2],
          c2[0], c2[1], c2[2],
          c3[0], c3[1], c3[2]);
}


// draw a square
void RenderManDisplayDevice::square(float *n, float *a, float *b, float *c, float *d) {
  float vec1[3], vec2[3], vec3[3], vec4[3];
  float norm[3];

  // Transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  (transMat.top()).multpoint3d(c, vec3);
  (transMat.top()).multpoint3d(d, vec4);
  (transMat.top()).multnorm3d(n, norm);

  // Write the square
  write_materials(1);
  fprintf(outfile, "Polygon \"P\" [ %g %g %g %g %g %g %g %g %g %g %g %g ] ",
          vec1[0], vec1[1], vec1[2],
          vec2[0], vec2[1], vec2[2],
          vec3[0], vec3[1], vec3[2],
          vec4[0], vec4[1], vec4[2]);
  fprintf(outfile, "\"N\" [ %g %g %g %g %g %g %g %g %g %g %g %g ]\n",
          norm[0], norm[1], norm[2],
          norm[0], norm[1], norm[2],
          norm[0], norm[1], norm[2],
          norm[0], norm[1], norm[2]);
}


// display a comment
void RenderManDisplayDevice::comment(const char *s) {
  fprintf(outfile, "# %s\n", s);
}

///////////////////// public virtual routines

void RenderManDisplayDevice::write_header() {
  int i, n;

  // Initialize the RenderMan interface
  fprintf(outfile, "##RenderMan RIB-Structure 1.0\n");
  fprintf(outfile, "version 3.03\n");

  fprintf(outfile, "Display \"plot.tif\" \"file\" \"rgba\"\n");
  fprintf(outfile, "Format %ld %ld 1\n", xSize, ySize);

  // Make coordinate system right-handed
  fprintf(outfile, "Scale 1 1 -1\n");

  fprintf( outfile, "FrameAspectRatio %g\n", Aspect);

  if ( projection() == PERSPECTIVE ){
    fprintf(outfile, "Projection \"perspective\" \"fov\" %g\n",
            360.0*atan2((double) 0.5*vSize, (double) eyePos[2]-zDist)*VMD_1_PI);
  } else {
    // scaling necessary to equalize sizes of vmd screen and image 
    fprintf(outfile, "ScreenWindow %g %g %g %g\n",
            -Aspect*vSize/4, Aspect*vSize/4, -vSize/4, vSize/4);
    fprintf(outfile, "Projection \"orthographic\"\n");
  }

  // Set up the camera position
  fprintf(outfile, "Translate %g %g %g\n", -eyePos[0], -eyePos[1], -eyePos[2]);

  // shadows on, comment out for no shadows
  fprintf( outfile, "Declare \"shadows\" \"string\"\n");
  fprintf( outfile, "Attribute \"light\" \"shadows\" \"on\"\n" );

  // ambient light source (for ambient shading values)
  fprintf(outfile, "LightSource \"ambientlight\" 0 \"intensity\" [1.0] \"lightcolor\" [1 1 1]\n" );
  
  n = 1;
  // Write out all the light sources as point lights
  for (i = 0; i < DISP_LIGHTS; i++) {
    if (lightState[i].on) {
//      fprintf(outfile, "LightSource \"pointlight\" %d \"intensity\" [1.0] \"lightcolor\" [%g %g %g] \"from\" [%g %g %g]\n",
      fprintf(outfile, "LightSource \"distantlight\" %d \"intensity\" [1.0] \"lightcolor\" [%g %g %g] \"from\" [%g %g %g] \"to\" [0 0 0]\n",
      n++,
      lightState[i].color[0], lightState[i].color[1], lightState[i].color[2],
      lightState[i].pos[0], lightState[i].pos[1], lightState[i].pos[2]);
    }
  }

  // background color rendering takes longer, but is expected behavior
  fprintf(outfile, "# Background colors slow down rendering, \n");
  fprintf(outfile, "# but this is what most people expect by \n");
  fprintf(outfile, "# default. Comment these lines for a transparent\n");
  fprintf(outfile, "# background.\n");
  fprintf(outfile, "Declare \"bgcolor\" \"uniform color\"\n");
  fprintf(outfile, "Imager \"background\" \"bgcolor\" [%g %g %g]\n",
          backColor[0], backColor[1], backColor[2]);

  fprintf(outfile, "WorldBegin\n");
}


void RenderManDisplayDevice::write_trailer(void){
  fprintf(outfile, "WorldEnd\n");
  reset_vars(); // reinitialize material cache
}


void RenderManDisplayDevice::write_materials(int write_color) {
  // keep track of what the last written material properties
  // are, that way we can avoid writing redundant def's
  if (write_color) {
    // the color has changed since last write, emit an update 
    if ((matData[colorIndex][0] != old_color[0]) ||
        (matData[colorIndex][1] != old_color[1]) ||
        (matData[colorIndex][2] != old_color[2])) {
      fprintf(outfile, "  Color %g %g %g\n",
              matData[colorIndex][0], 
              matData[colorIndex][1],
              matData[colorIndex][2]);
      // save the last color
      memcpy(old_color, matData[colorIndex], sizeof(float) * 3);
    }
  }

  // now check opacity
  if (mat_opacity != old_opacity) {
    fprintf(outfile, "  Opacity %g %g %g\n", 
            mat_opacity, mat_opacity, mat_opacity);
    old_opacity = mat_opacity;
  }

  // and the lighting and roughness coefficients
  if ((mat_ambient != old_ambient) || 
      (mat_diffuse != old_diffuse) ||
      (mat_specular != old_specular)) {
    float roughness=10000.0;
    if (mat_shininess > 0.00001f) {
      roughness = 1.0f / mat_shininess;
    }
    fprintf(outfile, "  Surface \"plastic\"" 
            "\"Ka\" %g \"Kd\" %g \"Ks\" %g \"roughness\" %g\n",
            mat_ambient, mat_diffuse, mat_specular, roughness);
    old_ambient = mat_ambient;
    old_specular = mat_specular;
    old_diffuse = mat_diffuse;
  }
}



