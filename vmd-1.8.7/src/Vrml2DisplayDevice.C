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
 *	$RCSfile: Vrml2DisplayDevice.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.17 $	$Date: 2009/04/29 15:43:32 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   VRML2 / VRML97 scene export code
 ***************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>  /* this is for the Hash Table */

#include "Vrml2DisplayDevice.h"
#include "Matrix4.h"
#include "DispCmds.h"
#include "Inform.h"
#include "utilities.h"

#define DEFAULT_RADIUS 0.002
#define DASH_LENGTH 0.02

///////////////////////// constructor and destructor

// constructor ... initialize some variables
Vrml2DisplayDevice::Vrml2DisplayDevice(void) : 
  FileRenderer("VRML-2", "render.wrl", "true") {
	
  tList = NULL;
}
               
///////////////////////// protected nonvirtual routines
void Vrml2DisplayDevice::set_color(int mycolorIndex) {
#if 0
  write_cindexmaterial(mycolorIndex, materialIndex);
#endif
}

// draw a sphere
void Vrml2DisplayDevice::sphere(float *xyzr) {
  fprintf(outfile, "    Transform {\n");
  fprintf(outfile, "      translation %f %f %f\n", xyzr[0], xyzr[1], xyzr[2]);
  fprintf(outfile, "      children [ Shape {\n");
  write_cindexmaterial(colorIndex, materialIndex);
  fprintf(outfile, "        geometry Sphere { radius %f }\n", xyzr[3]);
  fprintf(outfile, "      }]\n");
  fprintf(outfile, "    }\n");
}

//// draw a line from a to b
////  Doesn't yet support the dotted line method
void Vrml2DisplayDevice::line(float *a, float*b) {
  // ugly and wasteful, but it will work
  fprintf(outfile, "Shape {\n");
  write_cindexmaterial(colorIndex, materialIndex);
  fprintf(outfile, "  geometry IndexedLineSet { \n"); 
  fprintf(outfile, "    coordIndex [ 0, 1, -1 ]\n");
  fprintf(outfile, "    coord Coordinate { point [ %f %f %f,  %f %f %f ] }\n",
	  a[0], a[1], a[2], b[0], b[1], b[2]);
  fprintf(outfile, "  }\n");
  fprintf(outfile, "}\n");
}



// draw a cylinder
void Vrml2DisplayDevice::cylinder(float *a, float *b, float r, int filled) {
  float height = distance(a, b);

  if (a[0] == b[0] && a[1] == b[1] && a[2] == b[2]) {
    return;  // we don't serve your kind here
  }

  fprintf(outfile, "    Transform {\n");
  fprintf(outfile, "      translation %f %f %f\n", 
          a[0], a[1] + (height / 2.0), a[2]);

  float rotaxis[3];
  float cylaxdir[3];
  float yaxis[3] = {0.0, 1.0, 0.0};

  vec_sub(cylaxdir, b, a);
  vec_normalize(cylaxdir);
  float dp = dot_prod(yaxis, cylaxdir);

  cross_prod(rotaxis, cylaxdir, yaxis);
  vec_normalize(rotaxis);

  if ((rotaxis[0]*rotaxis[0] + 
      rotaxis[1]*rotaxis[1] + 
      rotaxis[2]*rotaxis[2]) > 0.5) { 
    fprintf(outfile, "      center 0.0 %f 0.0\n", -(height / 2.0));
    fprintf(outfile, "      rotation %f %f %f  %f\n", 
      rotaxis[0], rotaxis[1], rotaxis[2], 
      -acos(dp));
  }
          
  fprintf(outfile, "      children [ Shape {\n");
  write_cindexmaterial(colorIndex, materialIndex);

#if 0
  // draw the cylinder
  fprintf(outfile, "        geometry Cylinder { "
          "bottom %s height %f radius %f side %s top %s }\n", 
	  filled ? "TRUE" : "FALSE",
          height,  
          r, 
	  "TRUE",
	  filled ? "TRUE" : "FALSE");
#else
  if (filled) {
    fprintf(outfile, "        geometry Cylinder { "
            "height %f radius %f }\n", height,  r);
  } else {
    fprintf(outfile, "        geometry VMDCyl { "
            "h %f r %f }\n", height,  r);
  }
#endif

  fprintf(outfile, "      }]\n");
  fprintf(outfile, "    }\n");
}


void Vrml2DisplayDevice::cone(float *a, float *b, float r) {
  float height = distance(a, b);

  if (a[0] == b[0] && a[1] == b[1] && a[2] == b[2]) {
    return;  // we don't serve your kind here
  }

  fprintf(outfile, "    Transform {\n");
  fprintf(outfile, "      translation %f %f %f\n", 
          a[0], a[1] + (height / 2.0), a[2]);

  float rotaxis[3];
  float cylaxdir[3];
  float yaxis[3] = {0.0, 1.0, 0.0};

  vec_sub(cylaxdir, b, a);
  vec_normalize(cylaxdir);
  float dp = dot_prod(yaxis, cylaxdir);

  cross_prod(rotaxis, cylaxdir, yaxis);
  vec_normalize(rotaxis);

  if ((rotaxis[0]*rotaxis[0] + 
      rotaxis[1]*rotaxis[1] + 
      rotaxis[2]*rotaxis[2]) > 0.5) { 
    fprintf(outfile, "      center 0.0 %f 0.0\n", -(height / 2.0));
    fprintf(outfile, "      rotation %f %f %f  %f\n", 
      rotaxis[0], rotaxis[1], rotaxis[2], 
      -acos(dp));
  }
          
  fprintf(outfile, "      children [ Shape {\n");
  write_cindexmaterial(colorIndex, materialIndex);

  // draw the cone
  fprintf(outfile, "        geometry Cone { bottomRadius %f height %f }\n", 
          r, height);

  fprintf(outfile, "      }]\n");
  fprintf(outfile, "    }\n");
}



// draw a triangle
void Vrml2DisplayDevice::triangle(const float *a, const float *b, const float *c, 
				  const float *n1, const float *n2, const float *n3) {
  // ugly and wasteful, but it will work
  fprintf(outfile, "Shape {\n");
  write_cindexmaterial(colorIndex, materialIndex);
  fprintf(outfile, "  geometry IndexedFaceSet { \n"); 
  fprintf(outfile, "    coordIndex [ 0, 1, 2, -1 ]\n");
  fprintf(outfile, "    coord Coordinate { point [ %f %f %f,  %f %f %f,  %f %f %f ] }\n",
	  a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
   
  fprintf(outfile, "    normal Normal { vector [ %f %f %f, %f %f %f, %f %f %f ] }\n",
	  n1[0], n1[1], n1[2], n2[0], n2[1], n2[2], n3[0], n3[1], n3[2]);

  fprintf(outfile, "  }\n");
  fprintf(outfile, "}\n");
}


// draw a color-per-vertex triangle
void Vrml2DisplayDevice::tricolor(const float * xyz1, const float * xyz2, const float * xyz3, 
                        const float * n1,   const float * n2,   const float * n3,
                        const float *c1, const float *c2, const float *c3) {
  // ugly and wasteful, but it will work
  fprintf(outfile, "Shape {\n");
  write_cindexmaterial(colorIndex, materialIndex);
  fprintf(outfile, "  geometry IndexedFaceSet { \n"); 
  fprintf(outfile, "    coordIndex [ 0, 1, 2, -1 ]\n");
  fprintf(outfile, "    coord Coordinate { point [ %f %f %f,  %f %f %f,  %f %f %f ] }\n",
	  xyz1[0], xyz1[1], xyz1[2], xyz2[0], xyz2[1], xyz2[2], xyz3[0], xyz3[1], xyz3[2]);

  fprintf(outfile, "    color Color { color [ %f %f %f, %f %f %f, %f %f %f ] }\n", 
	  c1[0], c1[1], c1[2], c2[0], c2[1], c2[2], c3[0], c3[1], c3[2]);
   
  fprintf(outfile, "    normal Normal { vector [ %f %f %f, %f %f %f, %f %f %f ] }\n",
	  n1[0], n1[1], n1[2], n2[0], n2[1], n2[2], n3[0], n3[1], n3[2]);

  fprintf(outfile, "  }\n");
  fprintf(outfile, "}\n");
}

void Vrml2DisplayDevice::multmatrix(const Matrix4 &mat) {
}

void Vrml2DisplayDevice::load(const Matrix4 &mat) {
}

void Vrml2DisplayDevice::comment(const char *s) {
  fprintf (outfile, "# %s\n", s);
}

///////////////////// public virtual routines

// initialize the file for output
void Vrml2DisplayDevice::write_header(void) {
  fprintf(outfile, "#VRML V2.0 utf8\n");
  fprintf(outfile, "# Created with VMD: "
	  "http://www.ks.uiuc.edu/Research/vmd/\n");

  // define our special node types
  fprintf(outfile, "# Define some custom nodes VMD to decrease file size\n");
  fprintf(outfile, "# custom VMD cylinder node\n");
  fprintf(outfile, "PROTO VMDCyl [\n");
  fprintf(outfile, "  field SFBool  bottom FALSE\n");
  fprintf(outfile, "  field SFFloat h      2    \n");
  fprintf(outfile, "  field SFFloat r      1    \n");
  fprintf(outfile, "  field SFBool  side   TRUE \n");
  fprintf(outfile, "  field SFBool  top    FALSE\n");
  fprintf(outfile, "  ] {\n");
  fprintf(outfile, "  Cylinder {\n"); 
  fprintf(outfile, "    bottom IS bottom\n");
  fprintf(outfile, "    height IS h     \n");
  fprintf(outfile, "    radius IS r     \n");
  fprintf(outfile, "    top    IS top   \n");
  fprintf(outfile, "  }\n");
  fprintf(outfile, "}\n\n");

  fprintf(outfile, "# custom VMD materials node\n");
  fprintf(outfile, "PROTO VMDMat [\n");
  fprintf(outfile, "  field SFFloat Ka               0.0\n"); 
  fprintf(outfile, "  field SFColor Kd               0.8 0.8 0.8\n");
  fprintf(outfile, "  field SFColor emissiveColor    0.0 0.0 0.0\n");
  fprintf(outfile, "  field SFFloat Ksx              0.0\n"); 
  fprintf(outfile, "  field SFColor Ks               0.0 0.0 0.0\n");
  fprintf(outfile, "  field SFFloat Kt               0.0\n"); 
  fprintf(outfile, "  ] {\n");
  fprintf(outfile, "  Appearance {\n");
  fprintf(outfile, "    material Material {\n");
  fprintf(outfile, "      ambientIntensity IS Ka           \n");
  fprintf(outfile, "      diffuseColor     IS Kd           \n");
  fprintf(outfile, "      emissiveColor    IS emissiveColor\n");
  fprintf(outfile, "      shininess        IS Ksx          \n");
  fprintf(outfile, "      specularColor    IS Ks           \n");
  fprintf(outfile, "      transparency     IS Kt           \n");
  fprintf(outfile, "    }\n");
  fprintf(outfile, "  }\n");
  fprintf(outfile, "}\n\n");

  fprintf(outfile, "\n");
  fprintf(outfile, "# begin the actual scene\n");
  fprintf(outfile, "Group {\n");
  fprintf(outfile, "  children [\n");
}

void Vrml2DisplayDevice::write_trailer(void) {
  fprintf(outfile, "  ]\n");
  fprintf(outfile, "}\n");
}

void Vrml2DisplayDevice::write_cindexmaterial(int cindex, int material) {
  write_colormaterial((float *) &matData[cindex], material);
}

void Vrml2DisplayDevice::write_colormaterial(float *rgb, int) {

#if 0
  // use the current material definition
  fprintf(outfile, "        appearance Appearance {\n");
  fprintf(outfile, "          material Material {\n"); 
  fprintf(outfile, "            ambientIntensity %f\n", mat_ambient);
  fprintf(outfile, "            diffuseColor %f %f %f\n",
	  mat_diffuse * rgb[0],
	  mat_diffuse * rgb[1],
	  mat_diffuse * rgb[2]);
  fprintf(outfile, "            shininess %f\n", mat_shininess);
  fprintf(outfile, "            specularColor %f %f %f\n",
          mat_specular,
          mat_specular,
          mat_specular);
  fprintf(outfile, "            transparency %f\n", 1.0 - mat_opacity);
  fprintf(outfile, "          }\n");
  fprintf(outfile, "        }\n");
#else
  // use the current material definition
  fprintf(outfile, "        appearance VMDMat { ");

  if (mat_ambient > 0.0) {
    fprintf(outfile, "Ka %f ", mat_ambient);
  } 

  fprintf(outfile, "Kd %f %f %f ",
	  mat_diffuse * rgb[0],
	  mat_diffuse * rgb[1],
	  mat_diffuse * rgb[2]);

  if (mat_specular > 0.0) {
    fprintf(outfile, "Ksx %f ", mat_shininess);
    fprintf(outfile, "Ks %f %f %f ", mat_specular, mat_specular, mat_specular);
  }

  if (mat_opacity < 1.0) {
    fprintf(outfile, "Kt %f ", 1.0 - mat_opacity);
  }
  fprintf(outfile, " }\n");
#endif

}

