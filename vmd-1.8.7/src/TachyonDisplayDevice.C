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
*      $RCSfile: TachyonDisplayDevice.C,v $
*      $Author: johns $        $Locker:  $               $State: Exp $
*      $Revision: 1.97 $        $Date: 2009/06/01 07:05:20 $
*
***************************************************************************
* DESCRIPTION:
*
* FileRenderer type for the Tachyon Parallel / Multiprocessor Ray Tracer 
*
***************************************************************************/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "TachyonDisplayDevice.h"
#include "Matrix4.h"
#include "DispCmds.h"
#include "Inform.h"
#include "utilities.h"
#include "config.h"    // for VMDVERSION string

#define DEFAULT_RADIUS 0.002
#define DASH_LENGTH 0.02

#if defined(_MSC_VER) || defined(MINGW) || defined(WIN32)
#define TACHYON_RUN_STRING " -aasamples 12 %s -format BMP -o %s.bmp"
#else
#define TACHYON_RUN_STRING " -aasamples 12 %s -format TARGA -o %s.tga"
#endif

static char tachyon_run_string[2048];

static char * get_tachyon_run_string() {
  char *tbin;
  strcpy(tachyon_run_string, "tachyon");
  
  if ((tbin=getenv("TACHYON_BIN")) != NULL) {
    sprintf(tachyon_run_string, "\"%s\"", tbin);
  }
  strcat(tachyon_run_string, TACHYON_RUN_STRING);
 
  return tachyon_run_string;
}

void TachyonDisplayDevice::update_exec_cmd() {
  const char *tbin;
  if ((tbin = getenv("TACHYON_BIN")) == NULL)
    tbin = "tachyon";

  switch(curformat) {
    case 0:  // BMP
      sprintf(tachyon_run_string, 
        "\"%s\" -aasamples %d %%s -format %s -o %%s.%s", tbin, aasamples, "BMP", "bmp");
      break;

    case 1: // PPM
      sprintf(tachyon_run_string, 
        "\"%s\" -aasamples %d %%s -format %s -o %%s.%s", tbin, aasamples, "PPM", "ppm");
      break;

    case 2: // PPM48
      sprintf(tachyon_run_string, 
        "\"%s\" -aasamples %d %%s -format %s -o %%s.%s", tbin, aasamples, "PPM48", "ppm");
      break;

    case 3: // PSD
      sprintf(tachyon_run_string, 
        "\"%s\" -aasamples %d %%s -format %s -o %%s.%s", tbin, aasamples, "PSD48", "psd");
      break;

    case 4: // SGI RGB
      sprintf(tachyon_run_string, 
        "\"%s\" -aasamples %d %%s -format %s -o %%s.%s", tbin, aasamples, "RGB", "rgb");
      break;

    case 5: // TARGA
    default:
      sprintf(tachyon_run_string, 
        "\"%s\" -aasamples %d %%s -format %s -o %%s.%s", tbin, aasamples, "TARGA", "tga");
      break;
  }
  delete [] execCmd;
  execCmd = stringdup(tachyon_run_string);
}

///////////////////////// constructor and destructor

// constructor ... initialize some variables

TachyonDisplayDevice::TachyonDisplayDevice() : FileRenderer ("Tachyon","plot.dat", get_tachyon_run_string()) { 
  // Add supported file formats
  formats.add_name("BMP", 0);
  formats.add_name("PPM", 0);
  formats.add_name("PPM48", 0);
  formats.add_name("PSD48", 0);
  formats.add_name("RGB", 0);
  formats.add_name("TGA", 0);

  // Set default aa level
  has_aa = TRUE;
  aasamples = 12;
  aosamples = 12;

  reset_vars();

  // Default image format depends on platform
#if defined(_MSC_VER) || defined(MINGW) || defined(WIN32)
  curformat = 0; // Windows BMP
#else
  curformat = 5; // Targa
#endif
}
        
// destructor
TachyonDisplayDevice::~TachyonDisplayDevice(void) { }

///////////////////////// protected nonvirtual routines

void TachyonDisplayDevice::reset_vars(void) {
  inclipgroup = 0; // not currently in a clipping group
  involtex = 0;    // volume texturing disabled
  voltexID = -1;   // invalid texture ID
  memset(xplaneeq, 0, sizeof(xplaneeq));
  memset(yplaneeq, 0, sizeof(xplaneeq));
  memset(zplaneeq, 0, sizeof(xplaneeq));
}  


// emit a comment line 
void TachyonDisplayDevice::comment(const char *s) {
  fprintf(outfile, "# %s\n", s);
}


// draw a point
void TachyonDisplayDevice::point(float * spdata) {
  float vec[3];
  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);

  // draw the sphere
  fprintf(outfile, "Sphere \n");  // sphere
  fprintf(outfile, "  Center %g %g %g \n ", vec[0], vec[1], -vec[2]);
  fprintf(outfile, "  Rad %g \n",     float(lineWidth)*DEFAULT_RADIUS); 
  write_cindexmaterial(colorIndex, materialIndex);
}


// draw a sphere
void TachyonDisplayDevice::sphere(float * spdata) {
  float vec[3];
  float radius;
    
  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);
  radius = scale_radius(spdata[3]);
   
  // draw the sphere
  fprintf(outfile, "Sphere \n");  // sphere
  fprintf(outfile, "  Center %g %g %g \n ", vec[0], vec[1], -vec[2]);
  fprintf(outfile, "  Rad %g \n", radius ); 
  write_cindexmaterial(colorIndex, materialIndex);
}


// draw a line (cylinder) from a to b
void TachyonDisplayDevice::line(float *a, float*b) {
  int i, j, test;
  float dirvec[3], unitdirvec[3];
  float from[3], to[3], tmp1[3], tmp2[3];
  float len;
    
  if (lineStyle == ::SOLIDLINE) {
    // transform the world coordinates
    (transMat.top()).multpoint3d(a, from);
    (transMat.top()).multpoint3d(b, to);
    
    // draw the cylinder
    fprintf(outfile, "FCylinder\n"); // flat-ended cylinder
    fprintf(outfile, "  Base %g %g %g\n", from[0], from[1], -from[2]); 
    fprintf(outfile, "  Apex %g %g %g\n", to[0], to[1], -to[2]);
    fprintf(outfile, "  Rad %g \n", float(lineWidth)*DEFAULT_RADIUS);
    write_cindexmaterial(colorIndex, materialIndex);

  } else if (lineStyle == ::DASHEDLINE ) {
     // transform the world coordinates
    (transMat.top()).multpoint3d(a, tmp1);
    (transMat.top()).multpoint3d(b, tmp2);

    // how to create a dashed line
    for(i=0; i<3; i++) {
      dirvec[i] = tmp2[i] - tmp1[i];  // vector from a to b
    }
    len = sqrtf(dirvec[0]*dirvec[0] + dirvec[1]*dirvec[1] + dirvec[2]*dirvec[2]);
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
    
      // draw the cylinder
      fprintf(outfile, "FCylinder\n"); // flat-ended cylinder
      fprintf(outfile, "  Base %g %g %g\n", from[0], from[1], -from[2]); 
      fprintf(outfile, "  Apex %g %g %g\n", to[0], to[1], -to[2]);
      fprintf(outfile, "  Rad %g \n", float(lineWidth)*DEFAULT_RADIUS);
      write_cindexmaterial(colorIndex, materialIndex);
      i++;
    }
  } else {
    msgErr << "TachyonDisplayDevice: Unknown line style " 
           << lineStyle << sendmsg;
  }
}




// draw a cylinder
void TachyonDisplayDevice::cylinder(float *a, float *b, float r, int filled) {
  float from[3], to[3], norm[3];
  float radius;
  filled = filled;  

  // transform the world coordinates
  (transMat.top()).multpoint3d(a, from);
  (transMat.top()).multpoint3d(b, to);
  radius = scale_radius(r);
   
 
  // draw the cylinder
  fprintf(outfile, "FCylinder\n"); // flat-ended cylinder
  fprintf(outfile, "  Base %g %g %g\n", from[0], from[1], -from[2]); 
  fprintf(outfile, "  Apex %g %g %g\n", to[0], to[1], -to[2]);
  fprintf(outfile, "  Rad %g\n", radius);
  write_cindexmaterial(colorIndex, materialIndex);

  // Cylinder caps?
  if (filled) {
    float div;

    norm[0] = to[0] - from[0];
    norm[1] = to[1] - from[1];
    norm[2] = to[2] - from[2];

    div = 1.0f / sqrtf(norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2]);
    norm[0] *= div;
    norm[1] *= div;
    norm[2] *= div;

    if (filled & CYLINDER_TRAILINGCAP) {
      fprintf(outfile, "Ring\n");
      fprintf(outfile, "Center %g %g %g \n", from[0], from[1], -from[2]);
      fprintf(outfile, "Normal %g %g %g \n", norm[0], norm[1], -norm[2]); 
      fprintf(outfile, "Inner 0.0  Outer %g \n", radius);
      write_cindexmaterial(colorIndex, materialIndex);
    }
  
    if (filled & CYLINDER_LEADINGCAP) {
      fprintf(outfile, "Ring\n");
      fprintf(outfile, "Center %g %g %g \n", to[0], to[1], -to[2]);
      fprintf(outfile, "Normal %g %g %g \n", -norm[0], -norm[1], norm[2]); 
      fprintf(outfile, "Inner 0.0  Outer %g \n", radius);
      write_cindexmaterial(colorIndex, materialIndex);
    }
  }
}


// draw a triangle
void TachyonDisplayDevice::triangle(const float *a, const float *b, const float *c, const float *n1, const float *n2, const float *n3) {
  float vec1[3], vec2[3], vec3[3];
  float norm1[3], norm2[3], norm3[3];
  
  
  // transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  (transMat.top()).multpoint3d(c, vec3);

  // and the normals
  (transMat.top()).multnorm3d(n1, norm1);
  (transMat.top()).multnorm3d(n2, norm2);
  (transMat.top()).multnorm3d(n3, norm3);

  // draw the triangle
  fprintf(outfile, "STri\n"); // triangle
  fprintf(outfile, "  V0 %g %g %g\n", vec1[0], vec1[1], -vec1[2]);
  fprintf(outfile, "  V1 %g %g %g\n", vec2[0], vec2[1], -vec2[2]); 
  fprintf(outfile, "  V2 %g %g %g\n", vec3[0], vec3[1], -vec3[2]);
  fprintf(outfile, "  N0 %g %g %g\n", -norm1[0], -norm1[1], norm1[2]);
  fprintf(outfile, "  N1 %g %g %g\n", -norm2[0], -norm2[1], norm2[2]); 
  fprintf(outfile, "  N2 %g %g %g\n", -norm3[0], -norm3[1], norm3[2]);
  write_cindexmaterial(colorIndex, materialIndex);
}

// draw triangle with per-vertex colors
void TachyonDisplayDevice::tricolor(const float * xyz1, const float * xyz2, const float * xyz3,
                                    const float * n1,   const float * n2,   const float * n3,
                                    const float *c1, const float *c2, const float *c3) {
  float vec1[3], vec2[3], vec3[3];
  float norm1[3], norm2[3], norm3[3];

  // transform the world coordinates
  (transMat.top()).multpoint3d(xyz1, vec1);
  (transMat.top()).multpoint3d(xyz2, vec2);
  (transMat.top()).multpoint3d(xyz3, vec3);

  // and the normals
  (transMat.top()).multnorm3d(n1, norm1);
  (transMat.top()).multnorm3d(n2, norm2);
  (transMat.top()).multnorm3d(n3, norm3);

  // draw the triangle
  if (!involtex) {
    fprintf(outfile, "VCSTri\n"); // triangle
  } else {
    fprintf(outfile, "STri\n"); // triangle
  }
  fprintf(outfile, "  V0 %g %g %g\n", vec1[0], vec1[1], -vec1[2]);
  fprintf(outfile, "  V1 %g %g %g\n", vec2[0], vec2[1], -vec2[2]);
  fprintf(outfile, "  V2 %g %g %g\n", vec3[0], vec3[1], -vec3[2]);

  fprintf(outfile, "  N0 %g %g %g\n", -norm1[0], -norm1[1], norm1[2]);
  fprintf(outfile, "  N1 %g %g %g\n", -norm2[0], -norm2[1], norm2[2]);
  fprintf(outfile, "  N2 %g %g %g\n", -norm3[0], -norm3[1], norm3[2]);

  if (!involtex) {
    fprintf(outfile, "  C0 %g %g %g\n", c1[0], c1[1], c1[2]);
    fprintf(outfile, "  C1 %g %g %g\n", c2[0], c2[1], c2[2]);
    fprintf(outfile, "  C2 %g %g %g\n", c3[0], c3[1], c3[2]);
  }

  if (materials_on) {
    fprintf(outfile, "  Ambient %g Diffuse %g Specular %g Opacity %g\n",
            mat_ambient, mat_diffuse, 0.0, mat_opacity);
  } else {
    fprintf(outfile, "  Ambient %g Diffuse %g Specular %g Opacity %g\n",
            1.0, 0.0, 0.0, mat_opacity);
  }

  if (mat_outline > 0.0) {
    fprintf(outfile, "  Outline %g Outline_Width %g ", 
            mat_outline, mat_outlinewidth);
  }
  fprintf(outfile, "  Phong Plastic %g Phong_size %g ", mat_specular,
          mat_shininess);
  fprintf(outfile, "VCST\n\n");
}


// use an efficient mesh primitve rather than individual triangles
// when possible.
void TachyonDisplayDevice::trimesh(int numverts, float * cnv,
                                   int numfacets, int * facets) {
  int i;
  float vec1[3];
  float norm1[3];

  fprintf(outfile, "VertexArray");
  fprintf(outfile, "  Numverts %d\n", numverts);

  fprintf(outfile, "\nCoords\n");
  for (i=0; i<numverts; i++) {
    (transMat.top()).multpoint3d(cnv + i*10 + 7, vec1);
    fprintf(outfile, "%g %g %g\n", vec1[0], vec1[1], -vec1[2]);
  } 

  fprintf(outfile, "\nNormals\n");
  for (i=0; i<numverts; i++) {
    (transMat.top()).multnorm3d(cnv + i*10 + 4, norm1);
    fprintf(outfile, "%g %g %g\n", -norm1[0], -norm1[1], norm1[2]);
  } 

  // don't emit per-vertex colors when volumetric texturing is enabled
  if (!involtex) {
    fprintf(outfile, "\nColors\n");
    for (i=0; i<numverts; i++) {
      int idx = i * 10;
      fprintf(outfile, "%g %g %g\n", cnv[idx], cnv[idx+1], cnv[idx+2]);
    } 
  }

  // emit the texture to be used by the geometry that follows
  write_cindexmaterial(colorIndex, materialIndex);

  // loop over all of the facets in the mesh
  fprintf(outfile, "\nTriMesh %d\n", numfacets);
  for (i=0; i<numfacets*3; i+=3) {
    fprintf(outfile, "%d %d %d\n", facets[i], facets[i+1], facets[i+2]);
  }
 
  // terminate vertex array 
  fprintf(outfile, "\nEnd_VertexArray\n");
}


void TachyonDisplayDevice::tristrip(int numverts, const float * cnv,
                                   int numstrips, const int *vertsperstrip,
                                   const int *facets) {
  int i, strip, v=0;
  float vec1[3];
  float norm1[3];

  fprintf(outfile, "VertexArray");
  fprintf(outfile, "  Numverts %d\n", numverts);

  fprintf(outfile, "\nCoords\n");
  for (i=0; i<numverts; i++) {
    (transMat.top()).multpoint3d(cnv + i*10 + 7, vec1);
    fprintf(outfile, "%g %g %g\n", vec1[0], vec1[1], -vec1[2]);
  } 

  fprintf(outfile, "\nNormals\n");
  for (i=0; i<numverts; i++) {
    (transMat.top()).multnorm3d(cnv + i*10 + 4, norm1);
    fprintf(outfile, "%g %g %g\n", -norm1[0], -norm1[1], norm1[2]);
  } 

  // don't emit per-vertex colors when volumetric texturing is enabled
  if (!involtex) {
    fprintf(outfile, "\nColors\n");
    for (i=0; i<numverts; i++) {
      int idx = i * 10;
      fprintf(outfile, "%g %g %g\n", cnv[idx], cnv[idx+1], cnv[idx+2]);
    } 
  }

  // emit the texture to be used by the geometry that follows
  write_cindexmaterial(colorIndex, materialIndex);

  // loop over all of the triangle strips
  v=0;
  for (strip=0; strip < numstrips; strip++) {
    fprintf(outfile, "\nTriStrip %d\n", vertsperstrip[strip]);

    // loop over all triangles in this triangle strip
    for (i = 0; i < vertsperstrip[strip]; i++) {
      fprintf(outfile, "%d ", facets[v]);
      v++; // move on to the next triangle
    }
  }
 
  // terminate vertex array 
  fprintf(outfile, "\nEnd_VertexArray\n");
}


#if 1
// define a volumetric texture map
void TachyonDisplayDevice::define_volume_texture(int ID, 
                                                 int xs, int ys, int zs,
                                                 const float *xpq,
                                                 const float *ypq,
                                                 const float *zpq,
                                                 unsigned char *texmap) {
  voltexID = ID; // remember current texture ID

  memcpy(xplaneeq, xpq, sizeof(xplaneeq));
  memcpy(yplaneeq, ypq, sizeof(yplaneeq));
  memcpy(zplaneeq, zpq, sizeof(zplaneeq));

  fprintf(outfile, "# VMD volume texture definition: ID %d\n", ID);
  fprintf(outfile, "#  Res: %d %d %d\n", xs, ys, zs);
  fprintf(outfile, "#  xplaneeq: %g %g %g %g\n",
         xplaneeq[0], xplaneeq[1], xplaneeq[2], xplaneeq[3]);
  fprintf(outfile, "#  yplaneeq: %g %g %g %g\n",
         yplaneeq[0], yplaneeq[1], yplaneeq[2], yplaneeq[3]);
  fprintf(outfile, "#  zplaneeq: %g %g %g %g\n",
         zplaneeq[0], zplaneeq[1], zplaneeq[2], zplaneeq[3]);

  fprintf(outfile, "ImageDef ::VMDVolTex%d\n", ID);
  fprintf(outfile, "  Format RGB24\n");
  fprintf(outfile, "  Resolution %d %d %d\n", xs, ys, zs);
  fprintf(outfile, "  Encoding Hex\n");
 
  int x, y, z;
  for (z=0; z<zs; z++) {
    for (y=0; y<ys; y++) {
      int addr = (z * xs * ys) + (y * xs);
      for (x=0; x<xs; x++) {
        int addr2 = (addr + x) * 3;
        fprintf(outfile, "%02x%02x%02x ", 
                texmap[addr2    ],
                texmap[addr2 + 1],
                texmap[addr2 + 2]);
      }
      fprintf(outfile, "\n");
    }
    fprintf(outfile, "\n");
  }  
  fprintf(outfile, "\n");
  fprintf(outfile, "# End of volume texture ::VMDVolTex%d\n", ID);
  fprintf(outfile, "\n");
  fprintf(outfile, "\n");
}


// enable volumetric texturing, either in "replace" or "modulate" mode
void TachyonDisplayDevice::volume_texture_on(int texmode) {
  involtex = 1;
}


// disable volumetric texturing
void TachyonDisplayDevice::volume_texture_off(void) {
  involtex = 0;
}

#else

// define a volumetric texture map
void TachyonDisplayDevice::define_volume_texture(int ID, 
                                                 int xs, int ys, int zs,
                                                 const float *txplaneeq,
                                                 const float *typlaneeq,
                                                 const float *tzplaneeq,
                                                 unsigned char *texmap) {
  warningflags |= FILERENDERER_NOTEXTURE;
}


// enable volumetric texturing, either in "replace" or "modulate" mode
void TachyonDisplayDevice::volume_texture_on(int texmode) {
  warningflags |= FILERENDERER_NOTEXTURE;
}


// disable volumetric texturing
void TachyonDisplayDevice::volume_texture_off(void) {
  warningflags |= FILERENDERER_NOTEXTURE;
}

#endif




void TachyonDisplayDevice::start_clipgroup(void) {
  int i;
  int planesenabled = 0;

  for (i=0; i<VMD_MAX_CLIP_PLANE; i++) {
    if (clip_mode[i] > 0) {
      planesenabled++;  /* count number of active clipping planes */
      if (clip_mode[i] > 1)
        warningflags |= FILERENDERER_NOCLIP; /* emit warnings */
    }
  }

  if (planesenabled > 0) {
    fprintf(outfile, "Start_ClipGroup\n");
    fprintf(outfile, " NumPlanes %d\n", planesenabled);
    for (i=0; i<VMD_MAX_CLIP_PLANE; i++) { 
      if (clip_mode[i] > 0) {
        float tachyon_clip_center[3]; 
        float tachyon_clip_normal[3];
        float tachyon_clip_distance;

        inclipgroup = 1; // we're in a clipping group presently

        // Transform the plane center and the normal
        (transMat.top()).multpoint3d(clip_center[i], tachyon_clip_center);
        (transMat.top()).multnorm3d(clip_normal[i], tachyon_clip_normal);
        vec_negate(tachyon_clip_normal, tachyon_clip_normal);

        // Tachyon uses the distance from the origin to the plane for its
        // representation, instead of the plane center
        tachyon_clip_distance = dot_prod(tachyon_clip_normal, tachyon_clip_center);

        fprintf(outfile, "%g %g %g %g\n", tachyon_clip_normal[0], 
                tachyon_clip_normal[1], -tachyon_clip_normal[2], 
                tachyon_clip_distance);
      }    
    }
    fprintf(outfile, "\n");
  } else {
    inclipgroup = 0; // Not currently in a clipping group
  }
}


///////////////////// public virtual routines

// initialize the file for output
void TachyonDisplayDevice::write_header() {
  fprintf(outfile, "# \n"); 
  fprintf(outfile, "# Molecular graphics exported from VMD %s\n", VMDVERSION);
  fprintf(outfile, "# http://www.ks.uiuc.edu/Research/vmd/\n");
  fprintf(outfile, "# \n"); 
  fprintf(outfile, "# Requires Tachyon version 0.98.7 or newer\n");
  fprintf(outfile, "# \n"); 
  fprintf(outfile, "# Default tachyon rendering command for this scene:\n"); 
  fprintf(outfile, "#   tachyon %s\n", TACHYON_RUN_STRING); 
  fprintf(outfile, "# \n"); 
  // NOTE: the vmd variable "Aspect" has absolutely *nothing* to do
  //       with aspect ratio correction, it is only the ratio of the
  //       width of the graphics window to its height, and so it should
  //       be used only to cause the ray tracer to generate a similarly
  //       proportioned image.

  fprintf(outfile, "Begin_Scene\n");
  fprintf(outfile, "Resolution %d %d\n", (int) xSize, (int) ySize);


  // Emit shading mode information
  fprintf(outfile, "Shader_Mode ");

  // change shading mode depending on whether the user wants shadows
  // or ambient occlusion lighting.
  if (shadows_enabled() || ao_enabled()) {
    fprintf(outfile, "Full\n");
  } else {
    fprintf(outfile, "Medium\n");
  }

  // For VMD we always want to enable flags that preserve a more WYSIWYG  
  // type of output, although in some cases doing things in Tachyon's
  // preferred way might be nicer.  The user can override these with 
  // command line flags still if they want radial fog or other options.
  fprintf(outfile, "  Trans_VMD\n");
  fprintf(outfile, "  Fog_VMD\n");

  // render with ambient occlusion lighting if required
  if (ao_enabled()) {
    fprintf(outfile, "  Ambient_Occlusion\n");
    fprintf(outfile, "    Ambient_Color %g %g %g\n", 
            get_ao_ambient(), get_ao_ambient(), get_ao_ambient());
    fprintf(outfile, "    Rescale_Direct %g\n", get_ao_direct());
    fprintf(outfile, "    Samples %d\n", aosamples);
  }
  fprintf(outfile, "End_Shader_Mode\n");

  write_camera();    // has to be first thing in the file. 
  write_lights();    // could be anywhere.
  write_materials(); // has to be before objects that use them.
}


void TachyonDisplayDevice::end_clipgroup(void) {
  if (inclipgroup) {
    fprintf(outfile, "End_ClipGroup\n");
    inclipgroup = 0; // we're not in a clipping group anymore
  }
}


void TachyonDisplayDevice::write_trailer(void){
  fprintf(outfile, "End_Scene \n");
  if (inclipgroup) {
    msgErr << "TachyonDisplayDevice clipping group still active at end of scene\n" << sendmsg;
  }
  msgInfo << "Tachyon file generation finished" << sendmsg;

  reset_vars();
}



///////////////////// Private routines

void TachyonDisplayDevice::write_camera(void) {

  // Camera position
  // Tachyon uses a left-handed coordinate system
  // VMD uses right-handed, so z(Tachyon) = -z(VMD).

  switch (projection()) {
    // XXX code for new versions of Tachyon that support orthographic views
    case DisplayDevice::ORTHOGRAPHIC:
      fprintf(outfile, "Camera\n");
      fprintf(outfile, "  Projection Orthographic\n");
      fprintf(outfile, "  Zoom %g\n", 1.0 / (vSize / 2.0));
      fprintf(outfile, "  Aspectratio %g\n", 1.0f);
      fprintf(outfile, "  Antialiasing %d\n", aasamples);
      fprintf(outfile, "  Raydepth 8\n");
      fprintf(outfile, "  Center  %g %g %g\n", eyePos[0], eyePos[1], -eyePos[2]);
      fprintf(outfile, "  Viewdir %g %g %g\n", eyeDir[0], eyeDir[1], -eyeDir[2]);
      fprintf(outfile, "  Updir   %g %g %g\n", upDir[0], upDir[1], -upDir[2]);
      fprintf(outfile, "End_Camera\n");
      break;

    case DisplayDevice::PERSPECTIVE:
    default:
      fprintf(outfile, "Camera\n");
      fprintf(outfile, "  Zoom %g\n", (eyePos[2] - zDist) / vSize);
      fprintf(outfile, "  Aspectratio %g\n", 1.0f);
      fprintf(outfile, "  Antialiasing %d\n", aasamples);
      fprintf(outfile, "  Raydepth 8\n");
      fprintf(outfile, "  Center  %g %g %g\n", eyePos[0], eyePos[1], -eyePos[2]);
      fprintf(outfile, "  Viewdir %g %g %g\n", eyeDir[0], eyeDir[1], -eyeDir[2]);
      fprintf(outfile, "  Updir   %g %g %g\n", upDir[0], upDir[1], -upDir[2]);
      fprintf(outfile, "End_Camera\n");
      break;

  }
}

  
void TachyonDisplayDevice::write_lights(void) {  
  // Lights
  int i;  
  int lightcount = 0;
  for (i=0; i<DISP_LIGHTS; i++) {
    if (lightState[i].on) {
      /* give negated light position as the direction vector */
      fprintf(outfile, "Directional_Light Direction %g %g %g ", 
              -lightState[i].pos[0],
              -lightState[i].pos[1],
               lightState[i].pos[2]);
      fprintf(outfile, "Color %g %g %g\n", 
              lightState[i].color[0], lightState[i].color[1], lightState[i].color[2]);
      lightcount++;
    }
  }
  if (lightcount < 1) {
    msgInfo << "Warning: no lights defined in exported scene!!" << sendmsg;
  }
}

void TachyonDisplayDevice::write_materials(void) {
  // background color
  fprintf(outfile, "\nBackground %g %g %g\n", 
          backColor[0], backColor[1], backColor[2]);

  if (cueingEnabled) {
    switch (cueMode) {
      case CUE_LINEAR:
        fprintf(outfile, 
          "FOG LINEAR START %g END %g DENSITY %g COLOR %g %g %g\n", 
          get_cue_start(), get_cue_end(), get_cue_density(), 
          backColor[0], backColor[1], backColor[2]);
        break;
 
      case CUE_EXP:
        fprintf(outfile,
          "FOG EXP START %g END %g DENSITY %g COLOR %g %g %g\n", 
          0.0, get_cue_end(), get_cue_density(), 
          backColor[0], backColor[1], backColor[2]);
        break;
 
      case CUE_EXP2:
        fprintf(outfile, 
          "FOG EXP2 START %g END %g DENSITY %g COLOR %g %g %g\n", 
          0.0, get_cue_end(), get_cue_density(), 
          backColor[0], backColor[1], backColor[2]);
        break;

      case NUM_CUE_MODES:
        // this should never happen
        break;
    }
  } 
}

void TachyonDisplayDevice::write_cindexmaterial(int cindex, int material) {
  write_colormaterial((float *) &matData[cindex], material);
}

// XXX ignores material parameter, may need to improve this..
void TachyonDisplayDevice::write_colormaterial(float *rgb, int /* material */) {
  fprintf(outfile, "Texture\n");
  if (materials_on) {
    fprintf(outfile, "  Ambient %g Diffuse %g Specular %g Opacity %g\n",
            mat_ambient, mat_diffuse, 0.0, mat_opacity);
  } else {
    fprintf(outfile, "  Ambient %g Diffuse %g Specular %g Opacity %g\n",
            1.0, 0.0, 0.0, mat_opacity);
  }
  
  if (mat_outline > 0.0) {
    fprintf(outfile, "  Outline %g Outline_Width %g ", 
            mat_outline, mat_outlinewidth);
  }
  fprintf(outfile, "  Phong Plastic %g Phong_size %g ", mat_specular, 
          mat_shininess);
  fprintf(outfile, "Color %g %g %g ", rgb[0], rgb[1], rgb[2]);

#if 0
  fprintf(outfile, "TexFunc 0\n\n");
#else
  if (!involtex) {
    fprintf(outfile, "TexFunc 0\n\n");
  } else {
    float voluaxs[3];           ///< volume texture coordinate generation
    float volvaxs[3];           ///< parameters in world coordinates
    float volwaxs[3];
    float volcent[3];

    // transform the y/v/w texture coordinate axes from molecule
    // coordinates into world coordinates
    (transMat.top()).multplaneeq3d(xplaneeq, voluaxs);
    (transMat.top()).multplaneeq3d(yplaneeq, volvaxs);
    (transMat.top()).multplaneeq3d(zplaneeq, volwaxs);

    // undo the scaling operation applied by the transformation
    float invscale = 1.0f / scale_radius(1.0f); 
    int i;
    for (i=0; i<3; i++) {
      voluaxs[i] *= invscale;
      volvaxs[i] *= invscale;
      volwaxs[i] *= invscale;
    }

    // compute the volume origin in molecule coordinates by 
    // reverting the scaling factor that was previously applied
    // to the texture plane equation
    float volorgmol[3] = {0,0,0};
    volorgmol[0] = -xplaneeq[3] / norm(xplaneeq);
    volorgmol[1] = -yplaneeq[3] / norm(yplaneeq);
    volorgmol[2] = -zplaneeq[3] / norm(zplaneeq);

    // transform the volume origin into world coordinates
    (transMat.top()).multpoint3d(volorgmol, volcent);

    // emit the texture to the scene file
    fprintf(outfile, "\n  TexFunc  10  ::VMDVolTex%d\n", voltexID);
    fprintf(outfile, "  Center %g %g %g\n", volcent[0], volcent[1], -volcent[2]);
    fprintf(outfile, "  Rotate 0 0 0\n");
    fprintf(outfile, "  Scale  1 1 1\n");
    fprintf(outfile, "  Uaxis %g %g %g\n", voluaxs[0], voluaxs[1], -voluaxs[2]);
    fprintf(outfile, "  Vaxis %g %g %g\n", volvaxs[0], volvaxs[1], -volvaxs[2]);
    fprintf(outfile, "  Waxis %g %g %g\n", volwaxs[0], volwaxs[1], -volwaxs[2]);
    fprintf(outfile, "\n");
  }
#endif
}




