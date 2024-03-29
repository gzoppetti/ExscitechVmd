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
*      $RCSfile: LibTachyonDisplayDevice.C,v $
*      $Author: johns $        $Locker:  $               $State: Exp $
*      $Revision: 1.55 $        $Date: 2009/06/01 07:05:20 $
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
#include "LibTachyonDisplayDevice.h"
#include "Matrix4.h"
#include "DispCmds.h"
#include "Inform.h"
#include "utilities.h"
#include "config.h"     // needed for default image viewer

#if !(((TACHYON_MAJOR_VERSION >= 0) && (TACHYON_MINOR_VERSION > 98)) || ((TACHYON_MAJOR_VERSION == 0) && (TACHYON_MINOR_VERSION == 98) && (TACHYON_PATCH_VERSION >= 7)))
#error "LibTachyonDisplayDevice requires Tachyon version 0.98.7 or higher."
#endif

#define DEFAULT_RADIUS 0.002
#define DASH_LENGTH 0.02

extern "C" {

void vmd_rt_ui_message(int a, char * msg) {
  printf("Tachyon) %s\n", msg);
}

void vmd_rt_ui_progress(int percent) {
  printf("\rTachyon) Rendering progress: %3d%% complete          \r", percent);
  fflush(stdout);
}

}

///////////////////////// constructor and destructor
LibTachyonDisplayDevice::LibTachyonDisplayDevice() : FileRenderer ("TachyonInternal", "plot.tga", DEF_VMDIMAGEVIEWER) { 
  reset_vars();  // initialize internal state
  rt_initialize(0, NULL); // init scene-independent parts of Tachyon library

//  rt_set_ui_message(vmd_rt_ui_message);
  rt_set_ui_progress(vmd_rt_ui_progress);

  // Add supported file formats
  formats.add_name("Auto", 0);
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
  curformat = 0;
}
        
LibTachyonDisplayDevice::~LibTachyonDisplayDevice(void) { 
  rt_finalize(); // shut down Tachyon library
}


///////////////////////// private  routines

// reset internal state between renders
void LibTachyonDisplayDevice::reset_vars(void) {
  inclipgroup = 0; // not currently in a clipping group
  involtex = 0;    // volume texturing disabled
  voltexID = -1;   // invalid texture ID
  memset(xplaneeq, 0, sizeof(xplaneeq));
  memset(yplaneeq, 0, sizeof(xplaneeq));
  memset(zplaneeq, 0, sizeof(xplaneeq));
}


///////////////////////// protected nonvirtual routines

// draw a point
void LibTachyonDisplayDevice::point(float * spdata) {
  float vec[3];
  void *tex;
  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);

  // draw the sphere
  tex=tex_cindexmaterial(colorIndex, materialIndex);
  rt_sphere(rtscene, tex, 
            rt_vector(vec[0], vec[1], -vec[2]),
            float(lineWidth)*DEFAULT_RADIUS);
}

// draw a sphere
void LibTachyonDisplayDevice::sphere(float * spdata) {
  float vec[3];
  float radius;
  void *tex;
    
  // transform the world coordinates
  (transMat.top()).multpoint3d(spdata, vec);
  radius = scale_radius(spdata[3]);
   
  // draw the sphere
  tex=tex_cindexmaterial(colorIndex, materialIndex);
  rt_sphere(rtscene, tex, 
            rt_vector(vec[0], vec[1], -vec[2]),
            radius);
}

// draw a line (cylinder) from a to b
void LibTachyonDisplayDevice::line(float *a, float*b) {
  int i, j, test;
  float dirvec[3], unitdirvec[3];
  float from[3], to[3], tmp1[3], tmp2[3];
  float len;
  void *tex;
  
  if (lineStyle == ::SOLIDLINE) {
    // transform the world coordinates
    (transMat.top()).multpoint3d(a, from);
    (transMat.top()).multpoint3d(b, to);
    
    // draw the cylinder
    tex=tex_cindexmaterial(colorIndex, materialIndex);
    rt_fcylinder(rtscene, tex, 
                 rt_vector(from[0], from[1], -from[2]),
                 // rt_vector(to[0], to[1], -to[2]),
                 rt_vector(to[0]-from[0], to[1]-from[1], -to[2]+from[2]),
                 float(lineWidth)*DEFAULT_RADIUS);


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
      tex=tex_cindexmaterial(colorIndex, materialIndex);
      rt_fcylinder(rtscene, tex, 
                   rt_vector(from[0], from[1], -from[2]),
                   // rt_vector(to[0], to[1], -to[2]),
                   rt_vector(to[0]-from[0], to[1]-from[1], -to[2]+from[2]),
                   float(lineWidth)*DEFAULT_RADIUS);
      i++;
    }
  } else {
    msgErr << "LibTachyonDisplayDevice: Unknown line style " 
           << lineStyle << sendmsg;
  }
}




// draw a cylinder
void LibTachyonDisplayDevice::cylinder(float *a, float *b, float r, int filled) {
  float from[3], to[3], norm[3];
  float radius;
  filled = filled;  
  void * tex;

  // transform the world coordinates
  (transMat.top()).multpoint3d(a, from);
  (transMat.top()).multpoint3d(b, to);
  radius = scale_radius(r);
   
  // draw the cylinder
  tex=tex_cindexmaterial(colorIndex, materialIndex);
  rt_fcylinder(rtscene, tex, 
               rt_vector(from[0], from[1], -from[2]),
               // rt_vector(to[0], to[1], -to[2]),
               rt_vector(to[0]-from[0], to[1]-from[1], -to[2]+from[2]),
               radius);

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
      rt_ring(rtscene, tex,
              rt_vector(from[0], from[1], -from[2]),
              rt_vector(norm[0], norm[1], -norm[2]),
              0.0, radius);
    }
  
    if (filled & CYLINDER_LEADINGCAP) {
      rt_ring(rtscene, tex,
              rt_vector(to[0], to[1], -to[2]),
              rt_vector(-norm[0], -norm[1],  norm[2]),
              0.0, radius);
    }
  }
}

// draw a triangle
void LibTachyonDisplayDevice::triangle(const float *a, const float *b, const float *c, const float *n1, const float *n2, const float *n3) {
  float vec1[3], vec2[3], vec3[3];
  float norm1[3], norm2[3], norm3[3];
  void *tex;
  
  // transform the world coordinates
  (transMat.top()).multpoint3d(a, vec1);
  (transMat.top()).multpoint3d(b, vec2);
  (transMat.top()).multpoint3d(c, vec3);

  // and the normals
  (transMat.top()).multnorm3d(n1, norm1);
  (transMat.top()).multnorm3d(n2, norm2);
  (transMat.top()).multnorm3d(n3, norm3);

  // draw the triangle
  tex=tex_cindexmaterial(colorIndex, materialIndex);
  rt_stri(rtscene, tex, 
          rt_vector(vec1[0], vec1[1], -vec1[2]),
          rt_vector(vec2[0], vec2[1], -vec2[2]),
          rt_vector(vec3[0], vec3[1], -vec3[2]),
          rt_vector(-norm1[0], -norm1[1], norm1[2]),
          rt_vector(-norm2[0], -norm2[1], norm2[2]),
          rt_vector(-norm3[0], -norm3[1], norm3[2]));
}


// draw triangle with per-vertex colors
void LibTachyonDisplayDevice::tricolor(const float * xyz1, const float * xyz2, const float * xyz3,
                                       const float * n1,   const float * n2,   const float * n3,
                                       const float *c1, const float *c2, const float *c3) {
  float vec1[3], vec2[3], vec3[3];
  float norm1[3], norm2[3], norm3[3];
  float rgb[3];
  void *tex;

  // transform the world coordinates
  (transMat.top()).multpoint3d(xyz1, vec1);
  (transMat.top()).multpoint3d(xyz2, vec2);
  (transMat.top()).multpoint3d(xyz3, vec3);

  // and the normals
  (transMat.top()).multnorm3d(n1, norm1);
  (transMat.top()).multnorm3d(n2, norm2);
  (transMat.top()).multnorm3d(n3, norm3);

  rgb[0] = 0.0;
  rgb[1] = 0.0;
  rgb[2] = 0.0;

  // draw the triangle
  tex=tex_colormaterial(rgb, materialIndex);

  if (!involtex) {
    rt_vcstri(rtscene, tex,
              rt_vector(vec1[0], vec1[1], -vec1[2]),
              rt_vector(vec2[0], vec2[1], -vec2[2]),
              rt_vector(vec3[0], vec3[1], -vec3[2]),
              rt_vector(-norm1[0], -norm1[1], norm1[2]),
              rt_vector(-norm2[0], -norm2[1], norm2[2]),
              rt_vector(-norm3[0], -norm3[1], norm3[2]),
              rt_color(c1[0], c1[1], c1[2]),
              rt_color(c2[0], c2[1], c2[2]),
              rt_color(c3[0], c3[1], c3[2]));
  } else {
    rt_stri(rtscene, tex,
            rt_vector(vec1[0], vec1[1], -vec1[2]),
            rt_vector(vec2[0], vec2[1], -vec2[2]),
            rt_vector(vec3[0], vec3[1], -vec3[2]),
            rt_vector(-norm1[0], -norm1[1], norm1[2]),
            rt_vector(-norm2[0], -norm2[1], norm2[2]),
            rt_vector(-norm3[0], -norm3[1], norm3[2]));
  }
}


// draw triangle strips with per-vertex colors
void LibTachyonDisplayDevice::tristrip(int numverts, const float *cnv, 
                              int numstrips, const int *vertsperstrip, 
                              const int *facets) {
#if 1
  // render triangle strips one triangle at a time
  // triangle winding order is:
  //   v0, v1, v2, then v2, v1, v3, then v2, v3, v4, etc.
  int strip, t, v = 0;
  int stripaddr[2][3] = { {0, 1, 2}, {1, 0, 2} };

  // loop over all of the triangle strips
  for (strip=0; strip < numstrips; strip++) {
    // loop over all triangles in this triangle strip
    for (t = 0; t < (vertsperstrip[strip] - 2); t++) {
      // render one triangle, using lookup table to fix winding order
      int v0 = facets[v + (stripaddr[t & 0x01][0])] * 10;
      int v1 = facets[v + (stripaddr[t & 0x01][1])] * 10;
      int v2 = facets[v + (stripaddr[t & 0x01][2])] * 10;

      tricolor(cnv + v0 + 7, // vertices 0, 1, 2
               cnv + v1 + 7,
               cnv + v2 + 7,
               cnv + v0 + 4, // normals 0, 1, 2
               cnv + v1 + 4,
               cnv + v2 + 4,
               cnv + v0,     // colors 0, 1, 2
               cnv + v1,
               cnv + v2);
      v++; // move on to next vertex
    }
    v+=2; // last two vertices are already used by last triangle
  }
#else
  int i;
  float vec1[3], vec2[3], vec3[3];
  float norm1[3], norm2[3], norm3[3];
  float rgb[3];
  void *tex;

  Matrix4 topMatrix = transMat.top();

  /* transform all vertices and normals, copy colors as-is */
  float *tcnv = new float[numverts * 10];

  for (i=0; i<numverts; i++) {
    int addr = i * 10;
    int j;

    for (j=0; j<3; j++)
      tcnv[addr + j] = cnv[addr + j];

    topMatrix.multnorm3d(&cnv[addr + 4], &tcnv[addr + 4]);
    tcnv[addr + 4] = -tcnv[addr + 4];
    tcnv[addr + 5] = -tcnv[addr + 5];
    topMatrix.multpoint3d(&cnv[addr + 7], &tcnv[addr + 7]);
    tcnv[addr + 9] = -tcnv[addr + 9];
  }

  rgb[0] = 0.0;
  rgb[1] = 0.0;
  rgb[2] = 0.0;

  // draw the triangle strips
  tex=tex_colormaterial(rgb, materialIndex);
  rt_tristripscnv3fv(rtscene, tex, numverts, tcnv, 
                     numstrips, vertsperstrip, facets);

  delete [] tcnv;
#endif
}



///////////////////// public virtual routines

int LibTachyonDisplayDevice::open_file(const char *filename) {
  my_filename = stringdup(filename);
  isOpened = TRUE;
  return TRUE;
}

void LibTachyonDisplayDevice::close_file(void) {
  outfile = NULL;
  delete [] my_filename;
  my_filename = NULL;
  isOpened = FALSE;
}

static int checkfileextension(const char * s, const char * extension) {
  int sz, extsz;
  sz = strlen(s);
  extsz = strlen(extension);

  if (extsz > sz)
    return 0;

  if (!strupncmp(s + (sz - extsz), extension, extsz)) {
    return 1;
  }

  return 0;
}

// initialize the file for output
void LibTachyonDisplayDevice::write_header() {
  // NOTE: the vmd variable "Aspect" has absolutely *nothing* to do
  //       with aspect ratio correction, it is only the ratio of the
  //       width of the graphics window to its height, and so it should
  //       be used only to cause the ray tracer to generate a similarly
  //       proportioned image.

  buildtime = rt_timer_create();
  rendertime = rt_timer_create();

  rt_timer_start(buildtime);
  rtscene = rt_newscene();
  rt_outputfile(rtscene, my_filename);      // get filename from base class

  switch (curformat) {
    case 0: // autodetermine...
      // set appropriate image file format
      if (checkfileextension(my_filename, ".bmp")) {
        rt_outputformat(rtscene, RT_FORMAT_WINBMP);
      } else if (checkfileextension(my_filename, ".ppm")) {
        rt_outputformat(rtscene, RT_FORMAT_PPM);
      } else if (checkfileextension(my_filename, ".psd")) {
        rt_outputformat(rtscene, RT_FORMAT_PSD48);
      } else if (checkfileextension(my_filename, ".rgb")) {
        rt_outputformat(rtscene, RT_FORMAT_SGIRGB);
      } else if (checkfileextension(my_filename, ".tga")) {
        rt_outputformat(rtscene, RT_FORMAT_TARGA);
      } else {
#if defined(_MSC_VER) || defined(WIN32) || defined(MINGW)
        msgErr << "Unrecognized image file extension, writing Windows Bitmap file."
               << sendmsg;
        rt_outputformat(rtscene, RT_FORMAT_WINBMP);
#else
        msgErr << "Unrecognized image file extension, writing Targa file."
               << sendmsg;
        rt_outputformat(rtscene, RT_FORMAT_TARGA);
#endif
      }
      break;

    case 1:
      rt_outputformat(rtscene, RT_FORMAT_WINBMP);
      break;

    case 2:    
      rt_outputformat(rtscene, RT_FORMAT_PPM);
      break;

    case 3:
      rt_outputformat(rtscene, RT_FORMAT_PPM48);

    case 4:
      rt_outputformat(rtscene, RT_FORMAT_PSD48);

    case 5:
      rt_outputformat(rtscene, RT_FORMAT_SGIRGB);

    case 6:
    default:
      rt_outputformat(rtscene, RT_FORMAT_TARGA);
  }


  rt_resolution(rtscene,  (int) xSize, (int) ySize);

  // use opacity post-multiply transparency mode for output that closely 
  // matches what user's see in the OpenGL window in VMD.
  rt_trans_mode(rtscene, RT_TRANS_VMD);     

  // use planar fog mode (rather than the more physically correct radial fog)
  // for output that closely matches what user's see in the VMD OpenGL window.
  rt_fog_rendering_mode(rtscene, RT_FOG_VMD);

  write_camera();    // has to be first thing in the file. 
  write_lights();    // could be anywhere.
  write_materials(); // has to be before objects that use them.

  // render with/without shadows
  if (shadows_enabled() || ao_enabled()) {
    if (shadows_enabled() && !ao_enabled())
      msgInfo << "Shadow rendering enabled." << sendmsg;
 
    rt_shadermode(rtscene, RT_SHADER_FULL);   // full shading mode required
  } else {
    rt_shadermode(rtscene, RT_SHADER_MEDIUM); // disable shadows by default
  }

  // render with ambient occlusion, but only if shadows are also enabled
  if (ao_enabled()) {
    apicolor skycol;
    skycol.r = get_ao_ambient();
    skycol.g = get_ao_ambient();
    skycol.b = get_ao_ambient();

    msgInfo << "Ambient occlusion rendering enabled." << sendmsg;
    rt_rescale_lights(rtscene, get_ao_direct());
    rt_ambient_occlusion(rtscene, aosamples, skycol);
  } 
}


void LibTachyonDisplayDevice::write_trailer(void){
  rt_timer_stop(buildtime);
  rt_timer_start(rendertime);
  rt_renderscene(rtscene);
  rt_timer_stop(rendertime);
  rt_deletescene(rtscene);
   
  msgInfo << "Tachyon: preprocessing time " 
          << rt_timer_time(buildtime)  << " sec, render time "
          << rt_timer_time(rendertime) << " sec." << sendmsg;
  rt_timer_destroy(buildtime);
  rt_timer_destroy(rendertime);

  if (inclipgroup) {
    msgErr << "LibTachyonDisplayDevice clip group still active at end of scene" << sendmsg;
  }

  reset_vars(); // reset internal state between renders
}


#if 1
// define a volumetric texture map
void LibTachyonDisplayDevice::define_volume_texture(int ID,
                                                 int xs, int ys, int zs,
                                                 const float *xpq,
                                                 const float *ypq,
                                                 const float *zpq,
                                                 unsigned char *texmap) {
  char texname[1024];
  unsigned char *rgb=NULL;

  voltexID = ID; // remember current texture ID

  // remember texture plane equations
  memcpy(xplaneeq, xpq, sizeof(xplaneeq));
  memcpy(yplaneeq, ypq, sizeof(yplaneeq));
  memcpy(zplaneeq, zpq, sizeof(zplaneeq));
  
  sprintf(texname, "::VMDVolTex%d", voltexID);

  // copy incoming texture map to a new buffer
  // XXX ideally we'd have Tachyon use the existing image
  //     buffer rather than copy it
  rgb = (unsigned char *) malloc(xs * ys * zs * 3);
  memcpy(rgb, texmap, xs * ys * zs * 3);

  // define the image within Tachyon
  rt_define_image(texname, xs, ys, zs, rgb);
}


// enable volumetric texturing, either in "replace" or "modulate" mode
void LibTachyonDisplayDevice::volume_texture_on(int texmode) {
  involtex = 1;
}


// disable volumetric texturing
void LibTachyonDisplayDevice::volume_texture_off(void) {
  involtex = 0;
}

#else

// define a volumetric texture map
void LibTachyonDisplayDevice::define_volume_texture(int ID,
                                                 int xs, int ys, int zs,
                                                 const float *txplaneeq,
                                                 const float *typlaneeq,
                                                 const float *tzplaneeq,
                                                 unsigned char *texmap) {
  warningflags |= FILERENDERER_NOTEXTURE;
}


// enable volumetric texturing, either in "replace" or "modulate" mode
void LibTachyonDisplayDevice::volume_texture_on(int texmode) {
  warningflags |= FILERENDERER_NOTEXTURE;
}


// disable volumetric texturing
void LibTachyonDisplayDevice::volume_texture_off(void) {
  warningflags |= FILERENDERER_NOTEXTURE;
}

#endif






void LibTachyonDisplayDevice::start_clipgroup(void) {
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
    float *planes = (float *) malloc(planesenabled * 4 * sizeof(float));

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

        planes[i * 4    ] = tachyon_clip_normal[0];
        planes[i * 4 + 1] = tachyon_clip_normal[1];
        planes[i * 4 + 2] = -tachyon_clip_normal[2];
        planes[i * 4 + 3] = tachyon_clip_distance;

        rt_clip_fv(rtscene, planesenabled, planes); // add the clipping planes
      }
    }

    free(planes);
  } else {
    inclipgroup = 0; // Not currently in a clipping group
  }
}


void LibTachyonDisplayDevice::end_clipgroup(void) {
  if (inclipgroup) {
    rt_clip_off(rtscene); // disable clipping planes
    inclipgroup = 0;      // we're not in a clipping group anymore 
  }
}


///////////////////// Private routines

void LibTachyonDisplayDevice::write_camera(void) {
  // Camera position
  // Tachyon uses a left-handed coordinate system
  // VMD uses right-handed, so z(Tachyon) = -z(VMD).
  switch (projection()) {
    case DisplayDevice::ORTHOGRAPHIC:
      rt_camera_projection(rtscene, RT_PROJECTION_ORTHOGRAPHIC);
      rt_camera_setup(rtscene,
                     1.0 / (vSize / 2.0),           // zoom
                     1.0f,                          // aspect ratio
                     aasamples,                     // antialiasing
                     8,                             // ray depth
                     rt_vector(eyePos[0], eyePos[1], -eyePos[2]), // camcent
                     rt_vector(eyeDir[0], eyeDir[1], -eyeDir[2]), // camview
                     rt_vector(upDir[0],   upDir[1],  -upDir[2]));   // camup
      break;

    case DisplayDevice::PERSPECTIVE:
    default:
      rt_camera_projection(rtscene, RT_PROJECTION_PERSPECTIVE);
      rt_camera_setup(rtscene,
                     ((eyePos[2] - zDist) / vSize), // zoom
                     1.0f,                          // aspect ratio
                     aasamples,                     // antialiasing
                     8,                             // ray depth
                     rt_vector(eyePos[0], eyePos[1], -eyePos[2]), // camcent
                     rt_vector(eyeDir[0], eyeDir[1], -eyeDir[2]), // camview
                     rt_vector(upDir[0],   upDir[1], -upDir[2])); // camup
      break;
  }
}

  
void LibTachyonDisplayDevice::write_lights(void) {  
  int i;  
  int lightcount = 0;
  for (i=0; i<DISP_LIGHTS; i++) {
    if (lightState[i].on) {
      apitexture tex;
      memset(&tex, 0, sizeof(apitexture));

      tex.col.r=lightState[i].color[0];
      tex.col.g=lightState[i].color[1];
      tex.col.b=lightState[i].color[2];
     
      rt_directional_light(rtscene, 
               rt_texture(rtscene, &tex), 
               /* give negated light position for direction vector... */ 
               rt_vector(-lightState[i].pos[0], 
                         -lightState[i].pos[1], 
                          lightState[i].pos[2]));
 
      lightcount++;
    }
  }
  if (lightcount < 1) {
    msgInfo << "Warning: no lights defined in exported scene!!" << sendmsg;
  }
}

void LibTachyonDisplayDevice::write_materials(void) {
  // background color
  apicolor col;
  col.r = backColor[0];
  col.g = backColor[1];
  col.b = backColor[2];
  rt_background(rtscene, col);

  // set depth cueing parameters
  if (cueingEnabled) {
    switch (cueMode) {
      case CUE_LINEAR:
        rt_fog_mode(rtscene, RT_FOG_LINEAR);
        rt_fog_parms(rtscene, col, get_cue_start(), get_cue_end(), get_cue_density());
        break;
 
      case CUE_EXP:
        rt_fog_mode(rtscene, RT_FOG_EXP);
        rt_fog_parms(rtscene, col, 0.0, get_cue_end(), get_cue_density());
        break;
 
      case CUE_EXP2:
        rt_fog_mode(rtscene, RT_FOG_EXP2);
        rt_fog_parms(rtscene, col, 0.0, get_cue_end(), get_cue_density());
        break;

      case NUM_CUE_MODES:
        // this should never happen
        break;
    } 
  } else {
    rt_fog_mode(rtscene, RT_FOG_NONE);
  }
}


void * LibTachyonDisplayDevice::tex_cindexmaterial(int cindex, int material) {
  float *rgb = (float *) &matData[cindex];
  void *voidtex;

  voidtex = tex_colormaterial(rgb, material);
 
  return voidtex;
}


void * LibTachyonDisplayDevice::tex_colormaterial(float *rgb, int material) {
  apitexture tex;
  void *voidtex;

  memset(&tex, 0, sizeof(apitexture));

  if (materials_on) {
    tex.ambient  = mat_ambient;
    tex.diffuse  = mat_diffuse;
    tex.specular = 0.0;
  } else {
    tex.ambient  = 1.0;
    tex.diffuse  = 0.0;
    tex.specular = 0.0;
  }

  tex.opacity  = mat_opacity;
  tex.col.r =  rgb[0];
  tex.col.g =  rgb[1];
  tex.col.b =  rgb[2];

#if 0
  tex.texturefunc = RT_TEXTURE_CONSTANT;
  voidtex=rt_texture(rtscene, &tex);
#else
  if (!involtex) {
    tex.texturefunc = RT_TEXTURE_CONSTANT;
    voidtex=rt_texture(rtscene, &tex);
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

    tex.texturefunc = RT_TEXTURE_VOLUME_IMAGE;

    sprintf(tex.imap, "::VMDVolTex%d", voltexID);
    tex.ctr.x =  volcent[0];
    tex.ctr.y =  volcent[1];
    tex.ctr.z = -volcent[2];
    tex.rot.x = 0;
    tex.rot.y = 0;
    tex.rot.z = 0;
    tex.scale.x = 1;
    tex.scale.y = 1;
    tex.scale.z = 1;
    tex.uaxs.x =  voluaxs[0];
    tex.uaxs.y =  voluaxs[1];
    tex.uaxs.z = -voluaxs[2];
    tex.vaxs.x =  volvaxs[0];
    tex.vaxs.y =  volvaxs[1];
    tex.vaxs.z = -volvaxs[2];
    tex.waxs.x =  volwaxs[0];
    tex.waxs.y =  volwaxs[1];
    tex.waxs.z = -volwaxs[2];
 
    voidtex=rt_texture(rtscene, &tex);
  }
#endif

  rt_tex_phong(voidtex, mat_specular, mat_shininess, RT_PHONG_PLASTIC);
  rt_tex_outline(voidtex, mat_outline, mat_outlinewidth);
 
  return voidtex;
}


