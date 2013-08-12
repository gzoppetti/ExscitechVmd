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
 *	$RCSfile: FileRenderer.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.91 $	$Date: 2009/05/17 06:37:38 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The FileRenderer class implements the data and functions needed to 
 * render a scene to a file in some format (postscript, raster3d, etc.)
 *
 ***************************************************************************/
#ifndef FILERENDERER_H
#define FILERENDERER_H

#include <stdio.h>

#include "DisplayDevice.h"
#include "Scene.h"
#include "NameList.h"
#include "Inform.h"

#define FILERENDERER_NOWARNINGS    0
#define FILERENDERER_NOMISCFEATURE 1
#define FILERENDERER_NOCLIP        2
#define FILERENDERER_NOCUEING      4
#define FILERENDERER_NOTEXTURE     8
#define FILERENDERER_NOGEOM       16
#define FILERENDERER_NOTEXT       32

/// This is the base class for all the renderers that go to a
/// file and are on the render list.  There are five operations
/// available to the outside world
class FileRenderer : public DisplayDevice {
protected:
  // default parameters for this instance; these don't ever change.
  char *publicName, *defaultFilename, *defaultCommandLine;
  char *execCmd;     ///< current version of the post-render command
  FILE *outfile;     ///< the current file
  int isOpened;      ///< is the file opened correctly
  char *my_filename; ///< the current filename
  int has_aa;        ///< supports antialiasing; off by default
  int aasamples;     ///< antialiasing samples, -1 if unsupported.
  int aosamples;     ///< ambient occlusion samples, -1 if unsupported.
  int has_imgsize;   ///< True if the renderer can produce an arbitrary-sized
                     ///< image; false by default.
  int warningflags;  ///< If set, emit a warning message that this
                     ///< subclass doesn't support all of the render features
                     ///< in use by the current scene
  int imgwidth, imgheight;  ///< desired size of image
  float aspectratio;        ///< Desired aspect ratio.
  NameList<int> formats;    ///< Output formats supported by this renderer
  int curformat;     ///< Currently selected format.

  /// Renderer-specific function to update execCmd based on the current state
  /// of aasamples, image size, etc.  Default implementation is to do nothing.
  virtual void update_exec_cmd() {}

  /// light state, passed to renderer before render commands are executed.
  struct LightState {
    float color[3];             ///< RGB color of the light
    float pos[3];               ///< Position (or direction) of the light 
    int on;                     ///< on/off state of light
  };

  LightState lightState[DISP_LIGHTS]; ///< state of all lights

  /// color state, copied into here when do_use_colors is called
  float matData[MAXCOLORS][3];
  virtual void do_use_colors();

  /// background color, copied into here with set_background is called
  float backColor[3];
 
public:
  /// create the renderer; set the 'visible' name for the renderer list
  FileRenderer(const char *public_name, const char *default_file_name,
	       const char *default_command_line);
  virtual ~FileRenderer(void);

  const char *visible_name(void) const { return publicName;}
  const char *default_filename(void) const {return defaultFilename;}
  const char *default_exec_string(void) const {return defaultCommandLine;}
  const char *saved_exec_string(void) const { return execCmd; }

  void set_exec_string(const char *);

  /// Supports anti-aliasing?
  int has_antialiasing() const { return has_aa; }

  /// Get/set the AA level; return the new value.  Must be non-negative.
  int set_aasamples(int newval) {
    if (has_aa && (newval >= 0)) {
      aasamples = newval;
      update_exec_cmd();
    }
    return aasamples;
  }

  /// Get/set the AO samples; return the new value.  Must be non-negative.
  int set_aosamples(int newval) {
    if (newval >= 0) {
      aosamples = newval;
      update_exec_cmd();
    }
    return aosamples;
  }

  /// Supports arbitrary image size?
  int has_imagesize() const { return has_imgsize; }

  /// Get/set the image size.   Return success and places the current values in
  /// the passed-in pointers.  May fail if the renderer is not able to specify 
  /// the image size (e.g. snapshot).  Passing 0,0 just returns the current 
  /// values.
  int set_imagesize(int *w, int *h);

  /// Set the aspect ratio.  Negative values ignored.  Returns the new value.
  /// Also updates image size if it has been set.
  float set_aspectratio(float aspect);
  
  /// Number of output formats
  int numformats() const { return formats.num(); }
  
  /// get/set formats
  const char *format(int i) const { return formats.name(i); }
  const char *format() const { return formats.name(curformat); }
  int set_format(const char *format) {
    int ind = formats.typecode(format);
    if (ind < 0) return FALSE;
    if (curformat != ind) {
      curformat = ind;
      update_exec_cmd();
    }
    return TRUE;
  }

  /// copy in the background color
  virtual void set_background(const float *);

  /// open the file; don't write the header info
  /// return TRUE if opened okay
  /// if file already opened, complain, and close previous file
  /// this will also reset the state variables
  virtual int open_file(const char *filename);

  virtual int do_define_light(int n, float *color, float *position);
  virtual int do_activate_light(int n, int turnon);

private:
  void reset_state(void);
  int sph_nverts;   ///< data for tesselating spheres with triangles
  float *sph_verts; ///< data for tesselating spheres with triangles

protected:
  /// write the header info.  This is an alias for prepare3D
  virtual void write_header(void) {};

public:
  virtual int prepare3D(int); 
  virtual void render(const VMDDisplayList *); // render the display list

protected:
  /// write any trailer info.  This is called by update
  virtual void write_trailer(void) {};

  /// close the file.  This is called by update, and exists
  /// due to symmetry.  Also, is called for case when open is
  /// called when a file was already open.
  virtual void close_file(void);

public:
  /// don't need to override this (unless you want to do so)
  virtual void update(int) {
    if (isOpened) {
      write_trailer();
      close_file();
      isOpened = FALSE;
  
      // Emit any pending warning messages for missing or unsupported
      // geometric primitives.
      if (warningflags & FILERENDERER_NOCLIP)
        msgWarn << "User-defined clipping planes not exported for this renderer" << sendmsg;

      if (warningflags & FILERENDERER_NOTEXT)
        msgWarn << "Text not exported for this renderer" << sendmsg;

      if (warningflags & FILERENDERER_NOTEXTURE)
        msgWarn << "Texture mapping not exported for this renderer" << sendmsg;

      if (warningflags & FILERENDERER_NOCUEING)
        msgWarn << "Depth cueing not exported for this renderer" << sendmsg;

      if (warningflags & FILERENDERER_NOGEOM)
        msgWarn << "One or more geometry types not exported for this renderer" << sendmsg;

      if (warningflags != FILERENDERER_NOWARNINGS)
        msgWarn << "Unimplemented features may negatively affect the appearance of the scene" << sendmsg;
    }
  }

protected:
  /// pointer to data block (for indexed object types)
  float *dataBlock;
  
  ///////////// Information about the current state //////
  // (for those that do not want to take care of it themselves)
  // the 'super_' version is called by render to set the matrix.  It
  // then calls the non-super version
  Stack<Matrix4> transMat;
  void super_load(float *cmdptr);
  virtual void load(const Matrix4& /*mat*/) {}
  void super_multmatrix(const float *cmdptr);
  virtual void multmatrix(const Matrix4& /*mat*/) {}
  void super_translate(float *cmdptr);
  virtual void translate(float /*x*/, float /*y*/, float /*z*/) {}
  void super_rot(float *cmdptr);
  virtual void rot(float /*ang*/, char /*axis*/) {}
  void super_scale(float *cmdptr);
  void super_scale(float);
  virtual void scale(float /*scalex*/, float /*scaley*/, 
		     float /*scalez*/) {}
  float scale_radius(float);        ///< compute the current scaling factor

  // change the color definitions
  int colorIndex;                   ///< active color index
  void super_set_color(int index);  ///< only calls set_color when index changes
  virtual void set_color(int) {}    ///< set the color index
  
  /// compute nearest index in matData using given rgb value
  /// XXX We shouldn't be doing this; a better approach would be to store the
  /// new color in the matData color table and return the new index, rather 
  /// than trying to match a 17 color palette.   
  int nearest_index(float r, float g, float b) const;

  // change the material definition
  int materialIndex;                    ///< active material index
  float mat_ambient;                    ///< active ambient value
  float mat_diffuse;                    ///< active diffuse value
  float mat_specular;                   ///< active specular value
  float mat_shininess;                  ///< active shininess value
  float mat_opacity;                    ///< active opacity value
  float mat_outline;                    ///< active outline factor
  float mat_outlinewidth;               ///< active outline width
  void super_set_material(int index);   ///< only call set_material on idx chg
  virtual void set_material(int) {}     ///< change material index 

  float clip_center[VMD_MAX_CLIP_PLANE][3]; ///< clipping plane center
  float clip_normal[VMD_MAX_CLIP_PLANE][3]; ///< clipping plane normal
  float clip_color[VMD_MAX_CLIP_PLANE][3];  ///< clipping plane CSG color
  int clip_mode[VMD_MAX_CLIP_PLANE];        ///< clipping plane mode

  virtual void start_clipgroup();       ///< emit clipping plane group
  virtual void end_clipgroup() {}       ///< terminate clipping plane group

  // change the line definitions
  int lineWidth, lineStyle, pointSize;
  virtual void set_line_width(int new_width) {
    lineWidth = new_width;
  }
  virtual void set_line_style(int /*new_style*/) {}  ///< called by super

  // change the sphere definitions
  int sphereResolution, sphereStyle;
  virtual void set_sphere_res(int /*res*/) {}        ///< called by super
  virtual void set_sphere_style(int /*style*/) {}    ///< called by super

  int materials_on;
  void super_materials(int on_or_off);
  virtual void activate_materials(void) {}           ///< if previous is TRUE
  virtual void deactivate_materials(void) {}         ///< if super is FALSE
  

  ////////////////////// various virtual generic graphics commands

  // single-radius cones (pointy top)
  virtual void cone(float * xyz1, float * xyz2, float radius, int resolution) { 
    cone(xyz1, xyz2, radius, 0.0, resolution);
  }


  // two radius cones
  virtual void cone(float * /*xyz1*/, float * /*xyz2*/, 
                    float /* radius*/, float /* radius2 */, int /*resolution*/);


  // cylinders, with optional caps
  virtual void cylinder(float * base, float * apex, float radius, int filled);


  // simple lines
  virtual void line(float * a, float * b);


  // simple points
  virtual void point(float * xyz) {
    float xyzr[4];
    vec_copy(xyzr, xyz);
    xyzr[3] = lineWidth * 0.002f; // hack for renderers that don't have points
  }


  // simple sphere
  virtual void sphere(float * xyzr);


  // quadrilateral
  virtual void square(float * norm, float * a, float * b, 
		      float * c, float * d) {
    // draw as two triangles, with correct winding order etc
    triangle(a, b, c, norm, norm, norm);
    triangle(a, c, d, norm, norm, norm);
  }


  // single color triangle with interpolated surface normals
  virtual void triangle(const float * /*xyz1*/, const float * /*xyz2*/, const float * /*xyz3*/, 
                        const float * /*n1*/, const float * /*n2*/, const float * /*n3*/) {
    warningflags |= FILERENDERER_NOGEOM; // no triangles written
  }


  // triangle with interpolated surface normals and vertex colors
  virtual void tricolor(const float * xyz1, const float * xyz2, const float * xyz3, 
                        const float * n1, const float * n2, const float * n3,
                        const float *c1, const float *c2, const float *c3) {
    int index = 1;
    float r, g, b;
    r = (c1[0] + c2[0] + c3[0]) / 3.0f; // average three vertex colors 
    g = (c1[1] + c2[1] + c3[1]) / 3.0f;
    b = (c1[2] + c2[2] + c3[2]) / 3.0f;

    index = nearest_index(r,g,b); // lookup nearest color here.
    super_set_color(index); // use the closest color

    triangle(xyz1, xyz2, xyz3, n1, n2, n3); // draw a regular triangle   
  }


  // triangle mesh built from a vertex array and facet vertex index arrays
  virtual void trimesh(int /* numverts */, float * cnv, 
                       int numfacets, int * facets) { 
    int i;
    for (i=0; i<numfacets*3; i+=3) {
      int v0 = facets[i    ] * 10;
      int v1 = facets[i + 1] * 10;
      int v2 = facets[i + 2] * 10;
      tricolor(cnv + v0 + 7, // vertices 0, 1, 2
               cnv + v1 + 7, 
               cnv + v2 + 7,
               cnv + v0 + 4, // normals 0, 1, 2
               cnv + v1 + 4, 
               cnv + v2 + 4,
               cnv + v0,     // colors 0, 1, 2
               cnv + v1, 
               cnv + v2);
    }           
  }


  // triangle strips built from a vertex array and vertex index arrays
  virtual void tristrip(int /* numverts */, const float * cnv, 
                        int numstrips, const int *vertsperstrip, 
                        const int *facets) { 
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
  }


  // define a volumetric texture map
  virtual void define_volume_texture(int ID, int xs, int ys, int zs,
                                     const float *xplaneeq, 
                                     const float *yplaneeq,
                                     const float *zplaneeq,
                                     unsigned char *texmap) {
    warningflags |= FILERENDERER_NOTEXTURE;
  }


  // enable volumetric texturing, either in "replace" or "modulate" mode
  virtual void volume_texture_on(int texmode) {
    warningflags |= FILERENDERER_NOTEXTURE;
  }


  // disable volumetric texturing
  virtual void volume_texture_off(void) {
    warningflags |= FILERENDERER_NOTEXTURE;
  }


  // wire mesh built from a vertex array and an vertex index array
  virtual void wiremesh(int /* numverts */, float * cnv, 
                       int numlines, int * lines) { 
    int i;
    int index = 1;

    for (i=0; i<numlines; i++) {
      float r, g, b;
      int ind = i * 2;
      int v0 = lines[ind    ] * 10;
      int v1 = lines[ind + 1] * 10;

      r = cnv[v0 + 0] + cnv[v1 + 0] / 2.0f;
      g = cnv[v0 + 1] + cnv[v1 + 1] / 2.0f;
      b = cnv[v0 + 2] + cnv[v1 + 2] / 2.0f;

      index = nearest_index(r,g,b); // lookup nearest color here.
      super_set_color(index); // use the closest color

      line(cnv + v0 + 7, cnv + v1 + 7); 
    }           
  }


  virtual void text(const char *);
  virtual void comment(const char *) {}

  float textpos[3];
  void super_text_position(float x, float y, float z);
  virtual void text_position(float /* x*/, float /* y*/, float /* z*/) {
    warningflags |= FILERENDERER_NOTEXT;
  }

  /// here for completeness, only VRML or 'token' renderers would likely use it
  virtual void pick_point(float * /*xyz*/, int /*id*/) {}

};

#endif

