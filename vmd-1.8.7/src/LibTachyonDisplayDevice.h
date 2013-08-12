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
 *	$RCSfile: LibTachyonDisplayDevice.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.22 $	$Date: 2009/05/27 16:11:55 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 *  Tachyon Parallel / Multiprocessor Ray Tracer
 *
 ***************************************************************************/

#ifndef LIBTACHYONDISPLAYDEVICE
#define LIBTACHYONDISPLAYDEVICE

#include <stdio.h>
#include "FileRenderer.h"
#include "tachyon.h"       // main Tachyon library header

/// FileRenderer subclass renders via compiled-in Tachyon ray tracing engine
class LibTachyonDisplayDevice : public FileRenderer {
private:
  SceneHandle rtscene;        ///< handle to the ray tracer library
  int inclipgroup;            ///< whether a clipping group is currently active
  int involtex;               ///< volume texturing is enabled
  int voltexID;               ///< current volume texturing ID
  float xplaneeq[4];          ///< volumetric texture plane equations
  float yplaneeq[4];
  float zplaneeq[4];
  rt_timerhandle buildtime;   ///< timer handle
  rt_timerhandle rendertime;  ///< timer handle

  void reset_vars(void);      ///< reset internal state betwen renders
  void * tex_cindexmaterial(int, int); ///< calc texture
  void * tex_colormaterial(float *rgb, int); ///< calc texture
  void write_camera(void);    ///< write out camera specification
  void write_lights(void);    ///< write out the active lights
  void write_materials(void); ///< write out colors, textures, materials etc.

protected:
  // assorted graphics functions
  void cylinder(float *, float *, float rad, int filled);
  void line(float *xyz1, float *xyz2);
  void point(float *xyz);
  void sphere(float *xyzr);
  virtual void square(float * norm, float * a, float * b,
                      float * c, float * d) {
    // draw as two triangles, with LH-corrected winding order
    triangle(a, c, b, norm, norm, norm);
    triangle(a, d, c, norm, norm, norm);
  }
  void triangle(const float *, const float *, const float *, 
                const float *, const float *, const float *);
  void tricolor(const float * xyz1, const float * xyz2, const float * xyz3,
                const float * n1,   const float * n2,   const float * n3,
                const float *c1,    const float *c2,    const float *c3);
  void tristrip(int numverts, const float *cnv, int numstrips, 
                const int *vertsperstrip, const int *facets);

  // define a volumetric texture map
  virtual void define_volume_texture(int ID, int xs, int ys, int zs,
                                     const float *xplaneeq,
                                     const float *yplaneeq,
                                     const float *zplaneeq,
                                     unsigned char *texmap);

  // enable volumetric texturing, either in "replace" or "modulate" mode
  virtual void volume_texture_on(int texmode);

  // disable volumetric texturing
  virtual void volume_texture_off(void);

  // begin a group of objects to be clipped by the same set of
  // clipping planes
  void start_clipgroup(void);
  void end_clipgroup(void);
   
public: 
  LibTachyonDisplayDevice(void);           // constructor
  virtual ~LibTachyonDisplayDevice(void);  // destructor
  virtual int open_file(const char *filename);
  virtual void close_file(void);
  void write_header(void); 
  void write_trailer(void);
}; 

#endif

