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
 *	$RCSfile: FileRenderList.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.63 $	$Date: 2009/05/17 06:37:37 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The FileRenderList class maintains a list of available FileRenderer
 * objects
 *
 ***************************************************************************/

#include "config.h"  // create dependency so new compile options cause rebuild
#include "FileRenderList.h"
#include "VMDApp.h"
#include "CmdRender.h"
#include "CommandQueue.h"
#include "Scene.h"
#include "DisplayDevice.h"
#include "TextEvent.h"
#include "Inform.h"
#include <stdlib.h>  // for system()

//
// Supported external rendering programs
//
#include "ArtDisplayDevice.h"         // Art ray tracer
#include "GelatoDisplayDevice.h"      // nVidia Gelato
#include "POV3DisplayDevice.h"        // POVRay 3.x 
#include "PSDisplayDevice.h"          // Postscript
#include "R3dDisplayDevice.h"         // Raster3D
#include "RayShadeDisplayDevice.h"    // Rayshade 4.0 
#include "RadianceDisplayDevice.h"    // Radiance, unknown version 
#include "RenderManDisplayDevice.h"   // RenderMan interface
#include "SnapshotDisplayDevice.h"    // Built-in snapshot capability
#include "STLDisplayDevice.h"         // Stereolithography files
#include "TachyonDisplayDevice.h"     // Tachyon ray tracer
#include "VrmlDisplayDevice.h"        // VRML 1.0 
#include "Vrml2DisplayDevice.h"       // VRML 2.0 
#include "WavefrontDisplayDevice.h"   // Wavefront "OBJ" files

#if defined(VMDLIBTACHYON) 
#include "LibTachyonDisplayDevice.h"  // Compiled-in Tachyon ray tracer
#endif
#if defined(VMDLIBGELATO)
#include "LibGelatoDisplayDevice.h"   // Compiled-in Gelato renderer
#endif

// constructor, start off with the default means of rendering
FileRenderList::FileRenderList(VMDApp *vmdapp) : app(vmdapp) {
  add(new SnapshotDisplayDevice(app->display));
#if defined(VMDLIBTACHYON)
  add(new LibTachyonDisplayDevice());
#endif
#if defined(VMDLIBGELATO)
  // Only add the internally linked gelato display device to the
  // menu if the user has correctly set the GELATOHOME environment
  // variable.  If we allow them to use it otherwise, it may lead
  // to crashing, or failed renders.  This way they won't even see it
  // as an option unless they've got Gelato installed and the environment
  // at least minimally configured.
  if (getenv("GELATOHOME") != NULL) {
    add(new LibGelatoDisplayDevice());
  }
#endif
  add(new TachyonDisplayDevice());
  add(new POV3DisplayDevice());
  add(new Vrml2DisplayDevice());
  add(new RenderManDisplayDevice());
  add(new GelatoDisplayDevice());
  add(new STLDisplayDevice());
  add(new ArtDisplayDevice());
  add(new PSDisplayDevice());
  add(new RayShadeDisplayDevice());
  add(new RadianceDisplayDevice());
  add(new R3dDisplayDevice());
  add(new VrmlDisplayDevice());
  add(new WavefrontDisplayDevice());
}

// destructor, deallocate all the info
FileRenderList::~FileRenderList(void) {
  for (int i=0;i<renderList.num();i++)
    delete renderList.data(i);
}

// add a new render class with its corresponding name
void FileRenderList::add(FileRenderer *newRenderer) {
  if(newRenderer)
    renderList.add_name(newRenderer->name, newRenderer);
}

// figure out how many render classes are installed
int FileRenderList::num(void) {
  return renderList.num();
}

// return the name for the ith class, returns NULL if out of range
const char *FileRenderList::name(int i) {
  if(i>=0 && i < renderList.num()) {
    return renderList.name(i);
  }
  return NULL;
}

// find class (case-insensitive) for a renderer name, else return NULL  
FileRenderer *FileRenderList::find(const char *rname) {
  int indx = renderList.typecode(rname);
  
  if(indx >= 0)
    return renderList.data(indx);
  else
    return NULL;
}

int FileRenderList::render(const char *filename, const char *method,
                           const char *extcmd) {
  msgInfo << "Rendering current scene to '" << filename << "' ..." << sendmsg;

  FileRenderer *render = find(method);
  if (!render) {
    msgErr << "Invalid render method '" << method << sendmsg;
    return FALSE;
  }

  // XXX Snapshot grabs the wrong buffer, so if we're doing snapshot, swap
  // the buffers, render, then swap back.
  if (!strcmp(method, "snapshot")) app->display->update(TRUE);
  int retval = app->scene->filedraw(render, filename, app->display);
  if (!strcmp(method, "snapshot")) app->display->update(TRUE);

  // if successful, execute external command
  if (retval && extcmd && *extcmd != '\0') {
    JString strbuf(extcmd);
    strbuf.gsub("%s", filename);
    // substitute display %w and %h for display width and height
    int w=100, h=100;
    char buf[32];
    app->display_get_size(&w, &h);
    sprintf(buf, "%d", w);
    strbuf.gsub("%w", buf);
    sprintf(buf, "%d", h);
    strbuf.gsub("%h", buf);
    msgInfo << "Executing post-render cmd '" << (const char *)strbuf << "' ..." << sendmsg;
    vmd_system(strbuf);
  }

  // return result
  msgInfo << "Rendering complete." << sendmsg;
  return retval;
}

int FileRenderList::set_render_option(const char *method, const char *option) {
  FileRenderer *ren;
  ren = find(method);
  if (!ren) {
    msgErr << "No rendering method '" << method << "' available." << sendmsg;
    return FALSE;
  }
  ren->set_exec_string(option);
  return TRUE;
} 

int FileRenderList::has_antialiasing(const char *method) {
  FileRenderer *ren = find(method);
  if (ren) return ren->has_antialiasing();
  return 0;
}

int FileRenderList::aasamples(const char *method, int aasamples) {
  FileRenderer *ren = find(method);
  if (ren) return ren->set_aasamples(aasamples);
  return -1;
}

int FileRenderList::aosamples(const char *method, int aosamples) {
  FileRenderer *ren = find(method);
  if (ren) return ren->set_aosamples(aosamples);
  return -1;
}

int FileRenderList::imagesize(const char *method, int *w, int *h) {
  FileRenderer *ren = find(method);
  if (!ren) return FALSE;
  return ren->set_imagesize(w, h);
}

int FileRenderList::has_imagesize(const char *method) {
  FileRenderer *ren = find(method);
  if (!ren) return FALSE;
  return ren->has_imagesize();
}

int FileRenderList::aspectratio(const char *method, float *aspect) {
  FileRenderer *ren = find(method);
  if (!ren) return FALSE;
  *aspect = ren->set_aspectratio(*aspect);
  return TRUE;
}

int FileRenderList::numformats(const char *method) {
  FileRenderer *ren = find(method);
  if (!ren) return 0;
  return ren->numformats();
}

const char *FileRenderList::format(const char *method, int i) {
  FileRenderer *ren = find(method);
  if (!ren) return NULL;
  if (i < 0) return ren->format();
  return ren->format(i);
}

int FileRenderList::set_format(const char *method, const char *format) {
  FileRenderer *ren = find(method);
  if (!ren) return FALSE;
  return ren->set_format(format);
}

