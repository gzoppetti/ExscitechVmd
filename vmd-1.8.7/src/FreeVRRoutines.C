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
 *	$RCSfile: FreeVRRoutines.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.18 $	$Date: 2009/04/29 15:43:03 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * routines to get memory from and return memory to the 
 * FreeVR shared memory arena
 ***************************************************************************/


#include <stdlib.h>
#include "Inform.h"
#include "FreeVRRoutines.h"
#include "VMDApp.h"
#include "FreeVRDisplayDevice.h"
#include "FreeVRScene.h"

#include <freevr.h>

void *malloc_from_FreeVR_memory(size_t size) {
  return vrShmemAlloc(size);
}

void free_to_FreeVR_memory(void *data) {
  vrShmemFree(data);
}

// get megs o' memory from FreeVR, and create the arena
// Warning:  Don't make me do this twice.
void grab_FreeVR_memory(int megs) {
  int size = (megs>1?megs:1) * 1024 * 1024;

  if (vrShmemInit(size) == NULL) 
    msgErr << "Bad juju in the arena.  We're gonna die!" << sendmsg;
  else
    msgInfo <<  "Created arena." << sendmsg;
}


// set up the graphics, called from FreeVRInitApplication
void freevr_gl_init_fn(void) {
}

static Scene *freevrscene;
static DisplayDevice *freevrdisplay;

void set_freevr_pointers(Scene *scene, DisplayDevice *display) {
  freevrscene = scene;
  freevrdisplay = display;
}

// call the child display renderer, and wait until they are done
void freevr_renderer(void) {
  freevrscene->draw(freevrdisplay);
}

