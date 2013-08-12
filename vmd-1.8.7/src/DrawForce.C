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
 *	$RCSfile: DrawForce.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.51 $	$Date: 2009/04/29 15:42:59 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Another Child Displayable component for a remote molecule; this displays
 * and stores the information about the interactive forces being applied to
 * the molecule.  If no forces are being used, this draws nothing.
 *
 * The force information is retrieved from the Atom list in the parent
 * molecule.  No forces are stored here.
 *
 * This name is now a misnomer as accelerations are changed, _not_ forces.
 * This eliminates the problem of having hydrogens acceleterating 12 time
 * faster than carbons, etc.
 *
 * And now I'm changing it back.  We should draw the actual force...
 *
 ***************************************************************************/

#include "DrawForce.h"
#include "DrawMolecule.h"
#include "DisplayDevice.h"
#include "Scene.h"
#include "Atom.h"
#include "Timestep.h"
#include "Inform.h"
#include "utilities.h"
#include "Mouse.h"
#include "VMDApp.h"

////////////////////////////  constructor  

DrawForce::DrawForce(DrawMolecule *mr)
	: Displayable(mr) {

  // save data
  mol = mr;

  // initialize variables
  needRegenerate = TRUE;
  colorCat = (-1);
}


///////////////////////////  protected virtual routines

void DrawForce::do_color_changed(int ccat) {
  // right now this does nothing, since we always redraw the list.  But
  // the general thing it would do is set the flag that a redraw is needed,
  // so looking ahead I'll do this now.
  if(ccat == colorCat) {
    needRegenerate = TRUE;
  }
}

//////////////////////////////// private routines 

#define DRAW_FORCE_SCALE 0.3333f

// regenerate the command list
void DrawForce::create_cmdlist(void) {

  // do we need to recreate everything?
  if(needRegenerate) {

    // regenerate both data block and display commands
    needRegenerate = FALSE;
    reset_disp_list();

    // only put in commands if there is a current frame
    Timestep *ts = mol->current();
    if (ts) {
      append(DMATERIALON);

      // for each atom, if it has a nonzero user force, then display it
      for (int i=0; i < mol->nAtoms; i++) {
	// check if nonzero force
	const float *tsforce = ts->force;
	if(tsforce &&
	   (tsforce[3*i] > 0.0 ||
	   tsforce[3*i+1] > 0.0 ||
	   tsforce[3*i+2] > 0.0 ||
	   tsforce[3*i] < 0.0 ||
	   tsforce[3*i+1] < 0.0 ||
	   tsforce[3*i+2] < 0.0)) {

	  // get position of atom, and the position of the force vector
	  float *p1 = ts->pos + 3*i;
	  float fval[3], p2[3], p3[3];
          for(int k = 0; k < 3; k++) {
            fval[k] = tsforce[3*i + k] * DRAW_FORCE_SCALE;
	    p2[k] = p1[k] + fval[k];
	    p3[k] = p1[k] + 0.8f * fval[k];
          }

	  // find length of force
	  float p2norm = norm(fval);

	  // set cone color
	  int sc = (int)p2norm;
	  if(sc >= MAPCLRS)
	    sc = MAPCLRS - 1;
	  cmdColorIndex.putdata(MAPCOLOR(sc), cmdList);

	  float rada = 0.2f * p2norm;
	  if (rada > 0.3f)
	    rada = 0.3f;
	  float radb = 1.5f * rada;
	  cmdCone.putdata(p3, p1, rada, 0, 9, cmdList);
	  cmdCone.putdata(p3, p2, radb, 0, 9, cmdList);
        }
      }
    }
  }
}


//////////////////////////////// public routines 
// prepare for drawing ... do any updates needed right before draw.
void DrawForce::prepare() {

  if (parent->needUpdate()) {
    needRegenerate = TRUE;
  }
  
  create_cmdlist();
}

