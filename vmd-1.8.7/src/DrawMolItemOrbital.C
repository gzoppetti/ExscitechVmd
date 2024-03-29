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
 *	$RCSfile: DrawMolItemOrbital.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.41 $	$Date: 2009/07/29 04:45:13 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   A continuation of rendering types from DrawMolItem
 *
 *   This file only contains representations for visualizing QM orbital data
 ***************************************************************************/

#if defined(VMDORBITALS) 

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "DrawMolItem.h"
#include "DrawMolecule.h"
#include "BaseMolecule.h" // need volume data definitions
#include "Inform.h"
#include "Isosurface.h"
#include "MoleculeList.h"
#include "Scene.h"
#include "Orbital.h"
#include "QMData.h"
#include "VolumetricData.h"
#include "VMDApp.h"
#include "utilities.h" // timers

#define MYSGN(a) (((a) > 0) ? 1 : -1)

void DrawMolItem::draw_orbital(int wavefnctype, int wavefncspin, 
                               int wavefncexcitation, int orbid, 
                               float isovalue, 
                               int drawbox, int style, 
                               float gridspacing, int stepsize, int thickness) {
  if (!mol->numframes() || gridspacing <= 0.0f)
    return; 

  // only recalculate the orbital grid if necessary
  int regenorbital=0;
  if (wavefnctype != waveftype ||
      wavefncspin != wavefspin ||
      wavefncexcitation != wavefexcitation ||
      orbid != gridorbid ||
      gridspacing != orbgridspacing ||
      orbvol == NULL || 
      needRegenerate & MOL_REGEN ||
      needRegenerate & SEL_REGEN) {
    regenorbital=1;
  }

  double motime=0, voltime=0, gradtime=0;
  vmd_timerhandle timer = vmd_timer_create();
  vmd_timer_start(timer);

  if (regenorbital) {
    // XXX this needs to be fixed so that things like the
    //     draw multiple frames feature will work correctly for Orbitals
    int frame = mol->frame(); // draw currently active frame
    const Timestep *ts = mol->get_frame(frame);

    if (!ts->qm_timestep || !mol->qm_data || 
        !mol->qm_data->num_basis || orbid < 1)
      return;

    // Find the  timestep independent wavefunction ID tag
    // by comparing type, spin, and excitation with the
    // signatures of existing wavefunctions.
    int waveid = mol->qm_data->find_wavef_id_from_gui_specs(
                  wavefnctype, wavefncspin, wavefncexcitation);

    // Translate the wavefunction ID into the index the
    // wavefunction has in this timestep
    int iwave = ts->qm_timestep->get_wavef_index(waveid);

    if (iwave<0 || 
        !ts->qm_timestep->get_wavecoeffs(iwave) ||
        !ts->qm_timestep->get_num_orbitals(iwave) ||
        orbid > ts->qm_timestep->get_num_orbitals(iwave))
      return;

    // Get the orbital index for this timestep from the orbital ID.
    int orbindex = ts->qm_timestep->get_orbital_index_from_id(iwave, orbid);

    // Build an Orbital object and prepare to calculate a grid
    Orbital *orbital = mol->qm_data->create_orbital(iwave, orbindex, 
                                                    ts->pos, ts->qm_timestep);

    // Set the bounding box of the atom coordinates as the grid dimensions
    orbital->set_grid_to_bbox(ts->pos, 3.0, gridspacing);

    // XXX needs more testing, can get stuck for certain orbitals
#if 0
    // XXX for GPU, we need to only optimize to a stepsize of 4 or more, as
    //     otherwise doing this actually slows us down rather than speeding up
    //     orbital.find_optimal_grid(0.01, 4, 8);
    // 
    // optimize: minstep 2, maxstep 8, threshold 0.01
    orbital->find_optimal_grid(0.01, 2, 8);
#endif
    // Calculate the molecular orbital
    orbital->calculate_mo(mol);

    motime = vmd_timer_timenow(timer);

    // query orbital grid origin, dimensions, and axes 
    const int *numvoxels = orbital->get_numvoxels();
    const float *origin = orbital->get_origin();

    float xaxis[3], yaxis[3], zaxis[3];
    orbital->get_grid_axes(xaxis, yaxis, zaxis);

    // build a VolumetricData object for rendering
    char dataname[64];
    sprintf(dataname, "molecular orbital %i", orbid);

    waveftype = wavefnctype;
    wavefspin = wavefncspin;
    wavefexcitation = wavefncexcitation;
    gridorbid = orbid;
    orbgridspacing = gridspacing;
    delete orbvol;
    orbvol = new VolumetricData(dataname, origin, 
                                xaxis, yaxis, zaxis,
                                numvoxels[0], numvoxels[1], numvoxels[2],
                                orbital->get_grid_data());
    delete orbital;

    voltime = vmd_timer_timenow(timer);

    orbvol->compute_volume_gradient(); // calc gradients: smooth vertex normals

    gradtime = vmd_timer_timenow(timer);
  } // regen the orbital grid...


  // draw the newly created VolumetricData object 
  sprintf(commentBuffer, "MoleculeID: %d ReprID: %d Beginning Orbital",
          mol->id(), repNumber);
  cmdCommentX.putdata(commentBuffer, cmdList);

  if (drawbox > 0) {
    // don't texture the box if color by volume is active
    if (atomColor->method() == AtomColor::VOLUME) {
      append(DVOLTEXOFF);
    }
    // wireframe only?  or solid?
    if (style > 0 || drawbox == 2) {
      draw_volume_box_lines(orbvol);
    } else {
      draw_volume_box_solid(orbvol);
    }
    if (atomColor->method() == AtomColor::VOLUME) {
      append(DVOLTEXON);
    }
  }

  if ((drawbox == 2) || (drawbox == 0)) {
    switch (style) {
      case 3:
        // shaded points isosurface looping over X-axis, 1 point per voxel
        draw_volume_isosurface_lit_points(orbvol, isovalue, stepsize, thickness);
        break;

      case 2:
        // points isosurface looping over X-axis, max of 1 point per voxel
        draw_volume_isosurface_points(orbvol, isovalue, stepsize, thickness);
        break;

      case 1:
        // lines implementation, max of 18 line per voxel (3-per triangle)
        draw_volume_isosurface_lines(orbvol, isovalue, stepsize, thickness);
        break;

      case 0:
      default:
        // trimesh polygonalized surface, max of 6 triangles per voxel
        draw_volume_isosurface_trimesh(orbvol, isovalue, stepsize);
        break;
    }
  }

#if 0
  if (regenorbital) {
    double surftime = vmd_timer_timenow(timer);
    char strmsg[1024];
    sprintf(strmsg, "Total MO rep time: %.3f [MO: %.3f vol: %.3f grad: %.3f surf: %.2f]",
            vmd_timer_time(timer), motime, voltime - motime, gradtime - motime, surftime - gradtime);

    msgInfo << strmsg << sendmsg;
  }
#endif

  vmd_timer_destroy(timer);
}


#endif
