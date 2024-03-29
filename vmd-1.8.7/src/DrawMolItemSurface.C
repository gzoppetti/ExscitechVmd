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
 *	$RCSfile: DrawMolItemSurface.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.38 $	$Date: 2009/04/29 15:43:01 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This contains the surface code for DrawMolItem
 *
 * The surfaces are made with SURF, an external program.  It was
 * written by Amitabh Varshney when he was at UNC.  The code is
 * available from ftp.cs.unc.edu .
 ***************************************************************************/



#include <stdio.h>
#include <stdlib.h>
#include "DrawMolecule.h"
#include "DrawMolItem.h"
#include "utilities.h"
#include "Surf.h"
#include "Inform.h"
#include "Scene.h"

// In general, the method is
//    write the file for surf input
//    call surf
//    read the triangles
//    write them to the draw list

void DrawMolItem::draw_surface(float *framepos, int draw_wireframe, float radius) {
  // mapping from order printed out (selected) to atom id
  int count = 0; // count is number of atoms selected
  int *map = new int[mol->nAtoms];
  for (int i=0; i<mol->nAtoms; i++) {
    if (atomSel->on[i]) {
      map[count++] = i;
    }
  }

  int surfs_up = 1;

  // regenerate sphere coordinates if necessary 
  if ( needRegenerate & MOL_REGEN ||
       needRegenerate & SEL_REGEN ||
       needRegenerate & REP_REGEN) {

    // then we need to recalculate the SURF
    surf.clear();
    float *x = new float[count];
    float *y = new float[count];
    float *z = new float[count];
    float *r = new float[count];
    const float *aradius = mol->radius();

    // We add all displayed atoms to the sphere array
    int j;
    for (int i=0; i<count; i++) {
      j = map[i];
      r[i] = aradius[j];
      x[i] = framepos[3*j+0];
      y[i] = framepos[3*j+1];
      z[i] = framepos[3*j+2];
    }

    // make the new surface
    surfs_up = surf.compute(radius, count, r, x, y, z);  // returns 0 on fail
      
    delete [] r;
    delete [] x;
    delete [] y;
    delete [] z;
  }

  // and display everything
  if (surfs_up && surf.numtriangles > 0) {
    int i, ind, vnum, vsize;  
    float *c;        

    append(DMATERIALON);

     vnum = surf.numtriangles * 3; // 3 vertices per triangle
    vsize = vnum * 3;              // 3 floats per vertex

    c = new float[vsize];

    for (i=0; i<surf.numtriangles; i++) {
      int col = atomColor->color[map[surf.ind[i]]];
      const float *fp = scene->color_value(col);

      ind = i * 9;
      c[ind    ] = fp[0]; // Red
      c[ind + 1] = fp[1]; // Green
      c[ind + 2] = fp[2]; // Blue

      ind+=3;
      c[ind    ] = fp[0]; // Red
      c[ind + 1] = fp[1]; // Green
      c[ind + 2] = fp[2]; // Blue

      ind+=3;
      c[ind    ] = fp[0]; // Red
      c[ind + 1] = fp[1]; // Green
      c[ind + 2] = fp[2]; // Blue
    }                                                           

    if (draw_wireframe) {
      int *l = new int[surf.numtriangles * 6];
      int i;
      for (i=0; i<surf.numtriangles; i++) {
        int li = i * 6; 
        int ll = i * 3;
        l[li    ] = ll + 0; 
        l[li + 1] = ll + 1; 
        l[li + 2] = ll + 1;
        l[li + 3] = ll + 2;
        l[li + 4] = ll + 2;
        l[li + 5] = ll + 0;
      }

      // Create a wire mesh
      cmdWireMesh.putdata(&surf.v[0], &surf.n[0], c, vnum, 
                          l, surf.numtriangles * 3, cmdList);
    
      delete [] l;
    } else {
      // Create a triangle mesh, but don't try to stripify it since
      // Surf doesn't generate connected geometry.
      cmdTriMesh.putdata(&surf.v[0], &surf.n[0], c, vnum, 
                         &surf.f[0], surf.numtriangles, 
                         0, cmdList);
    }

    delete [] c;
    delete [] map;
  }

  msgInfo << "Done." << sendmsg;
}

