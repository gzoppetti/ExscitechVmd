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
 *      $RCSfile: CoorPluginData.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.22 $       $Date: 2009/04/29 15:42:58 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Interface code that manages loading and saving of coordinate data via 
 *  plugin interfaces.  Uses MolFilePlugin to do the file loading.
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CoorPluginData.h"
#include "MolFilePlugin.h"
#include "Inform.h"
#include "Molecule.h"

CoorPluginData::CoorPluginData(const char *nm, Molecule *m, MolFilePlugin *p,
    int input, int first, int stride, int last, const int *sel) 
: CoorData(nm), is_input(input), begFrame(first), frameSkip(stride), 
  endFrame(last) {

  /// selection is NULL by default, indicating all atoms are to be written
  selection = NULL;

  /// set plugin NULL to indicate trouble if we early exit
  plugin = NULL;

  /// initialize timer handle
  tm=NULL;

  /// initialize data size variables
  kbytesperframe=0;
  totalframes=0;

  // make sure frame data is correct
  if(begFrame < 0)
    begFrame = 0;
  if(endFrame < begFrame)
    endFrame = (-1);
  if(frameSkip <= 0)
    frameSkip = 1;
  recentFrame = -1;

  // make sure frame data is valid
  if(!m || (!is_input && 
              ( m->numframes() < begFrame || endFrame >= m->numframes() ) 
           ) ) {
    msgErr << "Illegal frames requested for coordinate file I/O" << sendmsg;
    return;
  }
  if (is_input) {
    // Checks for and reject attempts to use selections when reading
    // coordinates.  The most useful thing to do would be to allow coordinate
    // files to contain whatever number of atoms you want, and then use the
    // selection to filter those atoms.  However, one could go a number of ways
    // with this.  Should the selection be reparsed using the structure and/or
    // coordinate data in the new file in order to determine which atoms to
    // read, or should one simply use the already-computed atom indices?  I can
    // think of situations where both of those behaviors would be desirable.
    if (sel) {
      msgErr << "Internal error: cannot read selection of coordinates"
             << sendmsg;
      return;
    }
    // make sure that the number of atoms in the coordinate file is either valid
    // or unknown.
    if (p->natoms() != m->nAtoms) {
      if (p->natoms() == -1) {
        p->set_natoms(m->nAtoms);
      } else {
        msgErr << "Incorrect number of atoms (" << p->natoms() << ") in" 
               << sendmsg;
        msgErr << "coordinate file " << nm << sendmsg;
        return;
      } 
    } 
  }

  // make plugin and selection information valid
  plugin = p;
  if (sel) {
    selection = new int[m->nAtoms];
    memcpy(selection, sel, m->nAtoms * sizeof(int));
  }

  tm=vmd_timer_create();
  vmd_timer_start(tm);

  // If this is output, write the structure now.  
  if (!is_input && plugin->can_write_structure()) {
    if (plugin->write_structure(m, selection) == MOLFILE_SUCCESS) {
      totalframes++;
    } else {
      plugin = NULL;
    }
  }

  if (is_input) {
    kbytesperframe = (p->natoms() * 12) / 1024;
  } else {
    kbytesperframe = (m->nAtoms * 12) / 1024;
  }
}

CoorPluginData::~CoorPluginData() {
  delete plugin;
  plugin = NULL;

  if (tm) {
    vmd_timer_destroy(tm);
    tm=NULL;
  }

  delete [] selection;
}

CoorData::CoorDataState CoorPluginData::next(Molecule *m) {
  if (!plugin) 
    return DONE;

  if (is_input) {
    if (recentFrame < 0) {
      recentFrame = 0;
      while (recentFrame < begFrame) {
        plugin->skip(m);
        recentFrame++;
      }
    } else {
      for (int i=1; i<frameSkip; i++) 
        plugin->skip(m);
      recentFrame += frameSkip;
    }
    if (endFrame < 0 || recentFrame <= endFrame) {
      Timestep *ts = plugin->next(m); 
      if (ts) {
        m->append_frame(ts);
        totalframes++;
        return NOTDONE;
      }
    }
  } else if (m->numframes() > 0) {  // output
    if (recentFrame < 0)
      recentFrame = begFrame;
    else
      recentFrame += frameSkip;

    // get next frame, and write to file
    if ((endFrame < 0 || recentFrame <= endFrame) 
        && m->numframes() > recentFrame) {
      Timestep *ts = m->get_frame(recentFrame);
      if (ts) {
        if (!plugin->write_timestep(ts, selection)) {
          totalframes++;
          return NOTDONE;
        } else {
          msgErr << "write_timestep returned nonzero" << sendmsg;
        }
      }      
    }
  }

  if (tm != NULL && totalframes > 0) {
    double iotime = vmd_timer_timenow(tm);
    // emit I/O stats if it took more than 3 seconds
    if (iotime > 3.0) {
      char tmbuf[1024];
      sprintf(tmbuf, "%.1f", iotime);

      msgInfo << "Coordinate I/O rate " 
              << (int) (totalframes / iotime) << " frames/sec, "
              << (int) (((totalframes * kbytesperframe) / 1024.0) / iotime) 
              << " MB/sec, "
              << tmbuf << " sec" << sendmsg;
    }
  }

  // we're done; close file and stop reading/writing
  delete plugin;
  plugin = NULL;
  return DONE;
}

