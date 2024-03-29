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
 *      $RCSfile: PickMode.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.22 $       $Date: 2009/04/29 15:43:19 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Pick mode management class.
 ***************************************************************************/

#ifndef PICK_MODE_H__
#define PICK_MODE_H__

class DisplayDevice;
class DrawMolecule;

/// Pick mode management class
class PickMode {
public:
  PickMode() {}
  virtual ~PickMode() {}

  /// called for start, moving, and end of pick point.  Last argument 
  /// represents scaled [0,1] coordinates for 2-D and transformed position
  /// of pointer for 3-D.
  /// XXX should be pure virtual; needed for Query pick mode.
  virtual void pick_molecule_start(DrawMolecule *, DisplayDevice *, 
                             int /* btn */, int /* tag */, 
                             const int *cell, int /* dim */, 
                             const float * /* pos */ ) {}
  virtual void pick_molecule_move (DrawMolecule *, DisplayDevice *, 
                             int /* tag */, int /* dim */, 
                             const float * /* pos */) {} 
  virtual void pick_molecule_end  (DrawMolecule *, DisplayDevice *) {} 

  virtual void pick_graphics(int molid, int tag, int btn, DisplayDevice *d) {}
  // pick_axes?
  // pick_surface?
};

#endif

