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
 *      $RCSfile: GeometryMol.C,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.46 $      $Date: 2009/04/29 15:43:05 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * A base class for all Geometry objects which measure information about
 * atoms in a molecule.  A molecule Geometry monitor is assumed to operate
 * on N atoms, and be able to calculate a single floating-point value for
 * those atoms.  (i.e. the angle formed by three atoms in space)
 *
 ***************************************************************************/

#include "GeometryMol.h"
#include "MoleculeList.h"
#include "Molecule.h"
#include "Displayable.h"
#include "DispCmds.h"
#include "utilities.h"
#include "VMDApp.h"
#include "TextEvent.h"
#include "CommandQueue.h"

/// a monitor class that watches GeometryMol objects for updates
class GeometryMonitor : public DrawMoleculeMonitor {
private:
  GeometryMol *mygeo;

public:
  GeometryMonitor(GeometryMol *g, int id) : mygeo(g), molid(id) { }
  void notify(int id) {
    mygeo->calculate();
    mygeo->update();
  }
  const int molid;
};

////////////////////////  constructor  /////////////////////////////////
GeometryMol::GeometryMol(int n, int *mols, int *atms, const int *cells, 
    MoleculeList *mlist, CommandQueue *cq, Displayable *d) 
: Displayable(d) {

  objIndex = new int[n];
  comIndex = new int[n];
  cellIndex = new int[3*n];
  if (cells) {
    memcpy(cellIndex, cells, 3*n*sizeof(int));
  } else {
    memset(cellIndex, 0, 3*n*sizeof(int));
  }
  geomValue = 0.0;
  numItems = n;
  hasValue = TRUE;

  my_color = 8;
  my_text_size = 1.0f;
  my_text_offset[0] = my_text_offset[1] = 0;
  my_text_format = "%R%d:%a";

  molList = mlist;
  cmdqueue = cq;
  gmName = NULL;
  uniquegmName = NULL;

  for(int i=0; i < numItems; i++) {
    objIndex[i] = mols[i];
    comIndex[i] = atms[i];
    
    // make sure an atom is not repeated in this list
    if(i > 0 && objIndex[i-1]==objIndex[i] && comIndex[i-1]==comIndex[i] &&
        !memcmp(cellIndex+3*i, cellIndex+3*(i-1), 3*sizeof(int))) {
      // set a bogus value for the first atom index, to make the
      // check_mol routine fail
      comIndex[0] = (-1);
    }

    Molecule *m = molList->mol_from_id(mols[i]);
    if (m) {
      GeometryMonitor *mon = new GeometryMonitor(this, objIndex[i]);
      m->register_monitor(mon);
      monitors.append(mon);
    }
  }

  // sort the items properly
  sort_items();

  // create the name for this object
  geom_set_name();
}


////////////////////////  constructor  /////////////////////////////////
GeometryMol::~GeometryMol(void) {
  // delete name if necessary
  if(gmName)
    delete [] gmName;
  if(uniquegmName)
    delete [] uniquegmName;
  delete [] objIndex;
  delete [] comIndex;
  delete [] cellIndex;
  for (int i=0; i<monitors.num(); i++) {
    GeometryMonitor *mon = monitors[i];
    Molecule *m = molList->mol_from_id(mon->molid);
    if (m) m->unregister_monitor(mon);
    delete mon; 
  }
}


////////////////////////  protected routines  //////////////////////////

void GeometryMol::atom_full_name(char *buf, Molecule *mol, int ind) {
  sprintf(buf, "%-d/%-d", mol->id(), ind);
}

void GeometryMol::atom_short_name(char *buf, Molecule *mol, int ind) {
  MolAtom *nameAtom = mol->atom(ind);
  sprintf(buf, "%s%d:%s",
                mol->resNames.name(nameAtom->resnameindex),
                nameAtom->resid,
                mol->atomNames.name(nameAtom->nameindex));
}

void GeometryMol::atom_formatted_name(JString &str, Molecule *mol, int ind) {
  MolAtom *atom = mol->atom(ind);
  str = my_text_format;
  char buf[32];
  // resname
  JString resname = mol->resNames.name(atom->resnameindex);
  str.gsub("%R", (const char *)resname);
  // resname in campel case
  resname.to_camel();
  str.gsub("%r", (const char *)resname);
  // resid
  sprintf(buf, "%d", atom->resid);
  str.gsub("%d", buf);
  // atom name
  str.gsub("%a", mol->atomNames.name(atom->nameindex));
  // atom charge
  sprintf(buf, "%4.3f", mol->charge()[ind]);
  str.gsub("%q", buf);
  // atom index
  sprintf(buf, "%d", ind);
  str.gsub("%i", buf);
}

// set the name of this item
void GeometryMol::geom_set_name(void) {
  char namebuf[256];
  char namebuf2[256];
  char cellbuf[256];
  Molecule *mol;
  register int i;

  if(items() < 1)
    return;

  // put first name in string
  if((mol = check_mol(objIndex[0], comIndex[0]))) {
    atom_full_name(namebuf, mol, comIndex[0]);
    atom_short_name(namebuf2, mol, comIndex[0]);
    sprintf(cellbuf, "-%d-%d-%d", cellIndex[0], cellIndex[1], cellIndex[2]);
    strcat(namebuf, cellbuf);
  } else
    return;

  // put rest of names in string in format: n1/n2/n3/.../nN
  for(i = 1; i < items(); i++) {
    if((mol = check_mol(objIndex[i], comIndex[i]))) {
      strcat(namebuf, "/");
      atom_full_name(namebuf+strlen(namebuf), mol, comIndex[i]);
      strcat(namebuf2, "/");
      atom_short_name(namebuf2+strlen(namebuf2), mol, comIndex[i]);
      sprintf(cellbuf, "-%d-%d-%d", cellIndex[3*i+0], cellIndex[3*i+1],
          cellIndex[3*i+2]);
      strcat(namebuf, cellbuf);
    } else {
      return;
    }
  }
  
  // now make a copy of this name
  if(gmName)
    delete [] gmName;
  if(uniquegmName)
    delete [] uniquegmName;

  uniquegmName = stringdup(namebuf);
  gmName = stringdup(namebuf2);
}


// sort the elements in the list, so that the lowest atom index is first
// (but preserve the relative order, i.e. a-b-c or c-b-a)
void GeometryMol::sort_items(void) {
  register int i,j;

  // swap order if first component index > last component index
  if( (comIndex[0] > comIndex[items()- 1]) ||
      (comIndex[0] == comIndex[items()-1] &&
      		objIndex[0] > objIndex[items()-1]) ) {
    for(i=0, j=(items() - 1); i < j; i++, j--) {
      int tmpindex = comIndex[i];
      comIndex[i] = comIndex[j];
      comIndex[j] = tmpindex;
      tmpindex = objIndex[i];
      objIndex[i] = objIndex[j];
      objIndex[j] = tmpindex;
      int celltmp[3];
      memcpy(celltmp, cellIndex+3*i, 3*sizeof(int));
      memcpy(cellIndex+3*i, cellIndex+3*j, 3*sizeof(int));
      memcpy(cellIndex+3*j, celltmp, 3*sizeof(int));
    }
  }
}


// check whether the given molecule & atom index is OK
// if OK, return Molecule pointer; otherwise, return NULL
Molecule *GeometryMol::check_mol(int m, int a) {

  Molecule *mol = molList->mol_from_id(m);

  if(!mol || a < 0 || a >= mol->nAtoms)
    mol = NULL;
  
  return mol;
}


// for the given Molecule, find the TRANSFORMED coords for the given atom
// return Molecule pointer if successful, NULL otherwise.
Molecule *GeometryMol::transformed_atom_coord(int ind, float *pos) {
  Molecule *mol;

  // only return value if molecule is legal and atom is displayed
  int a=comIndex[ind];
  if((mol = normal_atom_coord(ind, pos)) && mol->atom_displayed(a)) {

    // now multiply it by the molecule's tranformation matrix
    (mol->tm).multpoint3d(pos, pos);

    // calculation was successful; return the molecule pointer
    return mol;
  }
  
  // if here, error (i.e. atom not displayed, or not proper mol id)
  return NULL;
}


// for the given Molecule, find the UNTRANSFORMED coords for the given atom
// return Molecule pointer if successful, NULL otherwise.
Molecule *GeometryMol::normal_atom_coord(int ind, float *pos) {
  Timestep *now;
  Molecule *mol;

  int m=objIndex[ind];
  int a=comIndex[ind];
  const int *cell = cellIndex+3*ind;

  // get the molecule pointer, and get the coords for the current timestep
  if((mol = check_mol(m, a)) && (now = mol->current())) {
    memcpy((void *)pos, (void *)(now->pos + 3*a), 3*sizeof(float));
    
    // Apply periodic image transformation before returning
    Matrix4 mat;
    now->get_transform_from_cell(cell, mat);
    mat.multpoint3d(pos, pos);
    
    return mol;
  }
  
  // if here, error (i.e. atom not displayed, or not proper mol id)
  return NULL;
}


// draws a line between the two given points
void GeometryMol::display_line(float *p1, float *p2, VMDDisplayList *d) {
  DispCmdLine cmdLineGeom;
  cmdLineGeom.putdata(p1, p2, d);
}


// print given text at current valuePos position
void GeometryMol::display_string(const char *str, VMDDisplayList *d) {
  DispCmdTextOffset cmdTextOffset;
  cmdTextOffset.putdata(my_text_offset[0], my_text_offset[1], d);

  DispCmdTextSize cmdTextSize;
  cmdTextSize.putdata(my_text_size, d);

  DispCmdText cmdTextGeom;
  cmdTextGeom.putdata(valuePos, str, d);
}

////////////////////  public virtual routines  //////////////////////////

// return the name of this geometry marker; by default, just blank
const char *GeometryMol::name(void) {
  return (gmName ? gmName : "");
}


// return 'unique' name of the marker, which should be different than
// other names for different markers of this same type
const char *GeometryMol::unique_name(void) {
  return (uniquegmName ? uniquegmName : name());
}


// check whether the geometry value can still be calculated
int GeometryMol::ok(void) {
  register int i;
  
  for(i=0; i < numItems; i++)
    if(!check_mol(objIndex[i], comIndex[i]))
      return FALSE;

  return TRUE;
}


// calculate a whole list of items, if this object can do so.  Return success.
int GeometryMol::calculate_all(ResizeArray<float> &valArray) {
  int i, max_ts, orig_ts;

  // use the first molecule to determine which frames to cycle through
  Molecule *mol = molList->mol_from_id(objIndex[0]);
  if( ! has_value() || !mol )
    return FALSE;

  // get the max & current frame number, and make sure there are frames
  max_ts = mol->numframes()-1;
  if((orig_ts = mol->frame()) < 0)
    return FALSE;

  // go through all the frames, calculating values
  for(i=0; i <= max_ts; i++) {
    mol->override_current_frame(i);
    valArray.append(calculate());
  }
  
  // reset the current frame and current value of this geometry monitor
  mol->override_current_frame(orig_ts);
  calculate();
  
  return TRUE;
}

// save the text of a selection to the interpreter variable "vmd_pick_selection"
// The format is of the form "index %d %d %d .... %d" for each atom picked.
// (If nothing is picked, the text is "none")
void GeometryMol::set_pick_selection(int , int num, int *atoms) {
  cmdqueue->runcommand(new PickSelectionEvent(num, atoms));
}

void GeometryMol::set_pick_selection() {
  cmdqueue->runcommand(new PickSelectionEvent(0, 0));
}
    
void GeometryMol::set_pick_value(double newval) {
  cmdqueue->runcommand(new PickValueEvent(newval));
}

