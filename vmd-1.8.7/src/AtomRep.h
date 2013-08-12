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
 *      $RCSfile: AtomRep.h,v $
 *      $Author: johns $        $Locker:  $                $State: Exp $
 *      $Revision: 1.64 $      $Date: 2009/07/14 20:32:22 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * Parse and maintain the data for how a molecule should be represented.
 *
 ***************************************************************************/
#ifndef ATOMREP_H
#define ATOMREP_H

// default atom representation, and max cmd string size
#define DEFAULT_ATOMREP		AtomRep::LINES
#define MAX_ATOMREP_CMD		255
#define MAX_ATOMREP_DATA        12

/// Parse and maintain the data for how a molecule should be represented.
class AtomRep {
public:
  // enumeration of all of the methods for representing molecules
  // XXX note that these MUST be ordered exactly the same way that
  //     they are ordered in AtomRep.C, and in the graphics forms.
  enum RepMethod { LINES, BONDS, DYNAMICBONDS, HBONDS,
                   POINTS, VDW, CPK, LICORICE, 
#ifdef VMDPOLYHEDRA
                   POLYHEDRA,
#endif
		   TRACE, TUBE, RIBBONS, 
                   NEWRIBBONS, STRUCTURE, NEWCARTOON, 
#ifdef VMDWITHCARBS
                   RINGS_PAPERCHAIN, RINGS_TWISTER,
#endif
#ifdef VMDMSMS
                   MSMS, 
#endif
#ifdef VMDSURF
	           SURF,
#endif
                   VOLSLICE, ISOSURFACE,
#ifdef VMDFIELDLINES
	           FIELDLINES,
#endif
#ifdef VMDORBITALS
                   ORBITAL,
#endif
                   BEADS,
                   DOTTED, SOLVENT,
		   TOTAL };

  // these are the same as above, but for historical reasons
  enum RepDataNames { SPHERERAD, BONDRAD, SPHERERES, BONDRES, LINETHICKNESS, ISOSTEPSIZE, ISOLINETHICKNESS, WAVEFNCTYPE, WAVEFNCSPIN, WAVEFNCEXCITATION, GRIDSPACING };

  /// initial string with representation command
  char cmdStr[MAX_ATOMREP_CMD + 1];

private:
  /// results of command ... how to represent atoms, and some parameters
  int repMethod;
  float repDataStorage[TOTAL][MAX_ATOMREP_DATA];
  float *repData;

  /// parse the given command, and store results.  Return success.
  int parse_cmd(const char *);

public:
  AtomRep(void);
  AtomRep(AtomRep &ar) { *this = ar; }
  ~AtomRep(void);
  
  /// equal operator, to change the current settings.
  AtomRep& operator=(const AtomRep &);

  /// provide new settings
  int change(const char *newcmd) { return parse_cmd(newcmd); }

  //
  // info about current settings
  //

  /// return representation method
  int method(void) { return repMethod; }
  
  /// return representation method
  bool is_volumetric(void) { 
#ifdef VMDFIELDLINES
    if (repMethod == VOLSLICE || repMethod == ISOSURFACE || repMethod == FIELDLINES)
      return true; 
#else
    if (repMethod == VOLSLICE || repMethod == ISOSURFACE) 
      return true; 
#endif
    return false; 
  }

  bool is_orbital(void) {
#ifdef VMDORBITALS
    if (repMethod == ORBITAL) 
      return true; 
#endif
    return false; 
  }

  /// return value of Nth data item
  float get_data(int n) { return repData[n]; }
};


//
// global structures and variables defining the data each rep requires
//

// The following structure is used to define how each individual data
// value is used by each rep style.  Each style uses a number of different
// data items to define that rep; this structure defines the type and range
// of values allowed for those items.
// Each AtomRep has storage for 5 optional data values.
// The order they are specified is defined by each rep style.

/// define which slots are used for a rep style, and their default values
typedef struct {
  int index;                 ///< which one of the 6 storage slots to use
  float defValue;            ///< default value for the item
} AtomRepDataInfo;

/// structure definition for structures used to define what data each rep needs
typedef struct {
  const char *name;          ///< name of this rep
  int numdata;               ///< number of data values required for style
  AtomRepDataInfo repdata[MAX_ATOMREP_DATA];  // info about each data value
} AtomRepParamStruct;


// array with definition of data for each rep
extern const AtomRepParamStruct AtomRepInfo[AtomRep::TOTAL];


#endif

