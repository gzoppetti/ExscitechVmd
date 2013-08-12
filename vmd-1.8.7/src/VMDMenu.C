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
 *      $RCSfile: VMDMenu.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.8 $       $Date: 2009/04/29 15:43:30 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  VMD menu base class.
 ***************************************************************************/

#include "VMDMenu.h"
#include <utilities.h>

VMDMenu::VMDMenu(const char *menuname, VMDApp *vmdapp)
: UIObject(vmdapp) {
  
  name = stringdup(menuname);
}

VMDMenu::~VMDMenu() {
  delete [] name;
}

