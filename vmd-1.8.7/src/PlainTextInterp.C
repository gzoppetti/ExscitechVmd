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
 *      $RCSfile: PlainTextInterp.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.10 $       $Date: 2009/04/29 15:43:20 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Last resort text interpreter if no other is available
 ***************************************************************************/

#include "PlainTextInterp.h"
#include "Inform.h"
#include "utilities.h"
#include <stdlib.h>

PlainTextInterp::PlainTextInterp() {
  msgInfo << "Starting VMD text interpreter..." << sendmsg; 
}

PlainTextInterp::~PlainTextInterp() {
  msgInfo << "Exiting VMD text interpreter." << sendmsg;
}

int PlainTextInterp::evalString(const char *s) {
  vmd_system(s);
  return 0;
}

void PlainTextInterp::appendString(const char *s) {
  msgInfo << s << sendmsg;
}

void PlainTextInterp::appendList(const char *s) {
  msgInfo << s << sendmsg;
}

 
