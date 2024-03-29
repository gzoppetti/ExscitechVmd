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
 *      $RCSfile: PlainTextInterp.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.9 $       $Date: 2009/04/29 15:43:20 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Last resort text interpreter if no other is available.
 ***************************************************************************/

#ifndef PLAIN_TEXT_INTERP_H
#define PLAIN_TEXT_INTERP_H

#include "TextInterp.h"

/// TextInterp subclass implementing a last resort text interpreter 
/// if no other is available.
class PlainTextInterp : public TextInterp {
public:
  PlainTextInterp();
  virtual ~PlainTextInterp();

  virtual int evalString(const char *);
  virtual void appendString(const char *);
  virtual void appendList(const char *);
};

#endif


  
