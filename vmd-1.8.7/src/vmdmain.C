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
 *      $RCSfile: vmdmain.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.5 $       $Date: 2009/04/29 15:43:39 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Main program.
 *
 ***************************************************************************/
#include "vmd.h"

// Exscitech
#include <GL/glew.h>
#include <GL/glut.h>

#include <QtGui/QApplication>
#include <string>
#include <iostream>
#include "Exscitech/Games/GameController.hpp"
#include <QtGui/QStyle>

#include <QtGui/QCleanlooksStyle>
#include <QtGui/QCDEStyle>
#include <QtGui/QMotifStyle>
#include <QtGui/QPlastiqueStyle>
// End Exscitech

int
main (int argc, char *argv[])
{

  if (!VMDinitialize (&argc, argv))
  {
    return 0;
  }

  QApplication::setStyle(new QWindowsStyle);
  //QApplication::setStyle (new QPlastiqueStyle);
  //QApplication::setStyle (new QCleanlooksStyle);
  //QApplication::setStyle(new QMotifStyle);
  //QApplication::setStyle(new QCDEStyle);
  QApplication qtApp (argc, argv);

  const char *displayTypeName = VMDgetDisplayTypeName ();
  int displayLoc[2], displaySize[2];
  VMDgetDisplayFrame (displayLoc, displaySize);

  VMDApp *vmdApp = new VMDApp (argc, argv);

  if (!vmdApp->VMDinit (argc, argv, displayTypeName, displayLoc, displaySize))
  {
    delete vmdApp;
    return 1;
  }

  // read various application defaults
  VMDreadInit (vmdApp);

  // read user-defined startup files
  VMDreadStartup (vmdApp);

  static Exscitech::GameController* instance = Exscitech::GameController::acquire();

  instance->initPlugin (vmdApp, &qtApp);
  glutInit(&argc, argv);
  // main event loop
  do
  {
    // If we knew that there were no embedded python interpreter, we could
    // process Tcl events here, rather than within the VMD instance.
#ifdef VMDTCL
    // Loop on the Tcl event notifier
    // while (Tcl_DoOneEvent(TCL_DONT_WAIT));
#endif

    // handle Fltk events
    VMDupdateFltk ();

    qtApp.processEvents ();

    instance->updatePlugin ();

#if 0
    // take over the console
    if (vmd_check_stdin())
    {
      vmdApp->process_console();
    }
#endif

  }
  while (!instance->shouldUpdate() || vmdApp->VMDupdate (VMD_CHECK_EVENTS));

  instance->shutdownPlugin ();

  // end of program
  delete vmdApp;
  VMDshutdown ();

  return 0;
}

