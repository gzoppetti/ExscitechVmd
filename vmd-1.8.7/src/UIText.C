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
 *	$RCSfile: UIText.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.188 $	$Date: 2009/04/29 15:43:28 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * This is the User Interface for text commands.  It reads characters from
 * the console, and executes the commands.
 *
 * This will use the Tcl library for general script interpretation, which
 * allows for general script capabilities such as variable substitution,
 * loops, etc.  If Tcl cannot be used, text commands will still be available
 * in the program, but the general script capabilities will be lost.
 ***************************************************************************/

#ifdef VMDPYTHON
#include "PythonTextInterp.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include "UIText.h"
#include "TextEvent.h"
#include "CommandQueue.h"
#include "Inform.h"
#include "config.h"
#include "utilities.h"
#include "PlainTextInterp.h"
#include "VMDApp.h"
#include "TclTextInterp.h"
#include "Molecule.h"
#include "MoleculeList.h"
#include "SymbolTable.h"

#if defined(VMDTKCON)
#include "vmdconsole.h"
#endif

////////////////////////////  UIText routines  ///////////////////////////

// constructor
UIText::UIText(VMDApp *vmdapp)
  : UIObject(vmdapp) {
  tclinterp = NULL;
  pythoninterp = NULL;

  // UIText logs all commands
#if defined(VMDTKCON)
  cmdbufstr = new Inform("", VMDCON_ALWAYS);
#else
  cmdbufstr = new Inform("");
#endif
  for (int i=0; i<Command::TOTAL; i++) command_wanted(i);

  // Initialize the Tcl interpreter if enabled
#ifdef VMDTCL
  tclinterp = new TclTextInterp(app);
#ifdef VMDTKCON
  // hook the default tcl interpreter into vmdcon as default.
  vmdcon_set_status(vmdcon_get_status(), ((TclTextInterp *)tclinterp)->get_interp());
#endif
#endif

  // The Python interpreter is initialized on first use.
  // However, if there's no Tcl interpreter, initialize the Python interpreter
  // right away.
#ifdef VMDPYTHON
  if (!tclinterp)
    pythoninterp = new PythonTextInterp(app);
#endif

  // start with the Tcl interpreter by default, if it exists.  If there is
  // no Tcl interpreter, go with Python.  Last resort: PlainTextInterpreter
  // (which has nothing to do with crypto, by the way).
  if (tclinterp)
    interp = tclinterp; 
  else if (pythoninterp)
    interp = pythoninterp;
  else
    interp = new PlainTextInterp;

  reset();
}

// This is called in VMDApp::VMDinit just before reading the .vmdrc file.
// It has to be done after the previous initialization because the
// event look was not yet available.
void UIText::read_init(void) {
  interp->doInit();
}

// destructor
UIText::~UIText(void) {
  if (tclinterp) tclinterp->logfile_cb("exit");
#ifdef VMDTCL
  delete tclinterp;
#endif

#ifdef VMDPYTHON
  delete pythoninterp;
#endif
  delete cmdbufstr;
}

int UIText::save_state(const char *fname) {
  if (tclinterp) {
    JString cmd("save_state ");
    cmd += "{";
    cmd += fname;
    cmd += "}";
    return tclinterp->evalString((const char *)cmd);
  }
  return FALSE;
}

// specify new file to read commands from
void UIText::read_from_file(const char *fname) {
  interp->evalFile(fname);
}

// check for an event; return TRUE if we found an event; FALSE otherwise
int UIText::check_event(void) {
  // let the text interpreter have control for a while
  // If a Python interpreter has been initialized, let it do the Tk updates.
  // Python takes care of updating Tk; if we try to update Tk from within
  // Tcl when Tkinter widgets have been created, we crash and burn as Tk
  // is not thread safe.
  // If Python was not able to update Tk, possibly because Tkinter was not
  // found, then use the Tcl interpreter. 
  if (!pythoninterp || (pythoninterp && !pythoninterp->doTkUpdate())) {
    if (tclinterp) {
      tclinterp->doTkUpdate();
    } else {
      // last resort - use whatever interpreter we've got.
      interp->doTkUpdate();
    }
  }
  interp->doEvent();
  return TRUE; 
}

int UIText::act_on_command(int cmdtype, Command *cmd) {
  if (tclinterp) {
    // log command, if possible
    if (cmd->has_text(cmdbufstr)) {
      tclinterp->logfile_cb(cmdbufstr->text());
      cmdbufstr->reset();
    }
  }
  if (cmdtype == Command::INTERP_EVENT) {
    // downcast to InterpEvent
    InterpEvent *event = (InterpEvent *)cmd;  
    if (tclinterp)
      event->do_callback(tclinterp);
    // ACK!  This used be "else if (pythoninterp)", which means python
    // callbacks never get called if you build with Tcl.  
    if (pythoninterp)
      event->do_callback(pythoninterp);
     
  } else {
    return 0;  // no action taken
  }
  return 1;    // used the command
}

int UIText::change_interp(const char *interpname) {
  if (!interpname) return FALSE;
  if (!strupcmp(interpname, "tcl")) {
    if (tclinterp) {
      msgInfo << "Text interpreter now Tcl" << sendmsg;
      interp = tclinterp;
      return TRUE;
    } else {
      msgErr << "Sorry, no Tcl text interpreter available" << sendmsg;
      // try for Python
      interpname = "python";
    }
  } 
  // fall through to Python if no Tcl interpreter is available
  if (!strupcmp(interpname, "python")) {
    if (pythoninterp) {
      msgInfo << "Text interpreter now Python" << sendmsg;
      interp = pythoninterp;
      // On MACOSX, when we change to the Python interpreter _after_ the
      // first time it's created (i.e. gopython, EOF, gopython), we get
      // kicked out right away because for some reason stdin has the EOF
      // flag set.  So, we clear the EOF flag here.
      clearerr(stdin);
      return TRUE;
    } else {
#if defined(VMDPYTHON) 
      pythoninterp = new PythonTextInterp(app);
      if (pythoninterp) {
        msgInfo << "Text interpreter now Python" << sendmsg;
        interp = pythoninterp;
        return TRUE;
      } else {
        msgErr << "Sorry, unable to start Python text interpreter" << sendmsg;
      }
#else
      msgErr << "Sorry, this version of VMD was compiled with Python support disabled" << sendmsg;
#endif
    }
  } else {
    msgErr << "Unsupported text interpreter requested" << sendmsg;
  }
  return FALSE;
}

