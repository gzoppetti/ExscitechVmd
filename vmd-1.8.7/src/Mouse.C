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
 *	$RCSfile: Mouse.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.140 $	$Date: 2009/04/29 15:43:11 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * The Mouse UI object, which maintains the current state of the 
 * mouse, including what it is currently used for, and how much it has moved
 * from one measurement to the next.  This also deals with any pop-up or
 * pull-down menus available by using the mouse, as well as picking objects.
 *
 * A three-button mouse is assumed here, with the following usage:
 *	1) Buttons 1 and 2 : manipulation and picking.
 *	2) Button 3 (right): pop-up menu
 *
 * This is the general base class definition; specific versions for each
 * windowing/graphics system must be supplied.
 *
 ***************************************************************************/

#include "Mouse.h"
#include "DisplayDevice.h"
#include "TextEvent.h"
#include "CommandQueue.h"
#include "Inform.h"
#include "PickList.h"
#include "VMDApp.h"


// constructor
Mouse::Mouse(VMDApp *vmdapp)
	: UIObject(vmdapp) {

#ifndef VMDGUI
  // tell the graphics display to queue events... 
  // only if there is no GUI available
  app->display->queue_events(); // enable GUI events from graphics window
#endif

  // Initial detail level state would be set here

  // set the default translation and rotation increments
  rocking_enabled = 1;
  transInc = 0.002f;
  rotInc = 1.0f/15.0f;
  scaleInc = 0.0002f;
  RotVelScale = 0.4f;
  currX = currY = oldX = oldY = 0;
  stop_rotation();
  moveObj = 0;
  moveMode = ROTATION;
  pickInProgress = B_NONE;
  buttonDown = B_NONE;

  app->display->set_cursor(DisplayDevice::NORMAL_CURSOR); // set normal cursor

  reset();
}


// destructor
Mouse::~Mouse(void) {
  move_mode(ROTATION);
}

const char* Mouse::get_mode_str(MoveMode mm) {
  const char* modestr;
  
  switch (mm) {
    default:
    case ROTATION:    
                      modestr = "rotate";     break; // default mouse cursor
    case TRANSLATION: modestr = "translate";  break;
    case SCALING:     modestr = "scale";      break;
    case LIGHT:       modestr = "light";      break;
    case USERPOINT:   modestr = "userpoint";  break;
  // All the picking modes follow:
    case PICK:        modestr = "pick";       break;
    case QUERY:       modestr = "query";      break;
    case CENTER:      modestr = "center";     break;
    case LABELATOM:   modestr = "labelatom";  break;
    case LABELBOND:   modestr = "labelbond";  break;
    case LABELANGLE:  modestr = "labelangle"; break;
    case LABELDIHEDRAL: modestr = "labeldihedral"; break;
    case MOVEATOM:    modestr = "moveatom";   break;
    case MOVERES:     modestr = "moveres";    break;
    case MOVEFRAG:    modestr = "movefrag";   break;
    case MOVEMOL:     modestr = "movemol";    break;
    case FORCEATOM:   modestr = "forceatom";  break;
    case FORCERES:    modestr = "forceres";   break;
    case FORCEFRAG:   modestr = "forcefrag";  break;
    case MOVEREP:     modestr = "moverep";    break;
    case ADDBOND:     modestr = "addbond";    break;
  }
  
  return modestr;
}

////////////////////////////  protected routines  /////////////////

// stop rotation of object
void Mouse::stop_rotation(void) {
  xRotVel = yRotVel = zRotVel = 0.0; // null out rotation rate
}

// set the mouse move mode to the given state; return success
int Mouse::move_mode(MoveMode mm, int mobj) {
  const char *modestr;

  // we cannot change the mouse mode if an active pick is going on
  if (pickInProgress)
    return FALSE; // report failure

  // stop rotating no matter what mode we've changed to
  stop_rotation();

  // disable light highlight if previous mode was light mode
  if (moveMode == LIGHT) {
    app->light_highlight(moveObj, FALSE); // turn off old light
  }

  // special actions based on the new mode
  if (mm == LIGHT) {
    moveObj = mobj;
    app->light_highlight(moveObj, TRUE); // turn on new light number mobj
  }

  // change the mouse mode now
  moveMode = mm; 

  // Tell the text interpreter the new mouse mode
  // Set the variable "vmd_mouse_mode" to the correct string
  // and set the variable "vmd_mouse_submode" to the mobj number
  modestr = get_mode_str(moveMode);
  runcommand(new MouseModeEvent(modestr, mobj));    // and set the variables

  // set the cursor style to match the mouse mode
  switch (moveMode) {
    case ROTATION:
    case LIGHT:
    case USERPOINT:
      app->display->set_cursor(DisplayDevice::NORMAL_CURSOR);
      break;

    case TRANSLATION:
      app->display->set_cursor(DisplayDevice::TRANS_CURSOR);
      break;

    case SCALING:
      app->display->set_cursor(DisplayDevice::SCALE_CURSOR);
      break;

    // all the remaining are picking modes
    default:
      app->display->set_cursor(DisplayDevice::PICK_CURSOR);
      break;
  }

  return TRUE; // report success
}



// do action when the mouse is moved
// arg: which buttons are currently being pressed
// return: whether the mouse moved any
int Mouse::mouse_moved() {
  int dx, dy, mymouseMoved;
  int b1Down, b2Down;

  b1Down = buttonDown == B_LEFT;

  // in order to better support old machines, the built-in mouse
  // modes of VMD treat the middle and right mouse buttons identically
  b2Down = (buttonDown == B_MIDDLE || buttonDown == B_RIGHT);

  if (b1Down || b2Down)
    xRotVel = yRotVel = zRotVel = 0.0; // null out rotation rate

  // get current mouse position
  currX = app->display->x();
  currY = app->display->y();

  // and calculate distance mouse has moved
  dx =  5 * (currX - oldX);
  dy = -5 * (currY - oldY); // negate Y coordinates

  mymouseMoved = (dx != 0 || dy != 0);
  if (!mymouseMoved)
    return FALSE;  // early-exit if nothing happened

  // report the mouse location to TCL
  if (make_callbacks && !b1Down && !b2Down) {
    float r[2], oldr[2];
    r[0] = (float) currX; 
    r[1] = (float) currY;
    oldr[0] = (float) oldX; 
    oldr[1] = (float) oldY;
    int tag;

    app->display->rel_screen_pos(r[0], r[1]);
    app->display->rel_screen_pos(oldr[0], oldr[1]);
    if ((r[0] >= 0.0 && r[0] <= 1.0 && r[1] >= 0.0 && r[1] <= 1.0)) {
      // must be in the screen to do a pick!
      app->pickList->pick_check(2, r, tag, NULL, 0.01f, (char *)"mouse");
    } else if (oldr[0] >= 0.0 && oldr[0] <= 1.0 && oldr[1] >= 0.0 &&
	       oldr[1] <= 1.0) {
      // but if we just moved out, inform TCL.
      app->commandQueue->runcommand(new PickAtomCallbackEvent(-1,-1,"mouse"));
    }
  }


  // save mouse coordinates for future reference
  oldX = currX;
  oldY = currY;

  if (!b1Down && !b2Down) return FALSE;

  // check if we are picking something; if so, generate pick-move command
  if (pickInProgress) {
    float mx = (float) currX;
    float my = (float) currY;
    app->display->rel_screen_pos(mx, my);
    if (mx >= 0.0 && mx <= 1.0 && my >= 0.0 && my <= 1.0) {
      float p[2];
      p[0] = mx;
      p[1] = my;
      app->pickList->pick_move(p);
    }
    return TRUE; // report that the mouse has moved
  }

  // Otherwise, if a button is pressed, check how far the mouse moved,
  // and transform the view accordingly.

  // check for button 1 action
  if (b1Down) {
    if (moveMode == ROTATION || moveMode == LIGHT || 
        moveMode >= PICK) {
      xRotVel = rotInc * (float)dy;
      yRotVel = rotInc * (float)dx;
      if (moveMode == ROTATION ||  moveMode >= PICK) {
        // rotate the scene
	      if (xRotVel != 0.0) {
          app->scene_rotate_by(xRotVel, 'x');
          xRotVel *= RotVelScale;
	      }
	      if (yRotVel != 0.0) {
          app->scene_rotate_by(yRotVel, 'y');
          yRotVel *= RotVelScale;
	      }
      } else {
        // rotate a particular light
	      if (xRotVel != 0.0) {
          app->light_rotate(moveObj, xRotVel, 'x');
          xRotVel *= RotVelScale;
	      }
	      if (yRotVel != 0.0) {
          app->light_rotate(moveObj, yRotVel, 'y');
          yRotVel *= RotVelScale;
	      }
      }
    } else if (moveMode == TRANSLATION && mymouseMoved) {
      app->scene_translate_by(transInc*(float)dx, -transInc*(float)dy, 0.0);
    } else if (moveMode == SCALING && dx != 0) {
      float scf = scaling + scaleInc * (float)dx;
      if(scf < 0.0)
        scf = 0.0;
      app->scene_scale_by(scf);
    }
  }
  
  // check for button 2 action
  if (b2Down) {
    if (moveMode == ROTATION || moveMode == LIGHT || 
         moveMode >= PICK) {
      zRotVel = rotInc * (float)dx;
      if (moveMode == ROTATION ||  moveMode >= PICK) {
	      if (zRotVel != 0.0) {
          app->scene_rotate_by(zRotVel, 'z');
          zRotVel *= RotVelScale;
	      }
      } else {
	      if (zRotVel != 0.0) {
          app->light_rotate(moveObj, zRotVel, 'z');
          zRotVel *= RotVelScale;
	      }
      }
    } else if(moveMode == TRANSLATION && dx != 0) {
      app->scene_translate_by(0.0, 0.0, transInc * (float)dx);
    } else if(moveMode == SCALING && dx != 0) {
      float scf = scaling + 10.0f * scaleInc * (float)dx;
      if(scf < 0.0)
        scf = 0.0;
      app->scene_scale_by(scf);
    }
  }

  return TRUE; // report that the mouse has moved
}



// mouse mode for special navigation/flying plugins,
// reports the mouse location and button state to TCL callbacks
int Mouse::mouse_userpoint() {
  float mpos[2];

  xRotVel = yRotVel = zRotVel = 0.0; // null out rotation rate

  // get current mouse position
  currX = app->display->x();
  currY = app->display->y();

  mpos[0] = (float) currX;
  mpos[1] = (float) currY;

  app->display->rel_screen_pos(mpos[0], mpos[1]);

  // inform TCL
  app->commandQueue->runcommand(new MousePositionEvent(mpos[0], mpos[1], buttonDown));

  // save mouse coordinates for future reference
  oldX = currX;
  oldY = currY;

  // Nothing happened for VMD to worry about for changing detail levels etc,
  // the user's Tcl callback will have to deal with this.
  return FALSE;
}



/////////////////////// virtual routines for UI init/display  /////////////
   
// reset the mouse to original settings
void Mouse::reset(void) {
  scaling = 1.0;
  stop_rotation();
  currX = oldX = app->display->x();
  currY = oldY = app->display->y();
}

void Mouse::handle_winevent(long dev, long val) {
  switch(dev) {
    case DisplayDevice::WIN_WHEELUP:
      app->scene_scale_by(1.200f); // mouse wheel up scales up
      break;

    case DisplayDevice::WIN_WHEELDOWN:
      app->scene_scale_by(0.833f); // mouse wheel down scales down
      break;

    case DisplayDevice::WIN_LEFT:
    case DisplayDevice::WIN_MIDDLE:
    case DisplayDevice::WIN_RIGHT:
      if (val == 1 && buttonDown == B_NONE) {
        // start of a fresh button down event.
        xRotVel = yRotVel = zRotVel = 0.0; // null out rotation rate

        oldX = currX = app->display->x(); // save current mouse coords
        oldY = currY = app->display->y();
  
        if (dev == DisplayDevice::WIN_LEFT)
          buttonDown = B_LEFT;
        else if (dev == DisplayDevice::WIN_MIDDLE)
          buttonDown = B_MIDDLE; 
        else
          buttonDown = B_RIGHT;

        // check for a picked item if we are in a picking mode
        if ( moveMode >= PICK && ! pickInProgress) {
          pickInProgress = buttonDown;
          float mx = (float) currX;
          float my = (float) currY;
          app->display->rel_screen_pos(mx, my);
  
          // if picking an object fails, assume we are rotating the object
          float p[2];
          p[0] = mx;
          p[1] = my;
          if (app->pickList->pick_start(pickInProgress, 2, p) < 0)
            pickInProgress = B_NONE;
        }
      } else if (val == 0 && buttonDown != B_NONE) {
        // we're done moving the mouse while the button is down
        if (pickInProgress) {
          pickInProgress = B_NONE;
          app->pickList->pick_end(); // must finish the picking process
        }

        // Would return to previous detail level here
        buttonDown = B_NONE;
      }
      break;

    case DisplayDevice::WIN_KEYBD:
      runcommand(new UserKeyEvent((char) val, app->display->shift_state()));
      break;

    default:
      ;		// ignore other events and just return
  }  // switch
}

void Mouse::set_rocking(int on) {
  rocking_enabled = on;
  if (!on) {
    xRotVel = yRotVel = zRotVel = 0; // null out rotation rate
  }
}

// check for and queue events.  Return TRUE if an event was generated.
int Mouse::check_event(void) {
  int retval = FALSE;
  long dev=0, val=0; // we check for events ourselves

  if ((retval = app->display->read_event(dev, val)))
    handle_winevent(dev, val);

  if (moveMode == USERPOINT) {
    mouse_userpoint(); // user-defined mouse behavior
  } else if (make_callbacks || buttonDown != B_NONE) {
    if (mouse_moved()) {
      // Change to alternate detail level here...
    } 
  }

  if (rocking_enabled) {
    // apply ang velocity, if necessary
    if (xRotVel != 0.0 || yRotVel != 0.0 || zRotVel != 0.0) {
      if (moveMode != LIGHT) {		// (possibly) rotate app->scene
        if (xRotVel != 0.0)
          app->scene_rotate_by(xRotVel, 'x');
        if (yRotVel != 0.0)
          app->scene_rotate_by(yRotVel, 'y');
        if (zRotVel != 0.0)
          app->scene_rotate_by(zRotVel, 'z');
      } else {				// (possibly) rotate particular light
        if (xRotVel != 0.0)
          app->light_rotate(moveObj, xRotVel, 'x');
        if (yRotVel != 0.0)
          app->light_rotate(moveObj, yRotVel, 'y');
        if (zRotVel != 0.0)
          app->light_rotate(moveObj, zRotVel, 'z');
      }
    }
  }
  return retval;
}



