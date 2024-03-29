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
 *	$RCSfile: CommandQueue.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.53 $	$Date: 2009/04/29 15:42:57 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * 
 * This stores all the Commands to be run in a queue.  The idea is that
 * the various events add Commands to the command queue then they
 * are read off the queue and the UIs are notified.
 *
 * Commands may be logged to a file, if desired.
 *
 ***************************************************************************/

#include "CommandQueue.h"
#include "Command.h"
#include "UIObject.h"
#include "TextEvent.h"
#include "utilities.h"
#include "config.h"

///////////////////////////  constructor
CommandQueue::CommandQueue(void) : cmdlist(64) {
}
    

///////////////////////////  destructor
// we must remove all commands, and delete them as well.
// if logging, must close file
CommandQueue::~CommandQueue(void) {
  for (int i=0; i<cmdlist.num(); i++)
    delete cmdlist[i];
}

void CommandQueue::register_UI(UIObject *ui) {
  if (uilist.find(ui) == -1) 
    uilist.append(ui);
}

void CommandQueue::unregister_UI(UIObject *ui) {
  int ind = uilist.find(ui);
  if (ind >= 0)
    uilist.remove(ind);
}

////////////////////////////  private routines  ////////////////////////
void CommandQueue::runcommand(Command *cmd) {
  // ... and report the action has been done
  Command::Cmdtype cmdtype = cmd -> gettype();
  int n = uilist.num();
  for (int i=0; i<n; i++) {
    UIObject *ui = uilist[i];
    // XXX call act_on_command even if not ui->active()
    if (ui->want_command(cmdtype)) 
      ui->act_on_command(cmdtype, cmd);
  }
  delete cmd;
}

////////////////////////////  public routines  ////////////////////////

// add a new command to the list ... always adds to queue, does not
// execute.  
void CommandQueue::append(Command *cmd) {
  cmdlist.append(cmd);
}

void CommandQueue::execute_all() {
  int n = cmdlist.num();
  for (int i=0; i<n; i++) {
    runcommand(cmdlist[i]);
  }
  cmdlist.clear();
}

void CommandQueue::check_events() {
  for (int i=0; i<uilist.num(); i++) {
    UIObject *ui = uilist[i];
    if (ui->active())
      ui->check_event();
  }
}


