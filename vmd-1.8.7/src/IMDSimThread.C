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
 *      $RCSfile: IMDSimThread.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.13 $       $Date: 2009/04/29 15:43:06 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  A multithreaded implementation of the interactive MD 
 *  coordinate/force communication update loop.
 ***************************************************************************/

#include <string.h>
#include <stdio.h>
#include "vmdsock.h"
#include "IMDMgr.h"
#include "IMDSimThread.h"
#include "Inform.h"
#include "utilities.h"

extern "C" void * imdreaderthread(void *v) {
  IMDSimThread *st = (IMDSimThread *)v;
  return st->reader(v);
}

IMDSimThread::IMDSimThread(const char *host, int port) : IMDSim(host, port) { 
  curpos = curbuf = posbuf1 = posbuf2 = NULL; 
  time2die = 0;

  if (!isConnected())
    return;

  deadsocket = 0;

  vmd_mutex_init(&sockmutex);
  vmd_mutex_init(&coordmutex);

  if (vmd_thread_create(&readerthread,
                     imdreaderthread, // my thread routine
                     this             // context for thread
  )) {
    msgErr << "IMDSimThread: unable to create thread" << sendmsg;
  } else {
    msgInfo << "Using multithreaded IMD implementation." << sendmsg;
  }
}

IMDSimThread::~IMDSimThread() {
  time2die = 1;        // time2die is modified here only!!!
  void *status;
  
  if (isConnected()) {
    if (vmd_thread_join(readerthread, &status)) {
      msgErr << "IMDSimThread: unable to join thread" << sendmsg;
    }  
  }
  delete [] posbuf1;
  delete [] posbuf2;
  disconnect();
}

void *IMDSimThread::reader(void *) {
  IMDType type;
  int32 length;
  while (!deadsocket && !time2die) {
    if (!vmdsock_selread(sock, 0)) {
      vmd_msleep(1);
      continue;
    }
    type = imd_recv_header(sock, &length);
     
    switch (type) {
      case IMD_FCOORDS: process_coordinates(length); break;
      case IMD_ENERGIES: process_energies(length);   break; 
      case IMD_MDCOMM: process_mdcomm(length);       break;
      case IMD_IOERROR: deadsocket = 1;              break;
      default: break;  // Don't need to read data 
    }
  }
  vmd_mutex_lock(&sockmutex);
  disconnect();
  vmd_mutex_unlock(&sockmutex);
  return NULL;
}

void IMDSimThread::process_coordinates(int32 length) {
  if (numcoords < length) { // Need to resize
    delete [] posbuf1;
    delete [] posbuf2;
    posbuf1 = new float[3*length];
    posbuf2 = new float[3*length];
    curbuf = posbuf1;
    curpos = posbuf2;  // should I lock?
  }
  numcoords = length; // should I lock?
  
  int errcode = imd_recv_fcoords(sock, numcoords, curbuf);
  
  if (errcode) {
    msgErr << "Error reading remote coordinates!" << sendmsg;
    deadsocket = 1;
  } else {
    // swap the buffers and announce that new coordinates are ready
    vmd_mutex_lock(&coordmutex);
    float *tmp = curpos;
    curpos = curbuf;
    curbuf = tmp;
    new_coords_ready = 1;
    vmd_mutex_unlock(&coordmutex);
  }
}

void IMDSimThread::process_energies(int32 /* length */) {
  vmd_mutex_lock(&coordmutex);

  int errcode = imd_recv_energies(sock, &imdEnergies);

  if (errcode) { 
    msgErr << "Error reading energies!" << sendmsg;
    deadsocket = 1;
  } else {
    if (need2flip) swap4_aligned(&imdEnergies, sizeof(imdEnergies) / 4);
  }

  vmd_mutex_unlock(&coordmutex);
}

// This should never happen, but I'll handle it in case it does
void IMDSimThread::process_mdcomm(int32 length) {
  int32 *ind = new int32[length];
  float *f = new float[3*length];
  
  int errcode = imd_recv_mdcomm(sock, length, ind, f);

  if (errcode) {
    msgErr << "Error reading MDComm-style forces!" << sendmsg;
    deadsocket = 1;
  }
  delete [] ind;
  delete [] f;
}

void IMDSimThread::get_next_ts(float *pos, IMDEnergies *buf) {
  vmd_mutex_lock(&coordmutex);
  memcpy(pos, curpos, 3*numcoords*sizeof(float));
  memcpy(buf, &imdEnergies, sizeof(IMDEnergies));
  new_coords_ready = 0;
  vmd_mutex_unlock(&coordmutex);
  // swap outside of the mutex - yeah baby!
  if (need2flip) swap4_aligned(pos, 3*numcoords);
}

void IMDSimThread::send_forces(int num, int *ind, float *forces) {
  // Total data sent will be one int and three floats for each atom 
  if (need2flip) {
    swap4_aligned(ind, num);
    swap4_aligned(forces, 3*num);
  }

  vmd_mutex_lock(&sockmutex);   
  if (isConnected()) {
    if (imd_send_mdcomm(sock, num, ind, forces)) {
      msgErr << "Error sending MDComm indices+forces" << sendmsg;
      deadsocket = 1;
    }
  }
  vmd_mutex_unlock(&sockmutex);   
}

void IMDSimThread::pause() {
  vmd_mutex_lock(&sockmutex);   
  if (isConnected() && (getSimState() == IMDRUNNING)) {
    simstate = IMDPAUSED;
    imd_pause(sock);
  }
  vmd_mutex_unlock(&sockmutex);   
}

void IMDSimThread::unpause() {
  vmd_mutex_lock(&sockmutex);   
  if (isConnected() && (getSimState() == IMDPAUSED)) {
    simstate = IMDRUNNING;
    imd_pause(sock);
  }
  vmd_mutex_unlock(&sockmutex);
}

void IMDSimThread::detach() {
  vmd_mutex_lock(&sockmutex);   
  if (isConnected()) {
    simstate = IMDOFFLINE;
    imd_disconnect(sock);
    deadsocket = 1;
  }
  vmd_mutex_unlock(&sockmutex);   
}

void IMDSimThread::kill() {
  vmd_mutex_lock(&sockmutex);   
  if (isConnected()) {
    simstate = IMDOFFLINE;
    imd_kill(sock);
    deadsocket = 1;
  }
  vmd_mutex_unlock(&sockmutex);   
}

void IMDSimThread::set_transrate(int rate) {
  vmd_mutex_lock(&sockmutex);   
  if (isConnected()) {
    imd_trate(sock, rate);
  }
  vmd_mutex_unlock(&sockmutex);   
}

