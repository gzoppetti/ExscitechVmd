
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
 *	$RCSfile: Inform.C,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.35 $	$Date: 2009/04/29 15:43:07 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Inform - takes messages and displays them to the given ostream.
 *
 ***************************************************************************/

#include "Inform.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "config.h"
#if defined(VMDTKCON)
#include "vmdconsole.h"
#endif

#if defined(VMDTKCON)
// XXX global instances of the Inform class
Inform msgInfo("Info) ",    VMDCON_INFO);
Inform msgWarn("Warning) ", VMDCON_WARN);
Inform msgErr("ERROR) ",    VMDCON_ERROR);
#else
// XXX global instances of the Inform class
Inform msgInfo("Info) ");
Inform msgWarn("Warning) ");
Inform msgErr("ERROR) ");
#endif

Inform& sendmsg(Inform& inform) { 
  Inform& rc = inform.send(); 

#if defined(VMDTKCON)
  vmdcon_purge();
#else
  fflush(stdout); // force standard output to be flushed here, otherwise output
                  // from Inform, stdio, Tcl, and Python can be weirdly 
                  // buffered, resulting in jumbled output from batch runs
#endif
  return rc;
}

Inform& ends(Inform& inform)    { return inform; }

#if defined(VMDTKCON)  
Inform::Inform(const char *myname, int lvl) {
  name = strdup(myname);
  loglvl=lvl;
  reset();
}
#else
Inform::Inform(const char *myname) {
  name = strdup(myname);
  reset();
}
#endif

Inform::~Inform() {
  free(name);
}

Inform& Inform::send() {
  char *nlptr, *bufptr;
  bufptr = buf;
  if (!strchr(buf, '\n'))
    strcat(buf, "\n");

  while ((nlptr = strchr(bufptr, '\n'))) {
    *nlptr = '\0';
#if defined(VMDTKCON)
    vmdcon_append(loglvl, name, -1);
    vmdcon_append(loglvl, bufptr, -1);
    vmdcon_append(loglvl, "\n", 1);
#else
    printf("%s%s\n", name, bufptr);
#endif
    bufptr = nlptr + 1; 
  }  
  buf[0] = '\0';     
  return *this;
}

Inform& Inform::reset() {
  buf[0] = '\0';     
  return *this;
}

Inform& Inform::operator<<(const char *s) {
  strncat(buf, s, MAX_MSG_SIZE - strlen(buf));
  return *this;
}

Inform& Inform::operator<<(char c) {
  char tmpbuf[2];
  tmpbuf[0] = c;
  tmpbuf[1] = '\0';
  strncat(buf, tmpbuf, MAX_MSG_SIZE - strlen(buf));
  return *this;
}

Inform& Inform::operator<<(int i) {
  char tmpbuf[128];
  sprintf(tmpbuf, "%d", i);
  strncat(buf, tmpbuf, MAX_MSG_SIZE - strlen(buf));
  return *this;
}

Inform& Inform::operator<<(long i) {
  char tmpbuf[128];
  sprintf(tmpbuf, "%ld", i);
  strncat(buf, tmpbuf, MAX_MSG_SIZE - strlen(buf));
  return *this;
}

Inform& Inform::operator<<(unsigned long u) {
  char tmpbuf[128];
  sprintf(tmpbuf, "%ld", u);
  strncat(buf, tmpbuf, MAX_MSG_SIZE - strlen(buf));
  return *this;
}

Inform& Inform::operator<<(double d) {
  char tmpbuf[128];
  sprintf(tmpbuf, "%f", d);
  strncat(buf, tmpbuf, MAX_MSG_SIZE - strlen(buf));
  return *this;
}

Inform& Inform::operator<<(Inform& (*f)(Inform &)) {
  return f(*this);
}

#ifdef TEST_INFORM

int main() {
  msgInfo << "1\n";
  msgInfo << "12\n";
  msgInfo << "123\n";
  msgInfo << sendmsg;
  msgInfo << "6789";
  msgInfo << sendmsg;
  return 0;
}

#endif
 
