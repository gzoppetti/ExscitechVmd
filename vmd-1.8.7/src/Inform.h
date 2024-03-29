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
 *      $RCSfile: Inform.h,v $
 *      $Author: johns $       $Locker:  $             $State: Exp $
 *      $Revision: 1.26 $       $Date: 2009/04/29 15:43:07 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 * Inform - takes messages and displays them to the given ostream.
 *
 ***************************************************************************/
#ifndef INFORM_H
#define INFORM_H

// largest message (in bytes) that can be kept
#define MAX_MSG_SIZE    (1024 * 8)


/// Takes messages and displays them to the given ostream.
/// Also creates 3 global instances: msgInfo, msgWarn, msgErr.
/// A message is sent to an Inform object by treating it as an ostream,
/// then ending the message by sending the 'sendmsg' manipulator.
class Inform {
private:
  char *name;                    ///< name printed at start of each line
  char buf[MAX_MSG_SIZE+1];      ///< buffer for messages
#if defined(VMDTKCON)
  int  loglvl;                   ///< vmdcon loglevel
#endif

public:
#if defined(VMDTKCON)
  Inform(const char *, int lvl); ///< constructor: give name and loglevel
#else
  Inform(const char *);          ///< constructor: give name
#endif
  ~Inform();                     ///< destructor
  Inform &send();                ///< print the current message to stdout
  Inform &reset();               ///< reset the buffer

  /// overload the << operator for various items
  Inform& operator<<(const char *);
  Inform& operator<<(char);
  Inform& operator<<(int);
  Inform& operator<<(long);
  Inform& operator<<(unsigned long);
  Inform& operator<<(double);
  Inform& operator<<(Inform& (*f)(Inform &));

  /// retrieve the current text
  const char *text() const {
    return buf;
  }
};

extern Inform& sendmsg(Inform&); ///< manipulator for sending the message.
extern Inform& ends(Inform&);    ///< manipulator for ending a message

// XXX these are global
extern Inform msgInfo;           ///< Generally informative messages
extern Inform msgWarn;           ///< Warnings of possible problems
extern Inform msgErr;            ///< Error messages, more serious

#endif // INFORM_H

