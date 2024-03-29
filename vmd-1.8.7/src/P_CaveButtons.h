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
 *	$RCSfile: P_CaveButtons.h,v $
 *	$Author: johns $	$Locker:  $		$State: Exp $
 *	$Revision: 1.19 $	$Date: 2009/04/29 15:43:14 $
 *
 ***************************************************************************
 * DESCRIPTION:
 * This is Paul's new Tracker code -- pgrayson@ks.uiuc.edu
 *
 * This is a Buttons that gets its info from the CAVE wand.
 *
 ***************************************************************************/

/// Buttons subclass that gets its info from the CAVE wand.
class CaveButtons : public Buttons {
private:
  int numButtons;
   
protected:
  /// Check if we are running CAVE environment
  virtual int do_start(const SensorConfig *);

public:
  CaveButtons();
  
  virtual const char *device_name() const { return "cavebuttons"; }
  virtual Buttons *clone() { return new CaveButtons; }

  virtual void update();
  inline virtual int alive() { return 1; }
  
};

