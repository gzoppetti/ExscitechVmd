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
 *      $RCSfile: VMDFltkMenu.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.24 $       $Date: 2009/06/05 16:37:37 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Class to manage FLTK menus within VMD.
 ***************************************************************************/

#ifndef VMDFLTKMENU_H
#define VMDFLTKMENU_H

#include <FL/Fl_Window.H>
#include "VMDMenu.h"

//
// VMD Window color scheme macros
//

#define VMDMENU_NEW_COLORSCHEME 1

#if defined(VMDMENU_NEW_COLORSCHEME)
//
// new experimental color scheme
//
#if 0
// alternate window background/menu/button color
#define VMDMENU_WINDOW              fl_gray_ramp(223 * (FL_NUM_GRAY - 1) / 255)
#define VMDMENU_BROWSER_SEL         fl_gray_ramp(207 * (FL_NUM_GRAY - 1) / 255)
#endif
#define VMDMENU_BROWSER_BG          fl_gray_ramp(243 * (FL_NUM_GRAY - 1) / 255)
//#define VMDMENU_BROWSER_SEL         fl_gray_ramp(215 * (FL_NUM_GRAY - 1) / 255)
#define VMDMENU_BROWSER_SEL         fl_rgb_color(210, 225, 210)

// Text colors with FLTK's "@Cxxx" color encoding:
//   http://www.fltk.org/documentation.php/doc-1.1/Fl_Browser.html#Fl_Browser.format_char

// A/D/F text/color strings for VMD main menu
#define VMDMENU_MOL_ACTIVE        "A"
#define VMDMENU_MOL_INACTIVE      "@C88A"
#define VMDMENU_MOL_DISPLAYED     "D"
#define VMDMENU_MOL_NONDISPLAYED  "@C88D"
#define VMDMENU_MOL_FIXED         "F"
#define VMDMENU_MOL_NONFIXED      "@C88F"

// graphics menu displayed/nondisplayed text color codes
#define VMDMENU_REP_ACTIVE        ""
//#define VMDMENU_REP_INACTIVE      "@C47"
#define VMDMENU_REP_INACTIVE      "@C88"

// label menu active/inactive text color codes
#define VMDMENU_GEOM_ACTIVE       ""
#define VMDMENU_GEOM_INACTIVE     "@C88"
#else

//
// original VMD color scheme
//
#define VMDMENU_BROWSER_BG          FL_DARKCYAN
#define VMDMENU_BROWSER_SEL         FL_YELLOW
#define VMDMENU_MOLBROWSER_INACTIVE "@C203A"

// Text colors with FLTK's "@Cxxx" color encoding:
//   http://www.fltk.org/documentation.php/doc-1.1/Fl_Browser.html#Fl_Browser.format_char

// A/D/F text/color strings for VMD main menu
#define VMDMENU_MOL_ACTIVE        "A"
#define VMDMENU_MOL_INACTIVE      "@C203A"
#define VMDMENU_MOL_DISPLAYED     "D"
#define VMDMENU_MOL_NONDISPLAYED  "@C203D"
#define VMDMENU_MOL_FIXED         "F"
#define VMDMENU_MOL_NONFIXED      "@C203F"

// graphics menu displayed/nondisplayed text color codes
#define VMDMENU_REP_ACTIVE        ""
#define VMDMENU_REP_INACTIVE      "@C203"

// label menu active/inactive text color codes
#define VMDMENU_GEOM_ACTIVE       ""
#define VMDMENU_GEOM_INACTIVE     "@C203"
#endif

// general VMD color scheme macros
#define VMDMENU_CHOOSER_BG          FL_PALEGREEN
#define VMDMENU_CHOOSER_SEL         FL_BLACK
#define VMDMENU_SLIDER_BG           FL_WHITE
#define VMDMENU_SLIDER_FG           FL_BLACK
#define VMDMENU_SLIDER_SEL          FL_YELLOW
#define VMDMENU_MENU_SEL            FL_BLACK
#define VMDMENU_CHECKBOX_BG         FL_BLACK
#define VMDMENU_CHECKBOX_FG         FL_RED

// selection highlight color for text value input controls
#define VMDMENU_VALUE_BG            FL_WHITE
#define VMDMENU_VALUE_SEL           FL_YELLOW

// used for frame control in main VMD window
#define VMDMENU_VALUE_SEL2          FL_BLACK

// used by "zoom" checkbox in main VMD window
#define VMDMENU_CHECKBOX_BG         FL_BLACK
#define VMDMENU_CHECKBOX_FG         FL_RED

// material menu sliders
#define VMDMENU_MATSLIDER_BG        FL_WHITE
#define VMDMENU_MATSLIDER_FG        FL_GRAY

// label menu positioner
#define VMDMENU_POSITIONER_BG       VMDMENU_BROWSER_BG
#define VMDMENU_POSITIONER_SEL      VMDMENU_BROWSER_SEL

#if (FL_MAJOR_VERSION >= 1) && (FL_MINOR_VERSION >= 1)
#define VMDFLTKTOOLTIP(obj, string)  (obj)->tooltip(string);
#else 
#define VMDFLTKTOOLTIP(obj, string)
#endif

class Fl_Choice;

/// VMDMenu and FL_Window subclass for managing all FLTK-based menus in VMD
class VMDFltkMenu : public VMDMenu, public Fl_Window {
private:
  char *_title;
  static void window_cb(Fl_Widget *, void *);

protected:
  virtual void do_on();
  virtual void do_off();

public:
  VMDFltkMenu(const char *menuname, const char *title, VMDApp *);
  ~VMDFltkMenu();

  virtual void move(int, int);
  virtual void where(int &, int &);
};

/// a convenience function for filling a molecule chooser
void fill_fltk_molchooser(Fl_Choice *, VMDApp *);

/// a convenience function for generating properly-escaped menu strings
char * escape_fltk_menustring(const char *);

#endif

