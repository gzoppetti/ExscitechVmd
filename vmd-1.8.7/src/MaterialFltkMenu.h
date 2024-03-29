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
 *      $RCSfile: MaterialFltkMenu.h,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.12 $      $Date: 2009/05/15 19:17:32 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Material properties GUI form.
 ***************************************************************************/
#ifndef MATERIAL_FLTK_MENU_H__
#define MATERIAL_FLTK_MENU_H__

#include "VMDFltkMenu.h"

class Fl_Value_Slider;
class Fl_Hold_Browser;
class Fl_Button;
class Fl_Input;

/// VMDFltkMenu subclass implementing a GUI for creating
/// and configuring material properties
class MaterialFltkMenu: public VMDFltkMenu {
private:
  int curmat;                     ///< current material

  void fill_material_browser();
  void set_sliders();

  void init(void);                ///< initialize the user interface

  Fl_Value_Slider *ambient;
  Fl_Value_Slider *specular;
  Fl_Value_Slider *diffuse;
  Fl_Value_Slider *shininess;
  Fl_Value_Slider *opacity;
  Fl_Value_Slider *outline;
  Fl_Value_Slider *outlinewidth;
  Fl_Hold_Browser *browser;
  Fl_Input *nameinput;
  Fl_Button *deletebutton;
  Fl_Button *defaultbutton;

private:
  static void slider_cb(Fl_Widget *w, void *v);
  static void createnew_cb(Fl_Widget *w, void *v);
  static void delete_cb(Fl_Widget *w, void *v);
  static void browser_cb(Fl_Widget *w, void *v);
  static void name_cb(Fl_Widget *w, void *v);
  static void default_cb(Fl_Widget *w, void *v);

protected:
  int act_on_command(int, Command *);

public:
  MaterialFltkMenu(VMDApp *);
};

#endif
