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
 *      $RCSfile: VMDFltkMenu.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.19 $       $Date: 2009/05/28 22:35:17 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Class to manage FLTK menus within VMD.
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include <FL/Fl.H>
#include <FL/Fl_Choice.H>
#include "VMDFltkMenu.h"
#include "VMDApp.h"
#include "utilities.h"

void VMDFltkMenu::window_cb(Fl_Widget *w, void *) {
  VMDFltkMenu *m = (VMDFltkMenu *)w;
  m->app->menu_show(m->get_name(), 0);
}

VMDFltkMenu::VMDFltkMenu(const char *menuname,const char *title,VMDApp *vmdapp) 
: VMDMenu(menuname, vmdapp), Fl_Window(0,0,NULL) 
{
	_title=stringdup(title);
	Fl_Window::label(_title);
#if defined(VMDMENU_WINDOW)
	Fl_Window::color(VMDMENU_WINDOW);
#endif
	callback(window_cb);
}

VMDFltkMenu::~VMDFltkMenu() 
{
	delete [] _title;
}

void VMDFltkMenu::do_on() {
	Fl_Window::show();
}

void VMDFltkMenu::do_off() {
	Fl_Window::hide();
}

void VMDFltkMenu::move(int x, int y) {
	Fl_Widget::position(x,y);
}

void VMDFltkMenu::where(int &x, int &y) {
	x = Fl_Widget::x();
	y = Fl_Widget::y();
}

void fill_fltk_molchooser(Fl_Choice *choice, VMDApp *app) {
  for (int i=0; i<app->num_molecules(); i++) {
    int id = app->molecule_id(i);
    const char *s = app->molecule_name(id); 
    char *buf = new char[strlen(s)+32];

    sprintf(buf, "%d: %s%s", id, s, 
      app->molecule_is_displayed(id) ? "" : " (off)");

    // Fltk doesn't allow adding a menu item with the same name as
    // an existing item, so we use replace, which also avoids 
    // problems with the escape characters interpreted by add()
    int ind = choice->add("foobar");
    choice->replace(ind, buf);

    delete [] buf;
  }
}

char * escape_fltk_menustring(const char * menustring) {
  char * newstr;
  int len = strlen(menustring);
  int i, j;

  // don't bother being precise, these are just menu strings, and they're
  // going to be freed immediately, so allocate largest possible memory block
  // we'll ever need (every char being escape) and avoid running through the
  // string twice to accurately count the number of escaped characters.
  newstr = (char *) malloc(((len * 2) + 1) * sizeof(char)); 
  if (newstr == NULL) 
    return NULL;

  i=0;
  j=0;
  while (menustring[i] != '\0') {
    // insert an escape character if necessary
    if (menustring[i] == '/' ||
        menustring[i] == '\\' ||
        menustring[i] == '_') {
      newstr[j] = '\\'; 
      j++;
    } else if (menustring[i] == '&') {
      // FLTK won't escape '&' characters for some reason, so I skip 'em
      i++;
      continue;
    }

    newstr[j] = menustring[i];
    i++;
    j++;
  }  
  newstr[j] = '\0'; // null terminate the string

  return newstr;
}



