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
 *      $RCSfile: MolBrowser.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.35 $      $Date: 2009/05/29 03:42:34 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *   Molecule browser
 ***************************************************************************/

#include "MolBrowser.h"
#include "FL/Fl_Menu_Button.H"
#include "FL/Fl_Menu_Item.H"
#include "FL/Fl_Multi_Browser.H"
#include "FL/forms.H"
#include "frame_selector.h"

#include "VMDApp.h"
#include "Molecule.h"
#include "MoleculeList.h"

#if FL_MAJOR_VERSION <= 1
#if FL_MINOR_VERSION < 1
#include "FL/fl_file_chooser.H"
#endif
#endif

static const int widths[] = { 30, 18, 18, 18, 18, 160, 70, 70, 22, 0 };

MolBrowser::MolBrowser(VMDApp *vmdapp, MainFltkMenu *mm, int x, int y)
: Fl_Multi_Browser(x, y, 450, 90), app(vmdapp) {
  mainmenu = mm;
  dragpending = 0;  
  align(FL_ALIGN_TOP_LEFT);
  column_widths(widths);
  color(VMDMENU_BROWSER_BG);
  selection_color(VMDMENU_BROWSER_SEL);

  new Fl_Box(x,   y-20,30,20,"ID");
  new Fl_Box(x+32,y-20,18,20,"T");
  new Fl_Box(x+50,y-20,18,20,"A");
  new Fl_Box(x+68,y-20,18,20,"D");
  new Fl_Box(x+86,y-20,18,20,"F");
  Fl_Box *b = new Fl_Box(x+102,y-20,220,20,"Molecule");
  b->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
  b = new Fl_Box(x+262,y-20,70,20,"Atoms");
  b->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
  b = new Fl_Box(x+332,y-20,60,20,"Frames");
  b->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
  b = new Fl_Box(x+402,y-20,30,20,"Vol");
  b->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
  end();
}

int MolBrowser::handle(int type) {
#if 1
  // handle paste operations
  if (type == FL_PASTE) {
    // ignore paste operations that weren't due to drag-and-drop 
    // since they could be any arbitrary data/text, and not just filenames.
    if (dragpending) {
      int len = Fl::event_length();

      // ignore zero-length paste events (why do these occur???)
      if (len > 0) {
        int numfiles, i;
        const char *lastc;
        int lasti;
        FileSpec spec; 
        const char *ctext = Fl::event_text();
        char *filename = (char *) malloc((1 + len) * sizeof(char));

        for (lasti=0,lastc=ctext,numfiles=0,i=0; i<len; i++) {
          // parse out all but last filename, which doesn't have a CR
          if (ctext[i] == '\n') {
            memcpy(filename, lastc, (i-lasti)*sizeof(char));
            filename[i-lasti] = '\0';

            // attempt to load the file into a new molecule
            app->molecule_load(-1, filename, NULL, &spec);

            lasti=i+1;
            lastc=&ctext[lasti];
            numfiles++;
          }

          // special-case last filename, since there's no CR
          if (i == (len-1)) {
            memcpy(filename, lastc, (1+i-lasti)*sizeof(char));
            filename[1+i-lasti] = '\0';

            // attempt to load the file into a new molecule
            app->molecule_load(-1, filename, NULL, &spec);
            numfiles++;
          }
        }

        free(filename);
      }

      dragpending = 0; // no longer waiting for drag-and-drop paste
    }

    return 1; // indicate that we handled the paste operation
  }

  // handle drag-and-drop operations
  if (type == FL_DND_ENTER || type == FL_DND_DRAG) {
    return 1; // indicate that we want the drag-and-drop operation
  }
  if (type == FL_DND_RELEASE) {
    Fl::paste(*this);
    dragpending = 1; // flag to expect incoming paste due to DND operation
    return 1;
  }
#endif

  if (type == FL_RELEASE) {
    // update the MainMenu gui in case the molecule selections have changed
    if (mainmenu) mainmenu->update_gui_state();
  }
  
  if (type == FL_PUSH && Fl::event_button() == FL_LEFT_MOUSE 
      && Fl::event_clicks()) {
    // figure out which line is selected; since it's a double click there
    // can be only one.
    int molid = -1; //unique ID of selected mol
    int selmol= -1; //GUI index of the selected mol
    
    for (int i=1; i<=size(); i++) {
      if (selected(i)) { 
        selmol = i-1;
        molid = app->molecule_id(selmol); 
        break;
      }
    }
 
    if (molid >= 0) {
      char need_more_clicks = FALSE;
      
      // figure out where on the line the double click occurred. 
      int mx = Fl::event_x();
      if (mx >= 30 && mx < 48) {
        if ( Fl::event_clicks() > 1){ // triple click: single A/T/D molecule
          for (int j=1; j<= size(); j++) {
            int id = app->molecule_id(j-1);
            app->molecule_activate(id, molid == id);
            app->molecule_display(id, molid == id);
            app->menu_select_mol("graphics", selmol);
          }
          app->molecule_make_top(molid);
          app->scene_resetview();
        }
        else {
          app->molecule_make_top(molid);
          need_more_clicks = TRUE;
        }
      } else if (mx >= 48 && mx < 66) {
        app->molecule_activate(molid, !app->molecule_is_active(molid));
      } else if (mx >= 66 && mx < 84) {
        app->molecule_display(molid, !app->molecule_is_displayed(molid));
      } else if (mx >= 84 && mx < 102) {
        app->molecule_fix(molid, !app->molecule_is_fixed(molid));
      } else if (mx >= 102 && mx < 262) { //rename
        // this code snippet is an exact copy of code in MainFltkMenu.C:
        const char *oldname = app->molecule_name(molid);
        const char *newname = fl_input("Enter a new name for molecule %d:", 
          oldname, molid);
        if (newname) app->molecule_rename(molid, newname);
      } else if (mx >= 332 && mx < 402) { //delete frames
        // this code snippet is an exact copy of code in MainFltkMenu.C:
        int numframes = app->molecule_numframes(molid);
        if (!numframes) {
          fl_alert("Molecule %d has no frames to delete!", molid);
        } else {
          const char *molname = app->molecule_name(molid);
          int frst=0, lst=numframes-1, stride=0;
          int ok = frame_delete_selector(molname, numframes-1, &frst, &lst, &stride);
          if (ok) app->molecule_deleteframes(molid, frst, lst, stride);
        }
      } 
      else need_more_clicks = TRUE;
      if (!need_more_clicks) Fl::event_is_click(0);
    }
  }

  if (mainmenu && type == FL_PUSH && Fl::event_button() == FL_RIGHT_MOUSE) {
    const Fl_Menu_Item *menuitem;
    menuitem=mainmenu->browserpopup_menuitems->popup(Fl::event_x(), Fl::event_y());
    if (menuitem && menuitem->callback()) 
      menuitem->do_callback((Fl_Widget *) mainmenu->menubar, menuitem->user_data());
    return 1; // do not allow parent to process event
  }

  return Fl_Multi_Browser::handle(type);
}

void MolBrowser::update() {
  MoleculeList *mlist = app->moleculeList;
  int nummols = mlist->num();

  if (size() > nummols)
    clear();

  for (int i=0; i<nummols; i++) {
    char buf[256], molnamebuf[81];
    Molecule *mol = mlist->molecule(i);

    // prevent string length overflows
    strncpy(molnamebuf, mol->molname(), sizeof(molnamebuf)-1);


    // display state of active/displayed/fixed by toggling the text color
    sprintf(buf, "%d\t%s\t%s\t%s\t%s\t%-13s\t%-7d\t%-7d\t%-3d",
      mol->id(),
      mlist->is_top(i) ? "T" : " ",
      mlist->active(i) ? VMDMENU_MOL_ACTIVE : VMDMENU_MOL_INACTIVE,
      mlist->displayed(i) ? VMDMENU_MOL_DISPLAYED : VMDMENU_MOL_NONDISPLAYED,
      mlist->fixed(i) ? VMDMENU_MOL_FIXED : VMDMENU_MOL_NONFIXED,
      molnamebuf, mol->nAtoms, mol->numframes(), mol->num_volume_data()
    );

    if (i < size()) 
      text(i+1, buf);
    else
      add(buf);
  }
  
  if (mainmenu) mainmenu->update_gui_state();
}
  
