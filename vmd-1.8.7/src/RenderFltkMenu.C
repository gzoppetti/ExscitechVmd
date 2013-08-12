/***************************************************************************
 *cr                                                                       
 *cr            (C) Copyright 1995-2009 The Board of Trustees of the           
 *cr                        University of Illinois                       
 *cr                         All Rights Reserved                        
 *cr                                                                   
 ***************************************************************************/

#include "RenderFltkMenu.h"
#include "CmdRender.h"
#include <FL/Fl.H>
#include <FL/forms.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Box.H>
#include "VMDApp.h"

void RenderFltkMenu::make_window() {
  size(330, 220);
  { 
    { Fl_Choice* o = formatchoice = new Fl_Choice(110, 45, 130, 25, "Render using:");
      o->down_box(FL_BORDER_BOX);
      o->color(VMDMENU_CHOOSER_BG, VMDMENU_CHOOSER_SEL);
      o->callback(formatchoice_cb, this);
    }
    { Fl_Input* o = filenameinput = new Fl_Input(15, 95, 235, 25, "Filename:");
      o->align(FL_ALIGN_TOP_LEFT);
      o->selection_color(VMDMENU_VALUE_SEL);
    }
    Fl_Button *browsebutton = new Fl_Button(255, 95, 65, 25, "Browse...");
    browsebutton->callback(browse_cb, this);
    { Fl_Input* o = commandinput = new Fl_Input(15, 145, 190, 25, "Render Command:");
      o->align(FL_ALIGN_TOP_LEFT);
      o->selection_color(VMDMENU_VALUE_SEL);
      o->when(FL_WHEN_CHANGED);
      o->callback(command_cb, this);
    }
    Fl_Button *defaultbutton = new Fl_Button(210, 145, 110, 25, "Restore default");
    defaultbutton->callback(default_cb, this);
    Fl_Button *renderbutton = new Fl_Button(15, 185, 305, 25, "Start Rendering");
    renderbutton->callback(render_cb, this);
    { Fl_Box* o = new Fl_Box(15, 15, 230, 25, "Render the current scene to a file.");
      o->align(FL_ALIGN_LEFT|FL_ALIGN_INSIDE);
    }
    end();
  }
}

void RenderFltkMenu::fill_render_choices() {
  formatchoice->clear();
  for (int n=0; n<app->filerender_num(); n++)
    formatchoice->add(app->filerender_name(n));
  formatchoice->value(0);
  formatchoice_cb(NULL, this);
}

RenderFltkMenu::RenderFltkMenu(VMDApp *vmdapp)
: VMDFltkMenu("render", "File Render Controls", vmdapp) {

  make_window();
  fill_render_choices();
  command_wanted(Command::RENDER_OPTION);
}

void RenderFltkMenu::formatchoice_cb(Fl_Widget *, void *v) {
  RenderFltkMenu *self = (RenderFltkMenu *)v;
  const char *method = self->formatchoice->text();
  const char *fname = self->app->filerender_default_filename(method);
  const char *opt = self->app->filerender_option(method, NULL);
  if (fname) self->filenameinput->value(fname);
  if (opt)   self->commandinput->value(opt);
}

void RenderFltkMenu::command_cb(Fl_Widget *, void *v) {
  RenderFltkMenu *self = (RenderFltkMenu *)v;
  const char *method = self->formatchoice->text();
  const char *cmd = self->commandinput->value();
  if (method && cmd) self->app->filerender_option(method, cmd);
}

void RenderFltkMenu::default_cb(Fl_Widget *, void *v) {
  RenderFltkMenu *self = (RenderFltkMenu *)v;
  const char *method = self->formatchoice->text();
  if (method) {
    const char *opt = self->app->filerender_default_option(method);
    self->app->filerender_option(method, opt);
    self->commandinput->value(opt);
  }
}

void RenderFltkMenu::browse_cb(Fl_Widget *, void *v) {
  RenderFltkMenu *self = (RenderFltkMenu *)v;
  char *fname = self->app->vmd_choose_file(
      "Select rendering output file:", "*", "All files",1);
  if (fname) {
    self->filenameinput->value(fname);
    delete [] fname;
  }
}

void RenderFltkMenu::render_cb(Fl_Widget *w, void *v) {
  RenderFltkMenu *self = (RenderFltkMenu *)v;
  Fl_Button *renderbutton = (Fl_Button *)w;
  const char *method = self->formatchoice->text();
  const char *outfile = self->filenameinput->value();
  const char *outcmd = self->commandinput->value();
  if (!method || !outfile || !strlen(outfile)) {
    fl_alert("Please select a file format and filename before rendering.");
    return;
  }
  renderbutton->label("Rendering in progress...");
  renderbutton->value(1);
  Fl::wait(0);

  int rc = self->app->filerender_render(method, outfile, outcmd);
  renderbutton->label("Start Rendering");
  renderbutton->value(0);
  if (!rc) {
    fl_alert("File rendering failed; check the VMD text console for errors.");
  }
}

int RenderFltkMenu::act_on_command(int type, Command *cmd) {
  if (type == Command::RENDER_OPTION) {
    CmdRenderOption *cmdrender = (CmdRenderOption *)cmd;
    if (!strcmp(cmdrender->method, formatchoice->text()) &&
         strcmp(cmdrender->option, commandinput->value())) {
      commandinput->value(cmdrender->option);
    }
    return TRUE;
  }
  return FALSE;
}


