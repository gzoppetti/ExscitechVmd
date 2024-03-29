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
 *      $RCSfile: PythonTextInterp.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.64 $       $Date: 2009/04/29 15:43:20 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  Python text interpreter
 ***************************************************************************/

#include "py_commands.h"
#include "Inform.h"
#include "PythonTextInterp.h"
#include "config.h"
#include "VMDApp.h"
#include "TextEvent.h"

#if defined(__APPLE__)
// use the Apple-provided Python framework
#include "Python/errcode.h"
#else
#include "errcode.h"
#endif

static PyObject *cbdict = NULL;

static PyObject *add_callback(PyObject *, PyObject *args) {
  char *type;
  PyObject *temp;

  if (!PyArg_ParseTuple(args, (char *)"sO:add_callback", &type, &temp)) 
    return NULL;

  if (!PyCallable_Check(temp)) {
    PyErr_SetString(PyExc_TypeError, "parameter must be callable");
    return NULL;
  }
  PyObject *cblist = PyDict_GetItemString(cbdict, type);
  if (!cblist) {
    PyErr_SetString(PyExc_KeyError, type);
    return NULL;
  }
  PyList_Append(cblist, temp);
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *del_callback(PyObject *, PyObject *args) {
  char *type;
  PyObject *temp;

  if (!PyArg_ParseTuple(args, (char *)"sO:del_callback", &type, &temp)) 
    return NULL;

  if (!PyCallable_Check(temp)) {
    PyErr_SetString(PyExc_TypeError, "parameter must be callable");
    return NULL;
  }
  PyObject *cblist = PyDict_GetItemString(cbdict, type);
  if (!cblist) {
    PyErr_SetString(PyExc_KeyError, type);
    return NULL;
  }
  int ind = PySequence_Index(cblist, temp);
  if (ind >= 0) {
    PySequence_DelItem(cblist, ind);
  }
  Py_INCREF(Py_None);
  return Py_None;
}

static void call_callbacks(const char *type, PyObject *arglist) {
  PyObject *cblist = PyDict_GetItemString(cbdict, (char *)type);
  if (!cblist) {
    msgErr << "Internal error: callback list " << type << " does not exist."
           << sendmsg;
    return;
  }
  for (int i=0; i<PyList_GET_SIZE(cblist); i++) {
    PyObject *obj = PyList_GET_ITEM(cblist, i);
    PyObject *result = PyEval_CallObject(obj, arglist);
    if (result == NULL) {
      PyErr_Print();
      PySequence_DelItem(cblist, i);
      i--;
    } else {
      Py_DECREF(result);
    }
  }
  Py_DECREF(arglist);
}
  
static PyMethodDef CallbackMethods[] = {
  {(char *)"add_callback", (vmdPyMethod)add_callback, METH_VARARGS },
  {(char *)"del_callback", (vmdPyMethod)del_callback, METH_VARARGS },
  {NULL, NULL}
};
  
static void initvmdcallbacks() {
  PyObject *m = Py_InitModule((char *)"vmdcallbacks", CallbackMethods);
  PyObject *dict = PyDict_New();
  PyDict_SetItemString(dict, (char *)"display_update", PyList_New(0));
  PyDict_SetItemString(dict, (char *)"frame", PyList_New(0));
  PyDict_SetItemString(dict, (char *)"initialize_structure", PyList_New(0));
  PyDict_SetItemString(dict, (char *)"molecule", PyList_New(0));
  PyDict_SetItemString(dict, (char *)"pick_atom", PyList_New(0));
  PyDict_SetItemString(dict, (char *)"pick_event", PyList_New(0));
  PyDict_SetItemString(dict, (char *)"pick_value", PyList_New(0));
  PyDict_SetItemString(dict, (char *)"timestep", PyList_New(0));
  PyDict_SetItemString(dict, (char *)"trajectory", PyList_New(0));
  PyDict_SetItemString(dict, (char *)"userkey", PyList_New(0));
  PyObject_SetAttrString(m, (char *)"callbacks", dict); 
  cbdict = dict;
}

extern "C" void initvmd(void);


PythonTextInterp::PythonTextInterp(VMDApp *vmdapp) 
: app(vmdapp) {
  msgInfo << "Starting Python..." << sendmsg;
  Py_Initialize();

  // Some modules (like Tk) assume that os.argv has been initialized
  PySys_SetArgv(app->argc_m, (char **)app->argv_m);

  set_vmdapp(app);

  // Set up the prompts
  PySys_SetObject((char *)"ps1", PyString_FromString((char *)""));
  PySys_SetObject((char *)"ps2", PyString_FromString((char *)"... "));
 
  initvmdcallbacks();
  initvmd();
  initanimate();
  initatomselection();
  initatomsel();
  initaxes();
  initcolor();
  initdisplay();
  initgraphics();
  initimd();
  initlabel();
  initmaterial();
  initmolecule();
  initmolrep();
  initmouse();
  initrender();
  inittrans();
  initvmdmenu();

#ifdef VMDNUMPY
  initvmdnumpy();
#endif

  // The VMD module imports all the above modules.
  evalString("import VMD");

  // have_tkinter and have_vmdcallback flags are set to zero if these calls
  // ever fail so that we don't fail over and over again and fill up the
  // screen with errors.
  have_tkinter = 1;
  in_tk = 0;
  needPrompt = 1;
}

PythonTextInterp::~PythonTextInterp() {
  Py_Finalize();
  msgInfo << "Done with Python." << sendmsg;
  
}

int PythonTextInterp::doTkUpdate() {
  // Don't recursively call into dooneevent - it makes Tkinter crash for
  // some infathomable reason.
  if (in_tk) return 0;
  if (have_tkinter) {
    in_tk = 1;
    int rc = evalString(
      "import Tkinter\n"
      "while Tkinter.tkinter.dooneevent(Tkinter.tkinter.DONT_WAIT):\n"
      "\tpass\n"
    );
    in_tk = 0;
    if (rc) {
      return 1; // success
    }
    // give up
    have_tkinter = 0;
  }
  return 0;
}
  
void PythonTextInterp::doEvent() {
  // Call any display loop callbacks
  // abort if the call ever fails
  PyObject *arglist = Py_BuildValue((char *)"()");
  call_callbacks("display_update", arglist);

  if (needPrompt) {
    printf(">>> ");
    fflush(stdout);
    needPrompt = 0;
  }

  if (!vmd_check_stdin()) 
	return;	
  int code = PyRun_InteractiveOne(stdin, (char *)"VMD");
  needPrompt = 1;
  if (code == E_EOF) {
    // Try to change to Tcl interpreter.  If that fails, UIText will
    // bounce us back to the Python interpreter again.
    app->textinterp_change("tcl");
  }
}

int PythonTextInterp::evalString(const char *s) {
  // evaluate the string in the interpreter
  // returns success.
  // XXX should print error message if there was one.
  return !PyRun_SimpleString((char *)s);
}

int PythonTextInterp::evalFile(const char *s) {
  FILE *fid = fopen(s, "r");
  if (!fid) { 
    msgErr << "Error opening file '" << s << "'" << sendmsg;
    return FALSE;
  }
  int code = PyRun_SimpleFile(fid, (char *)"VMD");
  fclose(fid);
  return !code;
}
 
void PythonTextInterp::frame_cb(int molid, int frame) {
  PyObject *arglist = Py_BuildValue((char *)"(i,i)", molid, frame);
  call_callbacks("frame", arglist);
}

void PythonTextInterp::initialize_structure_cb(int molid, int code) {
  PyObject *arglist = Py_BuildValue((char *)"(i,i)", molid, code);
  call_callbacks("initialize_structure", arglist);
}

void PythonTextInterp::molecule_changed_cb(int molid, int code) {
  PyObject *arglist = Py_BuildValue((char *)"(i,i)", molid, code);
  call_callbacks("molecule", arglist);
}

void PythonTextInterp::pick_atom_cb(int mol, int atom, int key_shift_state, bool ispick) {
  PyObject *arglist = Py_BuildValue((char *)"(i,i,i)", mol, atom, key_shift_state);
  call_callbacks("pick_atom", arglist);
  if (ispick) {
    // if this is a user pick event, give it its own callback event
    // to discourage devs from inappropriately overloading all pick events
    PyObject *arglist = Py_BuildValue((char *)"(i)", 1);
    call_callbacks("pick_event", arglist);
  }
}

void PythonTextInterp::pick_value_cb(float val) {
  PyObject *arglist = Py_BuildValue((char *)"(f)", val);
  call_callbacks("pick_value", arglist);
}

void PythonTextInterp::timestep_cb(int id, int frame) {
  PyObject *arglist = Py_BuildValue((char *)"(i,i)", id, frame);
  call_callbacks("timestep", arglist);
}

void PythonTextInterp::trajectory_cb(int id, const char *name) {
  PyObject *arglist = Py_BuildValue((char *)"(i,s)", id, name);
  call_callbacks("trajectory", arglist);
}

void PythonTextInterp::python_cb(const char *cmd) {
  evalString(cmd);
}

void PythonTextInterp::userkey_cb(const char *keydesc) {
  PyObject *arglist = Py_BuildValue((char *)"(s)", keydesc);
  call_callbacks("userkey", arglist);
}

