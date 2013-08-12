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
 *      $RCSfile: macosxvmdstart.C,v $
 *      $Author: johns $        $Locker:  $             $State: Exp $
 *      $Revision: 1.17 $      $Date: 2009/04/29 15:43:36 $
 *
 ***************************************************************************
 * DESCRIPTION:
 *  MacOS X startup code
 ***************************************************************************/

// only compile this file if we're building on MacOS X 
// and when the target build is meant for an application bundle install
// rather than a traditional X11/Unix style VMD install
#if !defined(VMDNOMACBUNDLE) && (defined(ARCH_MACOSX) || defined(ARCH_MACOSXX86))
#include <Carbon/Carbon.h>    /* Carbon APIs for process management */
#include <stdlib.h>
#include <stdio.h>            
#include "utilities.h"        /* string/filename manipulation routines */

OSErr GetApplicationBundleFSSpec(FSSpecPtr theFSSpecPtr) {
   OSErr err;
   ProcessSerialNumber psn;
   err = GetCurrentProcess(&psn);
   if (err != noErr) return err;

   FSRef location;
   err = GetProcessBundleLocation(&psn, &location);
   if (err != noErr) return err;

   return FSGetCatalogInfo(&location, kFSCatInfoNone, 
     NULL, NULL, theFSSpecPtr, NULL);
}

OSErr GetApplicationPackageFSSpecFromBundle(FSSpecPtr theFSSpecPtr) {
  OSErr err = fnfErr;
  CFBundleRef myAppsBundle = CFBundleGetMainBundle();
  if (myAppsBundle == NULL) return err;
  CFURLRef myBundleURL = CFBundleCopyBundleURL(myAppsBundle);
  if (myBundleURL == NULL) return err;

  FSRef myBundleRef;
  Boolean ok = CFURLGetFSRef(myBundleURL, &myBundleRef);
  CFRelease(myBundleURL);
  if (!ok) return err;

  return FSGetCatalogInfo(&myBundleRef, kFSCatInfoNone,
    NULL, NULL, theFSSpecPtr, NULL);
}

static char * vmd_get_vmddir(void) {
  FSSpec spec;
  FSRef ref;
  UInt8 * path;
  OSErr rc;
  char *bundledir;
  char *vmddir;
  char *tmp;

  bundledir = (char *) malloc(2048 * sizeof(UInt8));
  memset(bundledir, 0, 2048 * sizeof(UInt8));

#if 0
  if (!(rc = GetApplicationPackageFSSpecFromBundle(&spec))) {
#else
  if (!(rc = GetApplicationBundleFSSpec(&spec))) {
#endif
    rc = FSpMakeFSRef(&spec, &ref);
    if (rc) printf("makefsref OSErr: %d\n", rc);
      
    rc = FSRefMakePath(&ref,(UInt8 *) bundledir, 2048);
    if (rc) printf("makepath OSErr: %d\n", rc);
  } else {
    printf("getappbundlepath OSErr: %d\n", rc);
  }

  if (rc) {
    free(bundledir);
    return NULL;
  } 

  // truncate bundle path to parent bundle directory
  if ((tmp = strrchr(bundledir, '/')) != NULL) {
    tmp[0] = '\0';
  }
  if ((tmp = strrchr(bundledir, '/')) != NULL) {
    tmp[0] = '\0';
  }

  // add "/vmd" to parent bundle directory
  vmddir = (char *) malloc(strlen(bundledir) + 1 + strlen("/vmd"));
  strcpy(vmddir, bundledir);
  strcat(vmddir, "/vmd");

  free(bundledir);

  return (char *) vmddir;
}


#if 1
int macosxvmdstart(int argc, char **argv) {
#else
int main(int argc, char **argv) {
#endif
  char tmp[8192];
  char * vmddir;
  int i;

  vmddir = vmd_get_vmddir();
  if (vmddir == NULL) {
    return -1; // fail and exit
  }

#if 0
  if (!getenv("MACOSXVMDSTARTUP")) {
    int startterminal=1; // be default, we start one...

    setenv("MACOSXVMDSTARTUP", "1", 1);
    // check for -dispdev text, in which case we don't start a terminal...
    for (i=0; i < argc; i++) {
      if (!strupcmp(argv[i], "-dispdev")) {
        if (argc > i+1) {
          if (!strupcmp(argv[i+1], "text")) {
            startterminal=0;
          }
        } 
      }
    }

    if (startterminal) {
      char cmdbuf[16384];
      sprintf(cmdbuf, "\"%s/vmd_MACOSX\"", vmddir);
      if (argc > 1) {
        for (i=1; i < argc; i++) {
          strcat(cmdbuf, " ");
          strcat(cmdbuf, argv[i]);
        }
      }
      strcat(cmdbuf, " &");

printf("Executing VMD startup command: %s\n", cmdbuf);
      exit(system(cmdbuf));
    }
  }
#endif

  if (!getenv("VMDDIR")) {
    setenv("VMDDIR", vmddir, 1);
  }

  if (!getenv("TCL_LIBRARY")) {
    strcpy(tmp, vmddir);
    strcat(tmp, "/scripts/tcl");
    setenv("TCL_LIBRARY", tmp, 1);
  }

  if (!getenv("TK_LIBRARY")) {
    strcpy(tmp, vmddir);
    strcat(tmp, "/scripts/tk");
    setenv("TK_LIBRARY", tmp, 1);
  }

  if (!getenv("PYTHONPATH")) {
    strcpy(tmp, vmddir);
    strcat(tmp, "/scripts/python");
    setenv("PYTHONPATH", tmp, 1);
  } else {
    strcpy(tmp, getenv("PYTHONPATH"));
    strcat(tmp, ":");
    strcat(tmp, vmddir);
    strcat(tmp, "/scripts/python");
    setenv("PYTHONPATH", tmp, 1);
  }

  if (!getenv("STRIDE_BIN")) {
    strcpy(tmp, vmddir);
#if defined(ARCH_MACOSXX86)
    strcat(tmp, "/stride_MACOSXX86");
#else
    strcat(tmp, "/stride_MACOSX");
#endif
    setenv("STRIDE_BIN", tmp, 1);
  }

  if (!getenv("SURF_BIN")) {
    strcpy(tmp, vmddir);
#if defined(ARCH_MACOSXX86)
    strcat(tmp, "/surf_MACOSXX86");
#else
    strcat(tmp, "/surf_MACOSX");
#endif
    setenv("SURF_BIN", tmp, 1);
  }

  if (!getenv("TACHYON_BIN")) {
    strcpy(tmp, vmddir);
#if defined(ARCH_MACOSXX86)
    strcat(tmp, "/tachyon_MACOSXX86");
#else
    strcat(tmp, "/tachyon_MACOSX");
#endif
    setenv("TACHYON_BIN", tmp, 1);
  }

  return 0;
}


#endif

