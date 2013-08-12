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
 *      $RCSfile: lammpsplugin.c,v $
 *      $Author: akohlmey $       $Locker:  $             $State: Exp $
 *      $Revision: 1.27 $       $Date: 2009/07/21 22:32:52 $
 *
 ***************************************************************************/

/*
 *  LAMMPS atom style dump file format:
 *    ITEM: TIMESTEP
 *      %d (timestep number)
 *    ITEM: NUMBER OF ATOMS
 *      %d (number of atoms)
 *    ITEM: BOX BOUNDS
 *      %f %f (boxxlo, boxxhi)
 *      %f %f (boxylo, boxyhi)
 *      %f %f (boxzlo, boxzhi)
 *    ITEM: ATOMS
 *      %d %d %f %f %f  (atomid, atomtype, x, y, z)
 *      ...
 * newer LAMMPS versions have instead
 *    ITEM: ATOMS id x y z
 *      %d %d %f %f %f  (atomid, atomtype, x, y, z)
 *      ...
 * also triclinic boxes are possible (not yet supported):
 *    ITEM: BOX BOUNDS
 *      %f %f %f (boxxlo, boxxhi, xy)
 *      %f %f %f (boxylo, boxyhi, xz)
 *      %f %f %f (boxzlo, boxzhi, yz)
 *
 * the newer format allows to handle custom dumps with velocities
 * and other features that are not yet in VMD and the molfile API.
 */

#include "largefiles.h"   /* platform dependent 64-bit file I/O defines */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "molfile_plugin.h"

#define THISPLUGIN plugin
#include "vmdconio.h"

#define VMDPLUGIN_STATIC
#include "hash.h"
#include "inthash.h"

#ifndef LAMMPS_DEBUG
#define LAMMPS_DEBUG 0
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661922
#endif

/* maximum supported length of line for line buffers */
#define LINE_LEN 1024

/* lammps item keywords */
#define KEY_ATOMS "NUMBER OF ATOMS"
#define KEY_BOX   "BOX BOUNDS"
#define KEY_DATA  "ATOMS"
#define KEY_TSTEP "TIMESTEP"

/* lammps coordinate data styles */
#define LAMMPS_COORD_NONE       0x00U
#define LAMMPS_COORD_WRAPPED    0x01U
#define LAMMPS_COORD_SCALED     0x02U
#define LAMMPS_COORD_IMAGES     0x04U
#define LAMMPS_COORD_UNWRAPPED  0x08U
#define LAMMPS_COORD_UNKNOWN    0x10U
#define LAMMPS_COORD_VELOCITIES 0x20U
#define LAMMPS_COORD_FORCES     0x40U
#define LAMMPS_COORD_TRICLINIC  0x80U

/** flags to indicate the property stored in a custom lammps dump */
#define LAMMPS_MAX_NUM_FIELDS 32
enum lammps_attribute {
  LAMMPS_FIELD_UNKNOWN=0, 
  LAMMPS_FIELD_ATOMID, LAMMPS_FIELD_MOLID,  LAMMPS_FIELD_TYPE,
  LAMMPS_FIELD_POSX,   LAMMPS_FIELD_POSY,   LAMMPS_FIELD_POSZ, 
  LAMMPS_FIELD_POSXS,  LAMMPS_FIELD_POSYS,  LAMMPS_FIELD_POSZS,
  LAMMPS_FIELD_POSXU,  LAMMPS_FIELD_POSYU,  LAMMPS_FIELD_POSZU,
  LAMMPS_FIELD_IMGX,   LAMMPS_FIELD_IMGY,   LAMMPS_FIELD_IMGZ,
  LAMMPS_FIELD_VELX,   LAMMPS_FIELD_VELY,   LAMMPS_FIELD_VELZ,
  LAMMPS_FIELD_FORX,   LAMMPS_FIELD_FORY,   LAMMPS_FIELD_FORZ,
  LAMMPS_FIELD_CHARGE, LAMMPS_FIELD_RADIUS, LAMMPS_FIELD_QUATW,
  LAMMPS_FIELD_QUATI,  LAMMPS_FIELD_QUATJ,  LAMMPS_FIELD_QUATK,
  LAMMPS_FIELD_USER0,  LAMMPS_FIELD_USER1,  LAMMPS_FIELD_USER2,
  LAMMPS_FIELD_USER3,  LAMMPS_FIELD_USER4,  LAMMPS_FIELD_USER5,
  LAMMPS_FIELD_USER6,  LAMMPS_FIELD_USER7,  LAMMPS_FIELD_USER8,
  LAMMPS_FILED_USER9
};

typedef enum lammps_attribute l_attr_t;

/* for transparent reading of .gz files */
#ifdef _USE_ZLIB
#include <zlib.h>
#define FileDesc gzFile
#define myFgets(buf,size,fd) gzgets(fd,buf,size)
#define myFprintf gzprintf
#define myFopen gzopen
#define myFclose gzclose
#define myRewind gzrewind
#else
#define FileDesc FILE*
#define myFprintf fprintf
#define myFopen fopen
#define myFclose fclose
#define myFgets(buf,size,fd) fgets(buf,size,fd)
#define myRewind rewind
#endif

typedef struct {
  FileDesc file;
  char *file_name;
  int *atomtypes;
  int numatoms;
  int nstep;
  unsigned int coord_data; /* indicate type of coordinate data   */
  int numfields;           /* number of data fields present */
  l_attr_t field[LAMMPS_MAX_NUM_FIELDS]; /* type of data fields in dumps */
  inthash_t *idmap;        /* for keeping track of atomids */
#if vmdplugin_ABIVERSION > 10
  molfile_timestep_metadata_t ts_meta;
#endif
} lammpsdata;


/* sort for integer map. initial call  id_sort(idmap, 0, natoms - 1); */
static void id_sort(int *idmap, int left, int right)
{
  int pivot, l_hold, r_hold;

  l_hold = left;
  r_hold = right;
  pivot = idmap[left];
  
  while (left < right) {
    while ((idmap[right] >= pivot) && (left < right))
      right--;
    if (left != right) {
      idmap[left] = idmap[right];
      left++;
    }
    while ((idmap[left] <= pivot) && (left < right))
      left++;
    if (left != right) {
      idmap[right] = idmap[left];
      right--;
    }
  }
  idmap[left] = pivot;
  pivot = left;
  left = l_hold;
  right = r_hold;

  if (left < pivot)
    id_sort(idmap, left, pivot-1);
  if (right > pivot)
    id_sort(idmap, pivot+1, right);
}


/** Scan the file for the next line beginning with the string "ITEM: "
 *  and returns a string containing the remainder of that line or NULL.
 *  Upon return, the file descriptor points either to the beginning 
 *  of the next line or at the first character that didn't fit into
 *  the buffer (linebuf[buflen]). */
static char* find_next_item(FileDesc fd, char* linebuf, int buflen) {
  char* ptr;

  while(myFgets(linebuf, buflen, fd)) {

    /* strip of leading whitespace */
    ptr = linebuf;
    while (ptr && (*ptr == ' ' || *ptr == '\t'))
      ++ptr;

    /* check if this is an "item" */
    if(0 == strncmp(ptr, "ITEM:", 5)) {
      ptr += 5;
      return ptr;
    }
  }

  return NULL;
}

/** Scan the file for the next occurence of a record of the type given
 *  in keyword.  If such a record is found, the file descriptor points
 *  to the beginning of the record content, and this function returns a
 *  pointer to the remainder of the line (EOL character or or additional
 *  data). otherwise a NULL pointer is returned.
 *  a pointer to a line buffer and its length have to be given.
 *  the return value will point to some location inside this buffer.
 */
static char *find_item_keyword(FileDesc fd, const char* keyword,
                               char *linebuf, int buflen) {
  char *ptr;
  int len;
  
  while(1) {
    ptr = find_next_item(fd, linebuf, buflen);

    if (ptr == NULL) 
      break;
    
    while (ptr && (*ptr == ' ' || *ptr == '\t'))
      ++ptr;

#if LAMMPS_DEBUG
    fprintf(stderr, "text=%s/%s", keyword, ptr);
#endif
    len = strlen(keyword);
    if (0 == strncmp(ptr, keyword, len) ) {
      ptr += len;
      if (*ptr == '\0' || *ptr == ' ' || *ptr == '\n' || *ptr == '\r') {
#if LAMMPS_DEBUG
        fprintf(stderr, "return=%s", ptr);
#endif
        return ptr;
      } else continue; /* keyword was not an exact match, try again. */
    }
  }
#if LAMMPS_DEBUG
  fprintf(stderr, "return='NULL'\n");
#endif
  return NULL;
}

 
static void *open_lammps_read(const char *filename, const char *filetype, 
                           int *natoms) {
  FileDesc fd;
  lammpsdata *data;
  char buffer[LINE_LEN];
  char *ptr;

  fd = myFopen(filename, "rb");
  if (!fd) return NULL;
 
  data = (lammpsdata *)calloc(1, sizeof(lammpsdata));
  data->idmap = (inthash_t *)calloc(1, sizeof(inthash_t));
  data->file = fd;
  data->file_name = strdup(filename);
  *natoms = 0;
  
  ptr = find_item_keyword(data->file, KEY_ATOMS,  buffer, LINE_LEN);
  if (ptr == NULL) {
    vmdcon_printf(VMDCON_ERROR, "lammpsplugin) Unable to find '%s' item.\n",
                  KEY_ATOMS);
    return NULL;
  }

  if (!myFgets(buffer, LINE_LEN, data->file)) {
    vmdcon_printf(VMDCON_ERROR, "lammpsplugin) dump file '%s' should "
                  "have the number of atoms after line ITEM: %s.\n", 
                  KEY_ATOMS, filename);
    return NULL;
  }
  *natoms = atoi(buffer);
 
  data->numatoms = *natoms;
  data->coord_data = LAMMPS_COORD_NONE;  
  myRewind(data->file); /* prepare for first read_timestep call */
 
  return data;
}


static int read_lammps_structure(void *mydata, int *optflags, 
                                 molfile_atom_t *atoms) {
  int i, j;
  char buffer[LINE_LEN];
  lammpsdata *data = (lammpsdata *)mydata;
  int atomid, atomtype, *idlist, needhash;
  float x, y, z;
  char *k, *fieldlist;
  
  /* clear atom info. */
  *optflags = MOLFILE_NOOPTIONS; 
  data->coord_data = LAMMPS_COORD_NONE;
  memset(atoms, 0, data->numatoms * sizeof(molfile_atom_t)); 
#if vmdplugin_ABIVERSION > 10
  data->ts_meta.count = -1;
  data->ts_meta.has_velocities = 0;
#endif
 
  /* go to the beginning of the file */
  myRewind(data->file); /* prepare for first read_timestep call */

  /* find the boundary box info to determine if triclinic or not. */
  fieldlist = find_item_keyword(data->file, KEY_BOX, buffer, LINE_LEN);
  if (fieldlist == NULL) {
    vmdcon_printf(VMDCON_ERROR, "lammpsplugin) Could not find box boundaries in timestep.\n");
    return MOLFILE_ERROR;
  }
  k = myFgets(buffer, LINE_LEN, data->file);
  if (k == NULL) {
    vmdcon_printf(VMDCON_ERROR, "lammpsplugin) Could not find box boundaries in timestep.\n");
    return MOLFILE_ERROR;
  }

  j = sscanf(buffer, "%f%f%f", &x, &y, &z);
  if (j == 3) {
    vmdcon_printf(VMDCON_WARN, "lammpsplugin) Found trajectory with triclinic box. "
                  "Periodic display will not be correct and scaled coordinates are not supported.\n");
    data->coord_data |= LAMMPS_COORD_TRICLINIC;
  } else if (j < 2) {
    vmdcon_printf(VMDCON_ERROR, "lammpsplugin) Could not find box boundaries in timestep.\n");
    return MOLFILE_ERROR;
  }

  /* find the sections with atoms */
  fieldlist = find_item_keyword(data->file, KEY_DATA, buffer, LINE_LEN);
  if (fieldlist == NULL) {
    vmdcon_printf(VMDCON_ERROR, "lammpsplugin) Couldn't find data to "
                  "read structure from file '%s'.\n", data->file_name);
    return MOLFILE_ERROR;
  }

#if LAMMPS_DEBUG  
  fprintf(stderr,"fieldlist for atoms: %s", fieldlist);
#if 0  /* simulate old style trajectory */
  fieldlist = strdup("\n");
#endif
#endif

  /* parse list of fields */
  i = 0;
  k = strtok(fieldlist, " \t\n\r");
  if (k == NULL) {
    /* assume old style lammps trajectory  */
    vmdcon_printf(VMDCON_WARN, "lammpsplugin) Found old style trajectory. "
                  "assuming data is ordered "
                  "'id type x|xs|xu y|ys|yu z|zs|zu [...]'.\n");
    data->coord_data |= LAMMPS_COORD_UNKNOWN;
  } else {
    /* try to identify supported output types */
    do {
      if (0 == strcmp(k, "id")) {
        data->field[i] = LAMMPS_FIELD_ATOMID;
      } else if (0 == strcmp(k, "mol")) {
        data->field[i] = LAMMPS_FIELD_MOLID;
      } else if (0 == strcmp(k, "type")) {
        data->field[i] = LAMMPS_FIELD_TYPE;
      } else if (0 == strcmp(k, "x")) {
        data->field[i] = LAMMPS_FIELD_POSX;
        data->coord_data |= LAMMPS_COORD_WRAPPED;
      } else if (0 == strcmp(k, "y")) {
        data->field[i] = LAMMPS_FIELD_POSY;
        data->coord_data |= LAMMPS_COORD_WRAPPED;
      } else if (0 == strcmp(k, "z")) {
        data->field[i] = LAMMPS_FIELD_POSZ;
        data->coord_data |= LAMMPS_COORD_WRAPPED;
      } else if (0 == strcmp(k, "xs")) {
        data->field[i] = LAMMPS_FIELD_POSXS;
        data->coord_data |= LAMMPS_COORD_SCALED;
      } else if (0 == strcmp(k, "ys")) {
        data->field[i] = LAMMPS_FIELD_POSYS;
        data->coord_data |= LAMMPS_COORD_SCALED;
      } else if (0 == strcmp(k, "zs")) {
        data->field[i] = LAMMPS_FIELD_POSZS;
        data->coord_data |= LAMMPS_COORD_SCALED;
      } else if (0 == strcmp(k, "xu")) {
        data->field[i] = LAMMPS_FIELD_POSXU;
        data->coord_data |= LAMMPS_COORD_UNWRAPPED;
      } else if (0 == strcmp(k, "yu")) {
        data->field[i] = LAMMPS_FIELD_POSYU;
        data->coord_data |= LAMMPS_COORD_UNWRAPPED;
      } else if (0 == strcmp(k, "zu")) {
        data->field[i] = LAMMPS_FIELD_POSZU;
        data->coord_data |= LAMMPS_COORD_UNWRAPPED;
      } else if (0 == strcmp(k, "ix")) {
        data->field[i] = LAMMPS_FIELD_IMGX;
        data->coord_data |= LAMMPS_COORD_IMAGES;
      } else if (0 == strcmp(k, "iy")) {
        data->field[i] = LAMMPS_FIELD_IMGY;
        data->coord_data |= LAMMPS_COORD_IMAGES;
      } else if (0 == strcmp(k, "iz")) {
        data->field[i] = LAMMPS_FIELD_IMGZ;
        data->coord_data |= LAMMPS_COORD_IMAGES;
      } else if (0 == strcmp(k, "vx")) {
        data->field[i] = LAMMPS_FIELD_VELX;
#if vmdplugin_ABIVERSION > 10
        data->coord_data |= LAMMPS_COORD_VELOCITIES;
        data->ts_meta.has_velocities = 1;
#endif
      } else if (0 == strcmp(k, "vy")) {
        data->field[i] = LAMMPS_FIELD_VELY;
#if vmdplugin_ABIVERSION > 10
        data->coord_data |= LAMMPS_COORD_VELOCITIES;
        data->ts_meta.has_velocities = 1;
#endif
      } else if (0 == strcmp(k, "vz")) {
        data->field[i] = LAMMPS_FIELD_VELZ;
#if vmdplugin_ABIVERSION > 10
        data->coord_data |= LAMMPS_COORD_VELOCITIES;
        data->ts_meta.has_velocities = 1;
#endif
      } else if (0 == strcmp(k, "fx")) {
        data->field[i] = LAMMPS_FIELD_FORX;
        data->coord_data |= LAMMPS_COORD_FORCES;
      } else if (0 == strcmp(k, "fy")) {
        data->field[i] = LAMMPS_FIELD_FORY;
        data->coord_data |= LAMMPS_COORD_FORCES;
      } else if (0 == strcmp(k, "fz")) {
        data->field[i] = LAMMPS_FIELD_FORZ;
        data->coord_data |= LAMMPS_COORD_FORCES;
      } else if (0 == strcmp(k, "q")) {
        data->field[i] = LAMMPS_FIELD_CHARGE;
        *optflags |= MOLFILE_CHARGE; 
      } else if (0 == strcmp(k, "radius")) {
        data->field[i] = LAMMPS_FIELD_CHARGE;
        *optflags |= MOLFILE_RADIUS; 
      } else {
        data->field[i] = LAMMPS_FIELD_UNKNOWN;
      }
      ++i;
      data->numfields = i;
      k = strtok(NULL," \t\n\r");
    } while ((k != NULL) && (i < LAMMPS_MAX_NUM_FIELDS));
  
    vmdcon_printf(VMDCON_INFO, "lammpsplugin) New style dump with %d data "
                  "fields. 0x%02x\n", data->numfields, data->coord_data);
  }

  idlist = (int *)malloc(data->numatoms * sizeof(int));

  /* read and parse ATOMS data section */
  for(i=0; i<data->numatoms; i++) {
    k = myFgets(buffer, LINE_LEN, data->file);

    if (k == NULL) { 
      vmdcon_printf(VMDCON_ERROR, "lammpsplugin) Error while reading "
                    "structure from lammps dump file '%s': atom missing in "
                    "the first timestep\n", data->file_name);
      vmdcon_printf(VMDCON_ERROR, "lammpsplugin) expecting '%d' atoms, "
                    "found only '%d'\n", data->numatoms, i+1);
      free(idlist);
      return MOLFILE_ERROR;
    }

    /* if we have an old-style trajectory we have to guess what is there.
     * this chunk of code should only be executed once. LAMMPS_COORD_UNKNOWN
     * will be kept set until the very end or when we find that one position
     * is outside the box. */
    if ( (data->coord_data == (LAMMPS_COORD_UNKNOWN|LAMMPS_COORD_TRICLINIC) ) 
         || (data->coord_data == LAMMPS_COORD_UNKNOWN) ) {
      int ix, iy, iz;
      j = sscanf(buffer, "%d%d%f%f%f%d%d%d", &atomid, &atomtype, 
                 &x, &y, &z, &ix, &iy, &iz);
      if (j > 4) {  /* assume id type xs ys zs .... format */
        data->coord_data |= LAMMPS_COORD_SCALED;
        data->numfields = 5;
        data->field[0] = LAMMPS_FIELD_ATOMID;
        data->field[1] = LAMMPS_FIELD_TYPE;
        data->field[2] = LAMMPS_FIELD_POSXS;
        data->field[3] = LAMMPS_FIELD_POSYS;
        data->field[4] = LAMMPS_FIELD_POSZS;
      } else {
        vmdcon_printf(VMDCON_ERROR, "lammpsplugin) Error while reading "
                      "structure from lammps dump file '%s'. Unsupported "
                      "dump file format.\n", data->file_name);
        free(idlist);
        return MOLFILE_ERROR;
      }
    }

    /* some defaults */
    atoms[i].resid = 0; /* mapped to MolID, if present */
    strncpy(atoms[i].resname, "UNK", 4);
    strncpy(atoms[i].chain, "",1);
    strncpy(atoms[i].segid, "",1);
    atomid = i; /* required if there is no atomid in a custom dump. */

    /* parse the line of data */
    j = 0;
    k = strtok(buffer, " \t\n\r");
    while ((k != NULL) && (j < data->numfields)) {
      switch (data->field[j]) {

        case LAMMPS_FIELD_ATOMID:
          atomid = atoi(k) - 1; /* convert to 0 based list */
          break;

        case LAMMPS_FIELD_TYPE:
          strncpy(atoms[i].type, k, 16); 
          strncpy(atoms[i].name, k, 16);
          atoms[i].type[15] = '\0';
          atoms[i].name[15] = '\0';
          /* WARNINGWARNINGWARNINGWARNINGWARNINGWARNINGWARNINGWARNINGWARNING *\
           * Don't try using the atomid as name. This will waste a _lot_ of  *
           * memory due to names being stored in a string hashtable.         *
           * VMD currently cannot handle changing atomids anyways. We thus   *
           * use a hash table to track atom ids. Within VMD the atomids are  *
           * then lost, but atoms can be identified uniquely via 'serial'    *
           * or 'index' atom properties.
           * WARNINGWARNINGWARNINGWARNINGWARNINGWARNINGWARNINGWARNINGWARNING*/
          break;
          
        case LAMMPS_FIELD_MOLID:
          atoms[i].resid = atoi(k);
          break;

        case LAMMPS_FIELD_CHARGE:
          atoms[i].charge = atof(k);
          break;

        case LAMMPS_FIELD_RADIUS:
          atoms[i].radius = atof(k);
          break;

        case LAMMPS_FIELD_UNKNOWN: /* fallthrough */
        default:
          break;                /* do nothing */
      }
      ++j;
      k = strtok(NULL, " \t\n\r");
    }

    idlist[i] = atomid;

    /* for old-style files, we have to use some heuristics to determine
     * if we have scaled or unscaled (absolute coordinates).
     * we assume scaled unless proven differently, and we assume unwrapped
     * unless we have images present. */
    if ( (data->coord_data & LAMMPS_COORD_UNKNOWN) != 0) {
      x=y=z=0.0f;
      j = sscanf(buffer, "%*d%*d%f%f%f", &x, &y, &z);
      if ((x<-0.1) || (x>1.1) || (y<-0.1) || (y>1.1) 
          || (z<-0.1) || (x>1.1)) {
        data->coord_data &= ~LAMMPS_COORD_UNKNOWN;
        if ((data->coord_data & LAMMPS_COORD_IMAGES) != 0) {
          data->coord_data |= LAMMPS_COORD_WRAPPED;
          data->field[2] = LAMMPS_FIELD_POSX;
          data->field[3] = LAMMPS_FIELD_POSY;
          data->field[4] = LAMMPS_FIELD_POSZ;
        } else {
          data->coord_data |= LAMMPS_COORD_UNWRAPPED;
          data->field[2] = LAMMPS_FIELD_POSXU;
          data->field[3] = LAMMPS_FIELD_POSYU;
          data->field[4] = LAMMPS_FIELD_POSZU;
        }
      }
    }
  }
  data->coord_data &= ~LAMMPS_COORD_UNKNOWN;

  /* pick coordinate type that we want to read
   * and disable the rest. we want unwrapped > wrapped > scaled. */
  if (data->coord_data & LAMMPS_COORD_UNWRAPPED) {
    data->coord_data &= ~(LAMMPS_COORD_WRAPPED|LAMMPS_COORD_SCALED
                          |LAMMPS_COORD_IMAGES|LAMMPS_COORD_TRICLINIC);
  } else if (data->coord_data & LAMMPS_COORD_WRAPPED) {
    data->coord_data &= ~(LAMMPS_COORD_UNWRAPPED|LAMMPS_COORD_SCALED
                          |LAMMPS_COORD_TRICLINIC);
  } else if (data->coord_data & LAMMPS_COORD_TRICLINIC) {
    vmdcon_printf(VMDCON_ERROR, "lammpsplugin) Scaled coordinates with "
                  "triclinic box are not (yet) supported.\n");
    return MOLFILE_ERROR;
  } else {
    data->coord_data &= ~(LAMMPS_COORD_UNWRAPPED|LAMMPS_COORD_WRAPPED|LAMMPS_COORD_TRICLINIC);
  }

  if (data->coord_data == LAMMPS_COORD_NONE) {
    vmdcon_printf(VMDCON_ERROR, "lammpsplugin) No supported coordinate data "
                  "found in lammps dump file '%s'.\n", data->file_name);
    return MOLFILE_ERROR;
  }
  
  if (data->coord_data & LAMMPS_COORD_SCALED) {
    vmdcon_printf(VMDCON_INFO, "lammpsplugin) Reconstructing atomic "
                  "coordinates from fractional coordinates and box size.\n");
  } else {
    vmdcon_printf(VMDCON_INFO, "lammpsplugin) Using absolute atomic "
                  "coordinates directly.\n");
  }

#if vmdplugin_ABIVERSION > 10
  if (data->coord_data & LAMMPS_COORD_VELOCITIES) {
    vmdcon_printf(VMDCON_INFO, "lammpsplugin) Importing atomic velocities.\n");
  }
#endif

#if 0
  /* XXX: the sorting step breaks the construction of a hash. we'd have to 
     copy the list and waste more memory to figure it out. for the time 
     being, we just force using a hash until there is a better solution. */

  /* sort list of atomids and figure out if we need the hash table */
  id_sort(idlist, 0, data->numatoms-1);
  needhash=0;
  for (i=0; i < data->numatoms; ++i)
    if (idlist[i] != i) needhash=1;
#else
  needhash=1;
#endif    
  /* set up an integer hash to keep a sorted atom id map */
  if (needhash) {
    vmdcon_printf(VMDCON_INFO, "lammpsplugin) Using hash table to track "
                  "atom identities.\n");
    inthash_init(data->idmap, data->numatoms);
    for (i=0; i < data->numatoms; ++i) {
      if (inthash_insert(data->idmap, idlist[i], i) != HASH_FAIL) {
        vmdcon_printf(VMDCON_ERROR, "lammpsplugin) Duplicate atomid %d or "
                      "unsupported dump file format.\n", idlist[i]);
        free(idlist);
        return MOLFILE_ERROR;
      }
    }
  } else {
    free(data->idmap);
    data->idmap = NULL;
  }
  free(idlist);
  
  myRewind(data->file);
  return MOLFILE_SUCCESS;
}

#if vmdplugin_ABIVERSION > 10
/***********************************************************/
static int read_timestep_metadata(void *mydata,
                                  molfile_timestep_metadata_t *meta) {
  lammpsdata *data = (lammpsdata *)mydata;
  
  meta->count = -1;
  meta->has_velocities = data->ts_meta.has_velocities;
  if (meta->has_velocities) {
    vmdcon_printf(VMDCON_INFO, "lammpsplugin) Importing velocities from "
                      "custom LAMMPS dump file.\n");
  }
  return MOLFILE_SUCCESS;
}
#endif


static int read_lammps_timestep(void *mydata, int natoms, molfile_timestep_t *ts) {
  int i, j;
  char buffer[LINE_LEN];
  float x, y, z, vx, vy, vz;
  int atomid;
  float xlo, ylo, zlo, hi;

  lammpsdata *data = (lammpsdata *)mydata;

  /* check if there is another time step in the file. */
  if (NULL == find_item_keyword(data->file, KEY_TSTEP, buffer, LINE_LEN)) 
    return MOLFILE_ERROR;
 
  /* check if we should read or skip this step. */
  if (!ts) return MOLFILE_SUCCESS;

  /* search for the number of atoms in the timestep */
  if (NULL==find_item_keyword(data->file, KEY_ATOMS, buffer, LINE_LEN)) {
    vmdcon_printf(VMDCON_ERROR, "lammpsplugin) Unable to find item: %s for "
                  "current timestep in file %s.\n", KEY_ATOMS, data->file_name);
    return MOLFILE_ERROR;
  }

  if (!myFgets(buffer, LINE_LEN, data->file)) {
    vmdcon_printf(VMDCON_ERROR, "lammpsplugin) Premature EOF for %s.\n", data->file_name);
    return MOLFILE_ERROR;
  }

  if (natoms != atoi(buffer)) {
    vmdcon_printf(VMDCON_ERROR, "lammpsplugin) Inconsistent number of atoms in timestep.\n");
    return MOLFILE_ERROR;
  }

  /* now read the boundary box of the timestep */
  if (NULL == find_item_keyword(data->file, KEY_BOX, buffer, LINE_LEN)) {
    vmdcon_printf(VMDCON_ERROR, "lammpsplugin) Could not find box boundaries in timestep.\n");
    return MOLFILE_ERROR;
  }

  if (NULL == myFgets(buffer, LINE_LEN, data->file)) return MOLFILE_ERROR;
  sscanf(buffer,"%f%f", &xlo, &hi);
  ts->A = hi-xlo;

  if (NULL == myFgets(buffer, LINE_LEN, data->file)) return MOLFILE_ERROR;
  sscanf(buffer,"%f%f", &ylo, &hi);
  ts->B = hi-ylo;

  if (NULL == myFgets(buffer, LINE_LEN, data->file)) return MOLFILE_ERROR;
  sscanf(buffer,"%f%f", &zlo, &hi);
  ts->C = hi-zlo;

  /* XXX: non-orthogonal boxes are not supported yet. */
  ts->alpha = 90.0;
  ts->beta  = 90.0;
  ts->gamma = 90.0;

  /* read the coordinates */
  if (NULL == find_item_keyword(data->file, KEY_DATA, buffer, LINE_LEN)) {
    vmdcon_printf(VMDCON_ERROR, "lammpsplugin) could not find atom data for timestep.\n");
    return MOLFILE_ERROR;
  }

  for (i=0; i<natoms; i++) {
    float ix, iy, iz;
    char *k;
    
    k = myFgets(buffer, LINE_LEN, data->file);

    if (k == NULL) { 
      vmdcon_printf(VMDCON_ERROR, "lammpsplugin) Error while reading "
                    "data from lammps dump file '%s'.\n", data->file_name);
      vmdcon_printf(VMDCON_ERROR, "lammpsplugin) expecting '%d' atoms, "
                    "found only '%d'\n", natoms, i+1);
      return MOLFILE_ERROR;
    }

    x=y=z=ix=iy=iz=vx=vy=vz=0.0f;
    atomid=i;
    
    /* parse the line of data */
    j = 0;
    k = strtok(buffer, " \t\n\r");
    while ((k != NULL) && (j < data->numfields)) {
      switch (data->field[j]) {

        case LAMMPS_FIELD_ATOMID:
          atomid = atoi(k) - 1;
          break;

        case LAMMPS_FIELD_POSX:
          if (data->coord_data & LAMMPS_COORD_WRAPPED)
            x = atof(k);
          break;

        case LAMMPS_FIELD_POSY:
          if (data->coord_data & LAMMPS_COORD_WRAPPED)
            y = atof(k);
          break;

        case LAMMPS_FIELD_POSZ:
          if (data->coord_data & LAMMPS_COORD_WRAPPED)
            z = atof(k);
          break;

        case LAMMPS_FIELD_POSXU:
          if (data->coord_data & LAMMPS_COORD_UNWRAPPED)
            x = atof(k);
          break;

        case LAMMPS_FIELD_POSYU:
          if (data->coord_data & LAMMPS_COORD_UNWRAPPED)
            y = atof(k);
          break;

        case LAMMPS_FIELD_POSZU:
          if (data->coord_data & LAMMPS_COORD_UNWRAPPED)
            z = atof(k);
          break;

        case LAMMPS_FIELD_POSXS:
          if (data->coord_data & LAMMPS_COORD_SCALED)
            x = atof(k);
          break;

        case LAMMPS_FIELD_POSYS:
          if (data->coord_data & LAMMPS_COORD_SCALED)
            y = atof(k);
          break;

        case LAMMPS_FIELD_POSZS:
          if (data->coord_data & LAMMPS_COORD_SCALED)
            z = atof(k);
          break;

        case LAMMPS_FIELD_IMGX:
          if (data->coord_data & LAMMPS_COORD_IMAGES)
            ix = atof(k);
          break;

        case LAMMPS_FIELD_IMGY:
          if (data->coord_data & LAMMPS_COORD_IMAGES)
            iy = atof(k);
          break;

        case LAMMPS_FIELD_IMGZ:
          if (data->coord_data & LAMMPS_COORD_IMAGES)
            iz = atof(k);
          break;

#if vmdplugin_ABIVERSION > 10
        case LAMMPS_FIELD_VELX:
          vx = atof(k);
          break;

        case LAMMPS_FIELD_VELY:
          vy = atof(k);
          break;

        case LAMMPS_FIELD_VELZ:
          vz = atof(k);
          break;
#endif

        default: /* do nothing */
          break;
      }

      ++j;
      k = strtok(NULL, " \t\n\r");
    } 
    
    if (data->idmap != NULL) {
      j = inthash_lookup(data->idmap, atomid);
    } else {
      j = atomid;
    }
    
    if ((j == HASH_FAIL) || (j < 0) || (j >= data->numatoms)) {
      vmdcon_printf(VMDCON_WARN, "lammpsplugin) ignoring atom with "
                    "yet unknown id %d\n", j);
    } else {
      /* copy coordinates. we may have coordinates in different
       * formats available in custom dumps. those have been checked
       * before and we prefer to use unwrapped > wrapped > scaled 
       * in this order. in the second two cases, we also apply image
       * shifts, if that data is available. unnecessary or unsupported
       * combinations of flags have been cleared based on data in 
       * the first frame in read_lammps_structure(). */
      int addr = 3 * j;
      if (data->coord_data & LAMMPS_COORD_SCALED) {
        /* we have fractional coordinates, so they need 
         * to be scaled by a/b/c etc. */
        ts->coords[addr    ] = xlo + x * ts->A;
        ts->coords[addr + 1] = ylo + y * ts->B;
        ts->coords[addr + 2] = zlo + z * ts->C;
      } else {
        /* ... but they can also be absolute values */
        ts->coords[addr    ] = x;
        ts->coords[addr + 1] = y;
        ts->coords[addr + 2] = z;
      }
      if (data->coord_data & LAMMPS_COORD_IMAGES) {
        /* we have image counter data to unwrap coordinates. */
        ts->coords[addr    ] += ix * ts->A;
        ts->coords[addr + 1] += iy * ts->B;
        ts->coords[addr + 2] += iz * ts->C;
      }
#if vmdplugin_ABIVERSION > 10
      if (ts->velocities != NULL) {
        /* we have image counter data to unwrap coordinates. */
        ts->velocities[addr    ] = vx;
        ts->velocities[addr + 1] = vy;
        ts->velocities[addr + 2] = vz;
      }
#endif
    }
  }

  return MOLFILE_SUCCESS;
}
    
static void close_lammps_read(void *mydata) {
  lammpsdata *data = (lammpsdata *)mydata;
  myFclose(data->file);
  free(data->file_name);
#if LAMMPS_DEBUG
  if (data->idmap != NULL) 
    fprintf(stderr, "inthash stats: %s\n", inthash_stats(data->idmap));
#endif
  if (data->idmap != NULL) {
    inthash_destroy(data->idmap);
    free(data->idmap);
  }
  free(data);
}

static void *open_lammps_write(const char *filename, const char *filetype, 
                           int natoms) {
  FileDesc fd;
  lammpsdata *data;

  fd = myFopen(filename, "w");
  if (!fd) { 
    vmdcon_printf(VMDCON_ERROR, "Error) Unable to open lammpstrj file %s for writing\n",
            filename);
    return NULL;
  }
  
  data = (lammpsdata *)malloc(sizeof(lammpsdata));
  data->numatoms = natoms;
  data->file = fd;
  data->file_name = strdup(filename);
  data->nstep = 0;
  return data;
}

static int write_lammps_structure(void *mydata, int optflags, 
                               const molfile_atom_t *atoms) {
  lammpsdata *data = (lammpsdata *)mydata;
  int i, j;
  hash_t atomtypehash;

  hash_init(&atomtypehash,128);

  /* generate 1 based lookup table for atom types */
  for (i=0, j=1; i < data->numatoms; i++)
    if (hash_insert(&atomtypehash, atoms[i].type, j) == HASH_FAIL)
      j++;
  
  data->atomtypes = (int *) malloc(data->numatoms * sizeof(int));

  for (i=0; i < data->numatoms ; i++)
    data->atomtypes[i] = hash_lookup(&atomtypehash, atoms[i].type);

  hash_destroy(&atomtypehash);
  
  return MOLFILE_SUCCESS;
}

static int write_lammps_timestep(void *mydata, const molfile_timestep_t *ts) {
  lammpsdata *data = (lammpsdata *)mydata; 
  const float *pos;
  int i;

  myFprintf(data->file, "ITEM: TIMESTEP\n");
  myFprintf(data->file, "%d\n", data->nstep);
  myFprintf(data->file, "ITEM: NUMBER OF ATOMS\n");
  myFprintf(data->file, "%d\n", data->numatoms);
  myFprintf(data->file, "ITEM: BOX BOUNDS\n");
  myFprintf(data->file, "%g %g\n", 0.0, ts->A);
  myFprintf(data->file, "%g %g\n", 0.0, ts->B);
  myFprintf(data->file, "%g %g\n", 0.0, ts->C);
  myFprintf(data->file, "ITEM: ATOMS id type x y z\n");

  pos = ts->coords;
  
  for (i = 0; i < data->numatoms; ++i) {
    myFprintf(data->file, " %d %d %g %g %g\n", 
            i+1, data->atomtypes[i], pos[0], pos[1], pos[2]);
    pos += 3;
  }

  data->nstep ++;
  return MOLFILE_SUCCESS;
}


static void close_lammps_write(void *mydata) {
  lammpsdata *data = (lammpsdata *)mydata;

  myFclose(data->file);
  free(data->atomtypes);
  free(data->file_name);
  free(data);
}


/* registration stuff */
static molfile_plugin_t plugin;

VMDPLUGIN_API int VMDPLUGIN_init() {
  memset(&plugin, 0, sizeof(molfile_plugin_t));
  plugin.abiversion = vmdplugin_ABIVERSION;
  plugin.type = MOLFILE_PLUGIN_TYPE;
  plugin.name = "lammpstrj";
  plugin.prettyname = "LAMMPS Trajectory";
  plugin.author = "Marco Kalweit, Axel Kohlmeyer, Lutz Maibaum, John Stone";
  plugin.majorv = 0;
  plugin.minorv = 11;
  plugin.is_reentrant = VMDPLUGIN_THREADUNSAFE;
#ifdef _USE_ZLIB
  plugin.filename_extension = "lammpstrj,lammpstrj.gz";
#else
  plugin.filename_extension = "lammpstrj";
#endif
  plugin.open_file_read = open_lammps_read;
  plugin.read_structure = read_lammps_structure;
  plugin.read_next_timestep = read_lammps_timestep;
#if vmdplugin_ABIVERSION > 10
  plugin.read_timestep_metadata    = read_timestep_metadata;
#endif
  plugin.close_file_read = close_lammps_read;
  plugin.open_file_write = open_lammps_write;
  plugin.write_structure = write_lammps_structure;
  plugin.write_timestep = write_lammps_timestep;
  plugin.close_file_write = close_lammps_write;

  return VMDPLUGIN_SUCCESS;
}

VMDPLUGIN_API int VMDPLUGIN_register(void *v, vmdplugin_register_cb cb) {
  (*cb)(v, (vmdplugin_t *)&plugin);
  return VMDPLUGIN_SUCCESS;
}

VMDPLUGIN_API int VMDPLUGIN_fini() {
  return VMDPLUGIN_SUCCESS;
}


#ifdef TEST_PLUGIN

int main(int argc, char *argv[]) {
  molfile_timestep_t timestep;
  molfile_atom_t *atoms = NULL;
  void *v;
  int natoms;
  int i, j, opts;
#if vmdplugin_ABIVERSION > 10
  molfile_timestep_metadata_t ts_meta;
#endif

  while (--argc) {
    ++argv;
    v = open_lammps_read(*argv, "lammps", &natoms);
    if (!v) {
      fprintf(stderr, "open_lammps_read failed for file %s\n", *argv);
      return 1;
    }
    fprintf(stderr, "open_lammps_read succeeded for file %s\n", *argv);
    fprintf(stderr, "number of atoms: %d\n", natoms);

    timestep.coords = (float *)malloc(3*sizeof(float)*natoms);
    atoms = (molfile_atom_t *)malloc(sizeof(molfile_atom_t)*natoms);
    if (read_lammps_structure(v, &opts, atoms) == MOLFILE_ERROR) {
      close_lammps_read(v);
      continue;
    }
      
    fprintf(stderr, "read_lammps_structure: options=0x%08x\n", opts);
#if 0
    for (i=0; i<natoms; ++i) {
      fprintf(stderr, "atom %09d: name=%s, type=%s, resname=%s, resid=%d, segid=%s, chain=%s\n",
                      i, atoms[i].name, atoms[i].type, atoms[i].resname, atoms[i].resid,
                      atoms[i].segid, atoms[i].chain);
    }
#endif
#if vmdplugin_ABIVERSION > 10
    read_timestep_metadata(v,&ts_meta);
    if (ts_meta.has_velocities) {
      fprintf(stderr, "found timestep velocities metadata.\n");
    }
    timestep.velocities = (float *) malloc(3*natoms*sizeof(float));
#endif
    j = 0;
    while (!read_lammps_timestep(v, natoms, &timestep)) {
      for (i=0; i<10; ++i) {
        fprintf(stderr, "atom %09d: type=%s, resid=%d, "
                      "x/y/z = %.3f %.3f %.3f "
#if vmdplugin_ABIVERSION > 10
                      "vx/vy/vz = %.3f %.3f %.3f "
#endif
                      "\n",
                      i, atoms[i].type, atoms[i].resid, 
                      timestep.coords[3*i], timestep.coords[3*i+1], 
                      timestep.coords[3*i+2]
#if vmdplugin_ABIVERSION > 10
                      ,timestep.velocities[3*i], timestep.velocities[3*i+1], 
                      timestep.velocities[3*i+2]
#endif
                      );    
      }
      j++;
    }
    fprintf(stderr, "ended read_next_timestep on frame %d\n", j);

    close_lammps_read(v);
  }
#if vmdplugin_ABIVERSION > 10
  free(timestep.velocities);
#endif
  free(timestep.coords);
  free(atoms);
  return 0;
}

#endif
