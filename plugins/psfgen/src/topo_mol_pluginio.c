#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "topo_mol_pluginio.h"
#include "topo_mol_struct.h"
#include "extract_alias.h"

#if defined(WIN32)
#define snprintf _snprintf
#endif

#if !defined(PSFGEN_USEPLUGINS)

/*
 * If psfgen is compiled without plugin support, simply emit an 
 * error message and return an error code so scripts don't continue.
 */
int topo_mol_read_plugin(topo_mol *mol, const char *pluginname,
                         const char *filename, 
                         const char *segid, stringhash *h,
                         int coordinatesonly, int residuesonly,
                         void *v, void (*print_msg)(void *, const char *)) {
  print_msg(v, "ERROR: Plugin I/O not available in this build of psfgen.");
  return -1;
}

int topo_mol_write_plugin(topo_mol *mol, const char *pluginname,
                          const char *filename, 
                          void *v, void (*print_msg)(void *, const char *)) {
  print_msg(v, "ERROR: Plugin I/O not available in this build of psfgen.");
  return -1;
}


#else 

/*
 * Plugin header files; get plugin source from www.ks.uiuc.edu/Research/vmd"
 */
#include "libmolfile_plugin.h"
#include "molfile_plugin.h"
#include "hash.h"

/*
 * XXX this is very hackish and needs to be rewritten
 */ 
#define MAX_PLUGINS 200
static hash_t pluginhash;
static int num_plugins=0;
static molfile_plugin_t *plugins[200];


static void strtoupper(char *s) {
  while ( *s ) { *s = toupper(*s); ++s; }
}


static void strstripspaces(char *s) {
  int len, i;

  while((len = strlen(s)) > 0 && s[len-1] == ' ')
    s[len-1] = '\0';

  while(len > 0 && s[0] == ' ') {
    for(i=0; i < len; i++)  
      s[i] = s[i+1];
    len--;
  }
}

static int register_cb(void *v, vmdplugin_t *p) {
  const char *key = p->name;
  if (num_plugins >= MAX_PLUGINS) {
    fprintf(stderr, "Exceeded maximum allowed number of plugins; recompile. :(\n");
    return VMDPLUGIN_ERROR;
  }
  if (hash_insert(&pluginhash, key, num_plugins) != HASH_FAIL) {
    fprintf(stderr, "Multiple plugins for file type '%s' found!", key);
    return VMDPLUGIN_ERROR;
  }
  plugins[num_plugins++] = (molfile_plugin_t *)p;
  return VMDPLUGIN_SUCCESS;
}


static void init_plugins() {
  hash_init(&pluginhash, 20);
  MOLFILE_INIT_ALL
  MOLFILE_REGISTER_ALL(NULL, register_cb)
}


static molfile_plugin_t *get_plugin(const char *filetype) {
  int id;

  /* one-time initialization of the plugin array */
  if (num_plugins == 0) 
    init_plugins();

  if ((id = hash_lookup(&pluginhash, filetype)) == HASH_FAIL) {
    fprintf(stderr, "No plugin found for filetype '%s'\n", filetype);
    return NULL;
  }

  return plugins[id];
}


/*
 * File-scope helper routines borrowed from the psf extraction code
 */

/* 
 * Return the segment corresponding to the given segname.  If the segname
 * doesn't exist, add it.  Return NULL on error.
 */
static topo_mol_segment_t *get_segment(topo_mol *mol, const char *segname) {
  int id;
  topo_mol_segment_t *seg = NULL;

  if ( (id = hasharray_index(mol->segment_hash, segname)) != HASHARRAY_FAIL) {
    /* Then the segment exists.  Look it up and return it. */
    seg = mol->segment_array[id];
  } else {
    /* Must create new segment */
    id = hasharray_insert(mol->segment_hash, segname);
    if (id != HASHARRAY_FAIL) {
      seg = mol->segment_array[id] =
            (topo_mol_segment_t *) malloc(sizeof(topo_mol_segment_t));
      strcpy(seg->segid, segname);
      seg->residue_hash = hasharray_create(
        (void**) &(seg->residue_array), sizeof(topo_mol_residue_t));
      strcpy(seg->pfirst,"");
      strcpy(seg->plast,"");
      seg->auto_angles = 0;
      seg->auto_dihedrals = 0;
    }
  }
  return seg;
}


/*
 * Return a new residue with the given resid.  Add it to the given segment.
 * If the resid already exists, return NULL.  Return NULL if there's a problem.
 */
static topo_mol_residue_t *get_residue(topo_mol_segment_t *seg,
        const char *resid) {
 
  int id;
  topo_mol_residue_t *res;
 
  /* Check that the residue doesn't already exist */
  if ( hasharray_index(seg->residue_hash,resid) != HASHARRAY_FAIL ) {
    return NULL;
  }
  id = hasharray_insert(seg->residue_hash, resid);
  if (id == HASHARRAY_FAIL) {
    return NULL;
  }
  res = &(seg->residue_array[id]);
  strcpy(res->resid, resid);

  return res;
}


/*
 * read bonds into atomlist data structures
 */
static int plugin_read_bonds(molfile_plugin_t *plg, void *rv, 
                             topo_mol *mol, int natoms, 
                             topo_mol_atom_t **molatomlist,
                             void *v, void (*print_msg)(void *, const char *)) {
  int i, nbonds, *from, *to, *bondtype, nbondtypes;
  float *bondorder;
  char **bondtypename;
  char msg[128];

  /* must explicitly set these since they may are otherwise only  */
  /* initialized by the read_bonds() call in the new ABI          */
  nbondtypes = 0;
  bondtype = NULL;
  bondtypename = NULL;
 
#if vmdplugin_ABIVERSION >= 15
  if (plg->read_bonds(rv, &nbonds, &from, &to, &bondorder, &bondtype, &nbondtypes, &bondtypename)) {
#else
  if (plg->read_bonds(rv, &nbonds, &from, &to, &bondorder)) {
#endif
    print_msg(v, "ERROR: failed reading bond information.");
    return -1;
  } else {
    sprintf(msg, "bonds: %d", nbonds);
    print_msg(v, msg);

    for (i=0; i < nbonds; i++) {
      topo_mol_atom_t *atom1, *atom2;
      topo_mol_bond_t *tuple;
      int ind1, ind2;

      ind1 = from[i]-1;
      ind2 = to[i]-1;
      if (ind1 < 0 || ind2 < 0 || ind1 >= natoms || ind2 >= natoms) {
        return -1; /* Bad indices, abort now */
      }
   
      atom1 = molatomlist[ind1];
      atom2 = molatomlist[ind2];
  
      tuple = memarena_alloc(mol->arena,sizeof(topo_mol_bond_t));
      tuple->next[0] = atom1->bonds;
      tuple->atom[0] = atom1;
      tuple->next[1] = atom2->bonds;
      tuple->atom[1] = atom2;
      tuple->del = 0;

      atom1->bonds = tuple;
      atom2->bonds = tuple;
    } 
  }

  return 0;
}

static int plugin_read_angles(molfile_plugin_t *plg, void *rv, 
                              topo_mol *mol, int natoms, 
                              topo_mol_atom_t **molatomlist,
                              void *v, void (*print_msg)(void *, const char *)){
  int numangles, *angles, *angletypes, numangletypes;
  int numdihedrals, *dihedrals, *dihedraltypes, numdihedraltypes;
  int numimpropers, *impropers, *impropertypes, numimpropertypes; 
  int numcterms, *cterms, ctermcols, ctermrows;
  char **angletypenames, **dihedraltypenames, **impropertypenames;
  char msg[128];
  topo_mol_atom_t *atom1, *atom2, *atom3, *atom4;
  int i, j;

  if (plg->read_angles(rv, &numangles, &angles, &angletypes, 
                          &numangletypes, &angletypenames, &numdihedrals,
                          &dihedrals,  &dihedraltypes, &numdihedraltypes, 
                          &dihedraltypenames, &numimpropers, &impropers,
                          &impropertypes, &numimpropertypes, 
                          &impropertypenames, &numcterms, &cterms, 
                          &ctermcols, &ctermrows)) {
    print_msg(v, "ERROR: failed reading angle information.");
    return -1;
  } else {
    sprintf(msg, "angles: %d dihedrals: %d impropers: %d cross-terms: %d",
            numangles, numdihedrals, numimpropers, numcterms);
    print_msg(v, msg);

    for (i=0; i < numangles; i++) {
      topo_mol_angle_t *tuple;

      atom1 = molatomlist[angles[3*i  ]-1];
      atom2 = molatomlist[angles[3*i+1]-1];
      atom3 = molatomlist[angles[3*i+2]-1];

      tuple = memarena_alloc(mol->arena,sizeof(topo_mol_angle_t));
      tuple->next[0] = atom1->angles;
      tuple->atom[0] = atom1;
      tuple->next[1] = atom2->angles;
      tuple->atom[1] = atom2;
      tuple->next[2] = atom3->angles;
      tuple->atom[2] = atom3;
      tuple->del = 0;

      atom1->angles = tuple;
      atom2->angles = tuple;
      atom3->angles = tuple;
    }

    for (i=0; i < numdihedrals; i++) {
      topo_mol_dihedral_t *tuple;

      atom1 = molatomlist[dihedrals[4*i  ]-1];
      atom2 = molatomlist[dihedrals[4*i+1]-1];
      atom3 = molatomlist[dihedrals[4*i+2]-1];
      atom4 = molatomlist[dihedrals[4*i+3]-1];

      tuple = memarena_alloc(mol->arena,sizeof(topo_mol_dihedral_t));
      tuple->next[0] = atom1->dihedrals;
      tuple->atom[0] = atom1;
      tuple->next[1] = atom2->dihedrals;
      tuple->atom[1] = atom2;
      tuple->next[2] = atom3->dihedrals;
      tuple->atom[2] = atom3;
      tuple->next[3] = atom4->dihedrals;
      tuple->atom[3] = atom4;
      tuple->del = 0;

      atom1->dihedrals = tuple;
      atom2->dihedrals = tuple;
      atom3->dihedrals = tuple;
      atom4->dihedrals = tuple;
    }

    for (i=0; i < numimpropers; i++) {
      topo_mol_improper_t *tuple;
 
      atom1 = molatomlist[impropers[4*i  ]-1];
      atom2 = molatomlist[impropers[4*i+1]-1];
      atom3 = molatomlist[impropers[4*i+2]-1];
      atom4 = molatomlist[impropers[4*i+3]-1];

      tuple = memarena_alloc(mol->arena,sizeof(topo_mol_improper_t));
      tuple->next[0] = atom1->impropers;
      tuple->atom[0] = atom1;
      tuple->next[1] = atom2->impropers;
      tuple->atom[1] = atom2;
      tuple->next[2] = atom3->impropers;
      tuple->atom[2] = atom3;
      tuple->next[3] = atom4->impropers;
      tuple->atom[3] = atom4;
      tuple->del = 0;

      atom1->impropers = tuple;
      atom2->impropers = tuple;
      atom3->impropers = tuple;
      atom4->impropers = tuple;
    }

    for (i=0; i < numcterms; i++) {
      topo_mol_atom_t *atoml[8];
      topo_mol_cmap_t *tuple;

      tuple = memarena_alloc(mol->arena,sizeof(topo_mol_cmap_t));
      for (j = 0; j < 8; ++j) {
        atoml[j] = molatomlist[cterms[8*i+j]-1];
        tuple->next[j] = atoml[j]->cmaps;
        tuple->atom[j] = atoml[j];
      }
      tuple->del = 0;
      for (j = 0; j < 8; ++j) {
        atoml[j]->cmaps = tuple;
      }
    }
  }

  return 0;
}


/*
 * Externally callable routines for reading/writing via plugin APIs
 */

/*
 * Read file into psfgen data structures using specified plugin
 */
int topo_mol_read_plugin(topo_mol *mol, const char *pluginname,
                         const char *filename, 
                         const char *segid, stringhash *h,
                         int coordinatesonly, int residuesonly,
                         void *v, void (*print_msg)(void *, const char *)) {
  char msg[2048];
  molfile_plugin_t *plg=NULL; /* plugin handle */
  void *rv=NULL;              /* opaque plugin read handle */
  int natoms=0;
  int optflags = MOLFILE_BADOPTIONS; /* plugin must reset this correctly */
  molfile_atom_t *atomarray=NULL;
  molfile_timestep_t ts;
  float *atomcoords=NULL;
  int i, rc;

  if (!mol)
    return -1;

  print_msg(v, "WARNING: Plugin-based I/O is still in development and may still have bugs");

  if (coordinatesonly && !h) {
    print_msg(v, "INTERNAL ERROR: bad hash table pointer for alias lookups");
    return -1;
  }

  /* find plugin that matches the requested type */
  if ((plg = get_plugin(pluginname)) == NULL) {
    print_msg(v, "ERROR: Failed to match requested plugin type");
    return -1; 
  }

  /* check for one or more usable input scenarios */
  if (plg->open_file_read == NULL &&
      plg->read_structure == NULL && 
      plg->read_timestep == NULL) {
    print_msg(v, "ERROR: selected plugin type cannot load structure information");
    return -1; 
  }

  if ((rv = plg->open_file_read(filename, plg->name, &natoms)) == NULL)
    return -1;

  atomarray = (molfile_atom_t *) malloc(natoms*sizeof(molfile_atom_t));
  memset(atomarray, 0, natoms*sizeof(molfile_atom_t));

  rc = plg->read_structure(rv, &optflags, atomarray);
  if (rc != MOLFILE_SUCCESS && rc != MOLFILE_NOSTRUCTUREDATA) {
    print_msg(v, "ERROR: plugin failed reading structure data");
    free(atomarray);
    return -1;
  }

  if (optflags == MOLFILE_BADOPTIONS) {
    free(atomarray);
    print_msg(v, "ERROR: plugin didn't initialize optional data flags");
    return -1;
  }

  if (!coordinatesonly && !residuesonly) {
    print_msg(v, "Data fields found by plugin:");
  
    if (optflags & MOLFILE_ATOMICNUMBER)
      print_msg(v, "  Atomic number");

    if (optflags & MOLFILE_OCCUPANCY)
      print_msg(v, "  Occupancy");
 
    if (optflags & MOLFILE_BFACTOR)
      print_msg(v, "  B-factor");
  }


  /*
   * load atom coordinates
   */
  memset(&ts, 0, sizeof(molfile_timestep_t));

  /* set defaults for unit cell information */
  ts.A = ts.B = ts.C = 0.0f;
  ts.alpha = ts.beta = ts.gamma = 90.0f; 

  atomcoords = (float *) malloc(3*natoms*sizeof(float));
  memset(atomcoords, 0, 3*natoms*sizeof(float));
  ts.coords = atomcoords;
  
  if (plg->read_next_timestep(rv, natoms, &ts)) {
    print_msg(v, "ERROR: failed reading atom coordinates");
    free(atomcoords);
    free(atomarray);
    plg->close_file_read(rv);
    return -1;
  }


  /*
   * update psfgen data structures with the structure information that
   * has already been read in.
   */
  if (!coordinatesonly && !residuesonly) {
    int i;
    topo_mol_atom_t **molatomlist;

    molatomlist = (topo_mol_atom_t **)malloc(natoms * sizeof(topo_mol_atom_t*));

    i=0;
    while (i < natoms) {
      char residbuf[16];
      topo_mol_segment_t *seg;
      topo_mol_residue_t *res;
      topo_mol_atom_t *atomtmp;
      int firstatom, j, residn;
      const char *resid, *segname;

      sprintf(residbuf, "%d", atomarray[i].resid);
      resid = residbuf;
      residn = atomarray[i].resid;

      segname = atomarray[i].segid;
      seg = get_segment(mol, segname);
      if (!seg) {
        print_msg(v,"ERROR: unable to get segment!");
        break;
      }
      res = get_residue(seg, resid);
      if (!res) {
        char *buf;
        int len = strlen(resid) + strlen(segname);
        buf = (char *)malloc((50 + len)*sizeof(char));
        sprintf(buf, "Unable to add (duplicate?) residue %s:%s", segname, resid);
        print_msg(v,buf);
        free(buf);
        break;
      }
      strcpy(res->name, atomarray[i].resname);
#if 1
      strcpy(res->chain, atomarray[i].chain);
#else
      strcpy(res->chain, "");
#endif
      res->atoms = 0;
      firstatom = i;
#if 1
      while (i<natoms && (residn == atomarray[i].resid) &&
             !strcmp(segname, atomarray[i].segid)) {
#else
      while (i<natoms && !strcmp(resid, atomarray[i].resid) &&
             !strcmp(segname, atomlist[i].segname)) {
#endif
        /* Add atoms to residue */
        atomtmp = memarena_alloc(mol->arena, sizeof(topo_mol_atom_t));
        atomtmp->bonds = 0;
        atomtmp->angles = 0;
        atomtmp->dihedrals = 0;
        atomtmp->impropers = 0;
        atomtmp->cmaps = 0;
        atomtmp->conformations = 0;
        strcpy(atomtmp->name, atomarray[i].name);
        strcpy(atomtmp->type, atomarray[i].type);
        strcpy(atomtmp->element, "");
        atomtmp->mass = atomarray[i].mass;
        atomtmp->charge = atomarray[i].charge;
        if (atomcoords) {
          atomtmp->x = atomcoords[i*3    ];
          atomtmp->y = atomcoords[i*3 + 1];
          atomtmp->z = atomcoords[i*3 + 2];
          atomtmp->xyz_state = TOPO_MOL_XYZ_SET;
        } else {
          atomtmp->x = 0;
          atomtmp->y = 0;
          atomtmp->z = 0;
          atomtmp->xyz_state = TOPO_MOL_XYZ_VOID;
        }
        atomtmp->partition = 0;
        atomtmp->copy = 0;
        atomtmp->atomid = 0;

        /* Save pointer to atom in my table so I can put in the bond
           information without having find the atom.
        */
        molatomlist[i] = atomtmp;
        i++;
      }

      for (j=i-1; j >= firstatom; j--) {
        /* Add new atoms to head of linked list in reverse order, so that
           the linked list is in the order they appear in the psf file.
        */
        atomtmp = molatomlist[j];
        atomtmp->next = res->atoms;
        res->atoms = atomtmp;
      }
    } 

    /* Check to see if we broke out of the loop prematurely */
    if (i != natoms) {
      print_msg(v, "ERROR: failed reading structure");
      free(molatomlist);
      return -1;
    }

#if 0
    /* 
     * XXX not implemented yet, this is neede by autopsf and friends 
     */

    /* Get the segment patch first,last and auto angles,dihedrals info */
    /* We have to rewind the file and read the info now since it has to be added to */
    /* the existing segments which have just been read. */
    extract_segment_extra_data(file, mol);
#endif


    /*
     * generate bonds/angles/dihedrals/impropers/cross-terms
     */

    /*
     * read bonds
     */
    if (!coordinatesonly  && !residuesonly && plg->read_bonds != NULL)
      plugin_read_bonds(plg, rv, mol, natoms, molatomlist, v, print_msg);


    /*
     * Read angles/dihedrals/impropers/cross-terms
     */
    if (!coordinatesonly && !residuesonly && plg->read_angles != NULL)
      plugin_read_angles(plg, rv, mol, natoms, molatomlist, v, print_msg);



    if (atomcoords)
      free(atomcoords);
    free(molatomlist);
  } 


  /* 
   * Load atom coordinate data into internal data structures 
   */
  if (coordinatesonly) {
    for (i=0; i<natoms; i++) {
      topo_mol_ident_t target;
      char residbuf[16];
      char stmp[128];
      unsigned int utmp;
      int found=0;

      memset(&target, 0, sizeof(target));

      strtoupper(atomarray[i].resname); 
      strtoupper(atomarray[i].name);
      strstripspaces(atomarray[i].name);
      strtoupper(atomarray[i].chain);

      if (!segid)
        target.segid = atomarray[i].segid;
      else
        target.segid = segid;

      sprintf(residbuf, "%d", atomarray[i].resid);
      target.resid = residbuf;

#if 0
      printf("XXX copying atom[%d] name='%s'  segid='%s'\n", i, atomarray[i].name, target.segid);
#endif

      target.aname = extract_alias_atom_check(h, atomarray[i].resname, atomarray[i].name);
      found = !topo_mol_set_xyz(mol, &target, atomcoords[i*3], atomcoords[i*3+1], atomcoords[i*3+2]);

      /* Try reversing order so 1HE2 in pdb matches HE21 in topology */
      if ( !found && sscanf(atomarray[i].name, "%u%s", &utmp, stmp) == 2 ) {
        char altname[8];
  
        snprintf(altname, 8, "%s%u", stmp, utmp);
        target.aname = altname;
        if ( !topo_mol_set_xyz(mol,&target, atomcoords[i*3], atomcoords[i*3+1], atomcoords[i*3+2]) ) {
          found = 1;
        }
      }
      if ( !found ) {
        sprintf(msg,"Warning: failed to set coordinate for atom %s\t %s:%d\t  %s",
                atomarray[i].name,
                atomarray[i].resname,
                atomarray[i].resid,
                segid ?  segid : atomarray[i].segid);
        print_msg(v,msg);
      } 
#if 0
      else {
        /* only try element and chain if coordinates succeeds */
        if ( strlen(element) && topo_mol_set_element(mol,&target,element,0) ) {
          sprintf(msg,"Warning: failed to set element for atom %s\t %s:%s\t  %s",name,resname,resid,segid ? segid : segname);
          print_msg(v,msg);
        }
        if ( strlen(chain) && topo_mol_set_chain(mol,&target,chain,0) ) {
          sprintf(msg,"Warning: failed to set chain for atom %s\t %s:%s\t  %s",name,resname,resid,segid ? segid : segname);
          print_msg(v,msg);
        }
      }
#endif
    }
  }

  if (atomarray)
    free(atomarray);
  plg->close_file_read(rv);

  return 0;
}



/*
 * Write psfgen structure to output file using selected plugin
 */
int topo_mol_write_plugin(topo_mol *mol, const char *pluginname,
                          const char *filename, 
                          void *v, void (*print_msg)(void *, const char *)) {
  char buf[256];
  int iseg,nseg,ires,nres,atomid;
  int has_guessed_atoms = 0;
  double x,y,z,o,b;
  topo_mol_segment_t *seg=NULL;
  topo_mol_residue_t *res=NULL;
  topo_mol_atom_t *atom=NULL;
  topo_mol_bond_t *bond=NULL;
  int nbonds;
  topo_mol_angle_t *angl=NULL;
  int nangls;
  topo_mol_dihedral_t *dihe=NULL;
  int ndihes;
  topo_mol_improper_t *impr=NULL;
  int nimprs;
  topo_mol_cmap_t *cmap=NULL;
  int ncmaps;

  molfile_plugin_t *plg; /* plugin handle */
  void *wv;              /* opaque plugin write handle */
  int natoms, optflags;
  molfile_atom_t *atomarray=NULL;
  molfile_timestep_t ts;
  float *atomcoords=NULL;

  if (!mol) 
    return -1;

  print_msg(v, "WARNING: Plugin-based I/O is still in development and may still have bugs");

  /* find plugin that matches the requested type */
  if ((plg = get_plugin(pluginname)) == NULL) {
    print_msg(v, "ERROR: Failed to match requested plugin type");
    return -1; 
  }

  /* check for one or more usable output scenarios */
  if (plg->write_structure == NULL && 
      plg->write_timestep == NULL) {
    print_msg(v, "ERROR: selected plugin type cannot store structure information");
    return -1; 
  }

  /* 
   * count atoms/bonds/dihedrals/impropers/cterms prior to
   * initializing the selected plugin for output 
   */
  natoms = 0;
  atomid = 0;
  nbonds = 0;
  nangls = 0;
  ndihes = 0;
  nimprs = 0;
  ncmaps = 0;
  nseg = hasharray_count(mol->segment_hash);
  for ( iseg=0; iseg<nseg; ++iseg ) {
    seg = mol->segment_array[iseg];
    if (! seg) continue;
    nres = hasharray_count(seg->residue_hash);
    for ( ires=0; ires<nres; ++ires ) {
      res = &(seg->residue_array[ires]);
      for ( atom = res->atoms; atom; atom = atom->next ) {
        atom->atomid = ++atomid;
        for ( bond = atom->bonds; bond;
                bond = topo_mol_bond_next(bond,atom) ) {
          if ( bond->atom[0] == atom && ! bond->del ) {
            ++nbonds;
          }  
        } 
        for ( angl = atom->angles; angl;
                angl = topo_mol_angle_next(angl,atom) ) {
          if ( angl->atom[0] == atom && ! angl->del ) {
            ++nangls;
          }  
        } 
        for ( dihe = atom->dihedrals; dihe;
                dihe = topo_mol_dihedral_next(dihe,atom) ) {
          if ( dihe->atom[0] == atom && ! dihe->del ) {
            ++ndihes;
          }  
        } 
        for ( impr = atom->impropers; impr;
                impr = topo_mol_improper_next(impr,atom) ) {
          if ( impr->atom[0] == atom && ! impr->del ) {
            ++nimprs;
          }
        }
        for ( cmap = atom->cmaps; cmap;
                cmap = topo_mol_cmap_next(cmap,atom) ) {
          if ( cmap->atom[0] == atom && ! cmap->del ) {
            ++ncmaps;
          }
        }
      }
    }
  }
  natoms=atomid;
  sprintf(buf,"total of %d atoms",atomid);
  print_msg(v,buf);
  sprintf(buf,"total of %d bonds",nbonds);
  print_msg(v,buf);
  sprintf(buf,"total of %d angles",nangls);
  print_msg(v,buf);
  sprintf(buf,"total of %d dihedrals",ndihes);
  print_msg(v,buf);
  sprintf(buf,"total of %d impropers",nimprs);
  print_msg(v,buf);

  /* allocate atom arrays */
  atomarray = (molfile_atom_t *) malloc(natoms*sizeof(molfile_atom_t));
  atomcoords = (float *) malloc(natoms * 3 * sizeof(float));
  if (atomarray == NULL) {
    print_msg(v, "ERROR: failed to allocate plugin atom attribute array");
    return -1;
  }
  if (atomcoords == NULL) {
    print_msg(v, "ERROR: failed to allocate plugin atom coordinate array");
    return -1;
  }

  memset(atomarray, 0, natoms*sizeof(molfile_atom_t));
  memset(atomcoords, 0, natoms*3*sizeof(float));

  /* open plugin for output */
  if ((wv = plg->open_file_write(filename, pluginname, natoms)) == NULL) {
    print_msg(v, "ERROR: plugin failed to open file for output");
    free(atomarray);
    free(atomcoords);
    return -1; 
  }

  /*
   * Emit equivalent of PSF "REMARKS" lines to track patches etc for
   * use by the higher level structure building plugins
   */
#if 0
  write_pdb_remark(file,"original generated coordinate pdb file");

  if (mol->npatch) {
    ntitle_count++;
    fprintf(file," REMARKS %i patches were applied to the molecule.\n", mol->npatch);
  }

  ntopo = hasharray_count(mol->defs->topo_hash);
  for ( itopo=0; itopo<ntopo; ++itopo ) {
    topo = &(mol->defs->topo_array[itopo]);
    ntitle_count++;
    fprintf(file," REMARKS topology %s \n", topo->filename);
  }

  nseg = hasharray_count(mol->segment_hash);
  for ( iseg=0; iseg<nseg; ++iseg ) {
    char angles[20], diheds[20];
    seg = mol->segment_array[iseg];
    if (! seg) continue;
    strcpy(angles,"none");
    strcpy(diheds,"");
    if (seg->auto_angles)    strcpy(angles,"angles");
    if (seg->auto_dihedrals) strcpy(diheds,"dihedrals");
    ntitle_count++;
    fprintf(file," REMARKS segment %s { first %s; last %s; auto %s %s }\n", seg->segid, seg->pfirst, seg->plast, angles, diheds);
  }

  for ( patch = mol->patches; patch; patch = patch->next ) {
    strcpy(defpatch,"");
    if (patch->deflt) strcpy(defpatch,"default");
    npres = patch->npres;
    ipres = 0;
    for ( patchres = patch->patchresids; patchres; patchres = patchres->next ) {
      /* Test the existence of segid:resid for the patch */
      if (!topo_mol_validate_patchres(mol,patch->pname,patchres->segid, patchres->resid)) {  
        break;
      };

      if (ipres==0) {
        ntitle_count++;
        fprintf(file," REMARKS %spatch %s ", defpatch, patch->pname);
      }
      if (ipres>0 && !ipres%6) {
        ntitle_count++;
        fprintf(file,"\n REMARKS patch ---- ");
      }
      fprintf(file,"%s:%s  ", patchres->segid, patchres->resid);
      if (ipres==npres-1) fprintf(file,"\n");
      ipres++;
     }
  }
  fprintf(file,"\n");
#endif


  /*
   * Emit atom coordinates and basic structure data
   */
  atomid = 0;
  nseg = hasharray_count(mol->segment_hash);
  for ( iseg=0; iseg<nseg; ++iseg ) {
    seg = mol->segment_array[iseg];
    if (! seg) continue;
    nres = hasharray_count(seg->residue_hash);
    for ( ires=0; ires<nres; ++ires ) {
      res = &(seg->residue_array[ires]);
      for ( atom = res->atoms; atom; atom = atom->next ) {
        molfile_atom_t *atm = &atomarray[atomid];

        /* Paranoid: make sure x,y,z,o are set. */
        x = y = z = 0.0; o = -1.0;
        switch ( atom->xyz_state ) {
        case TOPO_MOL_XYZ_SET:
          x = atom->x;  y = atom->y;  z = atom->z;  o = 1.0;
          break;
        case TOPO_MOL_XYZ_GUESS:
        case TOPO_MOL_XYZ_BADGUESS:
          x = atom->x;  y = atom->y;  z = atom->z;  o = 0.0;
          has_guessed_atoms = 1;
          break;
        default:
          print_msg(v,"ERROR: Internal error, atom has invalid state.");
          print_msg(v,"ERROR: Treating as void.");
          /* Yes, fall through */
        case TOPO_MOL_XYZ_VOID:
          x = y = z = 0.0;  o = -1.0;
          break;
        }
        b = atom->partition;

        /* save atom attributes */ 
        strcpy(atm->name, atom->name);
        strcpy(atm->type, atom->type);
        strcpy(atm->resname, res->name);
        atm->resid = atoi(res->resid);
        strcpy(atm->chain, res->chain);
        strcpy(atm->segid, seg->segid);
        strcpy(atm->insertion, "");
        strcpy(atm->altloc, "");
        atm->atomicnumber = -1; /* we should be able to do much better */
        atm->occupancy = o;
        atm->bfactor = b;
        atm->mass = atom->mass;
        atm->charge = atom->charge;
        atm->radius = 0.0;

        /* save coords */
        atomcoords[atomid*3    ] = x;
        atomcoords[atomid*3 + 1] = y;
        atomcoords[atomid*3 + 2] = z;

        atomid++;
      }
    }
  }

  if (has_guessed_atoms) {
    print_msg(v, 
        "Info: Atoms with guessed coordinates will have occupancy of 0.0.");
  }

  /* set flags indicating what data is populated/valid */
  optflags = MOLFILE_OCCUPANCY | MOLFILE_BFACTOR | 
             MOLFILE_MASS | MOLFILE_CHARGE;

  /* build bond list here */
  if (nbonds > 0 && plg->write_bonds != NULL) {
    int *from, *to;
    int bondcnt=0;
    from = (int *) malloc(nbonds * sizeof(int));
    to = (int *) malloc(nbonds * sizeof(int));

    for ( iseg=0; iseg<nseg; ++iseg ) {
      seg = mol->segment_array[iseg];
      if (! seg) continue;
      nres = hasharray_count(seg->residue_hash);
      for ( ires=0; ires<nres; ++ires ) {
        res = &(seg->residue_array[ires]);
        for ( atom = res->atoms; atom; atom = atom->next ) {
          for ( bond = atom->bonds; bond;
                  bond = topo_mol_bond_next(bond,atom) ) {
            if ( bond->atom[0] == atom && ! bond->del ) {
              from[bondcnt]=atom->atomid;
              to[bondcnt]=bond->atom[1]->atomid;
              bondcnt++;               
            }
          }
        }
      }
    }

#if vmdplugin_ABIVERSION >= 15
    if (plg->write_bonds(wv, nbonds, from, to, NULL, NULL, 0, NULL)) {
#else
    if (plg->write_bonds(wv, nbonds, from, to, NULL)) {
#endif
      print_msg(v, "ERROR: plugin failed to write bonds");
      free(from);
      free(to);
      free(atomarray);
      free(atomcoords);
      return -1;
    }

    free(from);
    free(to);
  }
 
  /* build angle/dihedral/improper/cterm lists here */
  if ((nangls > 0 || ndihes > 0 || nimprs > 0 || ncmaps > 0) &&
      plg->write_angles != NULL) {
    int anglcnt=0;
    int dihecnt=0;
    int imprcnt=0;
    int cmapcnt=0;

    int *angles = (int *) malloc(3 * nangls * sizeof(int));
    int *dihedrals = (int *) malloc(4 * ndihes * sizeof(int));
    int *impropers = (int *) malloc(4 * nimprs * sizeof(int));
    int *cmaps = (int *) malloc(8 * ncmaps * sizeof(int));

    /*
     * angles
     */
    for ( iseg=0; iseg<nseg; ++iseg ) {
      seg = mol->segment_array[iseg];
      if (! seg) continue;
      nres = hasharray_count(seg->residue_hash);
      for ( ires=0; ires<nres; ++ires ) {
        res = &(seg->residue_array[ires]);
        for ( atom = res->atoms; atom; atom = atom->next ) {
          for ( angl = atom->angles; angl;
                  angl = topo_mol_angle_next(angl,atom) ) {
            if ( angl->atom[0] == atom && ! angl->del ) {
              angles[anglcnt++] = atom->atomid;
              angles[anglcnt++] = angl->atom[1]->atomid;
              angles[anglcnt++] = angl->atom[2]->atomid;
            }
          }
        }
      }
    }

    /*
     * dihedrals
     */
    for ( iseg=0; iseg<nseg; ++iseg ) {
      seg = mol->segment_array[iseg];
      if (! seg) continue;
      nres = hasharray_count(seg->residue_hash);
      for ( ires=0; ires<nres; ++ires ) {
        res = &(seg->residue_array[ires]);
        for ( atom = res->atoms; atom; atom = atom->next ) {
          for ( dihe = atom->dihedrals; dihe;
                  dihe = topo_mol_dihedral_next(dihe,atom) ) {
            if ( dihe->atom[0] == atom && ! dihe->del ) {
              dihedrals[dihecnt++] = atom->atomid;
              dihedrals[dihecnt++] = dihe->atom[1]->atomid;
              dihedrals[dihecnt++] = dihe->atom[2]->atomid;
              dihedrals[dihecnt++] = dihe->atom[3]->atomid;
            }
          }
        }
      }
    }

    /*
     * impropers
     */
    for ( iseg=0; iseg<nseg; ++iseg ) {
      seg = mol->segment_array[iseg];
      if (! seg) continue;
      nres = hasharray_count(seg->residue_hash);
      for ( ires=0; ires<nres; ++ires ) {
        res = &(seg->residue_array[ires]);
        for ( atom = res->atoms; atom; atom = atom->next ) {
          for ( impr = atom->impropers; impr;
                  impr = topo_mol_improper_next(impr,atom) ) {
            if ( impr->atom[0] == atom && ! impr->del ) {
              impropers[imprcnt++] = atom->atomid;
              impropers[imprcnt++] = impr->atom[1]->atomid;
              impropers[imprcnt++] = impr->atom[2]->atomid;
              impropers[imprcnt++] = impr->atom[3]->atomid;
            }
          }
        }
      }
    }

    /*
     * cross-terms
     */
    for ( iseg=0; iseg<nseg; ++iseg ) {
      seg = mol->segment_array[iseg];
      if (! seg) continue;
      nres = hasharray_count(seg->residue_hash);
      for ( ires=0; ires<nres; ++ires ) {
        res = &(seg->residue_array[ires]);
        for ( atom = res->atoms; atom; atom = atom->next ) {
          for ( cmap = atom->cmaps; cmap;
                cmap = topo_mol_cmap_next(cmap,atom) ) {
            if ( cmap->atom[0] == atom && ! cmap->del ) {
              cmaps[cmapcnt++] = atom->atomid;
              cmaps[cmapcnt++] = cmap->atom[1]->atomid;
              cmaps[cmapcnt++] = cmap->atom[2]->atomid;
              cmaps[cmapcnt++] = cmap->atom[3]->atomid;
              cmaps[cmapcnt++] = cmap->atom[4]->atomid;
              cmaps[cmapcnt++] = cmap->atom[5]->atomid;
              cmaps[cmapcnt++] = cmap->atom[6]->atomid;
              cmaps[cmapcnt++] = cmap->atom[7]->atomid;
            }
          }
        }
      }
    }

    if (plg->write_angles(wv, 
                          nangls, angles, NULL, 0, NULL,
                          ndihes, dihedrals, NULL, 0, NULL,
                          nimprs, impropers, NULL, 0, NULL,
                          ncmaps, cmaps, 0, 0)) {
      print_msg(v, "ERROR: plugin failed to write angles/dihedrals/impropers/cross-terms");
      if (angles != NULL)
        free(angles);
      if (dihedrals != NULL)
        free(dihedrals);
      if (impropers != NULL)
        free(impropers);
      if (cmaps != NULL)
        free(cmaps);
      free(atomarray);
      free(atomcoords);
      return -1;
    }

    if (angles != NULL)
      free(angles);
    if (dihedrals != NULL)
      free(dihedrals);
    if (impropers != NULL)
      free(impropers);
    if (cmaps != NULL)
      free(cmaps);
  }

  /* emit the completed structure */
  if (plg->write_structure(wv, optflags, atomarray)) {
    free(atomarray);
    free(atomcoords);
    print_msg(v, "ERROR: plugin failed to write structure data");
    return -1;
  }
  free(atomarray);
  
  /* emit atom coordinates */
  if (plg->write_timestep != NULL) {
    ts.A = 0;
    ts.B = 0;
    ts.C = 0;
    ts.alpha = 0;
    ts.beta = 0;
    ts.gamma = 0;
    ts.coords = atomcoords;
    ts.velocities = NULL;
    if (plg->write_timestep(wv, &ts)) {
      free(atomcoords);
      print_msg(v, "ERROR: plugin failed to write atom coordinates");
      return -1;
    }
  }
  free(atomcoords);

  /* close the output file and release memory allocated by the plugin */
  plg->close_file_write(wv);

  print_msg(v, "WARNING: Plugin-based I/O is still in development and may still have bugs");

  return 0;
}

#endif

