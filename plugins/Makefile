##
## Plugin compilation and distribution rules
## Copyright 2002-2004, Board of Trustees of the University of Illinois
## Theoretical and Computational Biophysics Group
##

#.SILENT:

default: make-arch-help

world:
	@echo "Building all supported targets..."
	csh -f build.csh

# These plugins are kept in a separate multiseq CVS tree and must be
# periodically synced up with the main VMD plugin CVS when updated.
# They are listed separately here to make this distinction more transparent.
MSEQBUILDDIRS = blast clustalw cluster libbiokit \
                multiseq multiseqdialog phylotree \
                psipred seqdata seqedit stamp

# When building from the independent multiseq tree, we'll use an empty list
# since the dependency will be dealt with manually.
#MSEQBUILDDIRS = 

# List of plugins to build
#   Note: molfile plugins must be first since some of the 
#         other plugins directly depend on them
BUILDDIRS = \
  apbsrun \
  atomedit \
  autoimd \
  autoionize \
  autopsf \
  bignum \
  biocore \
  catdcd \
  cgtools \
  cionize \
  cliptool \
  clonerep \
  colorscalebar \
  contactmap \
  dataimport \
  demomaster \
  dipwatch \
  dowser \
  exectool \
  extendedpdb \
  gofrgui \
  hbonds \
  hesstrans \
  idatm \
  ilstools \
  imdmenu \
  inorganicbuilder \
  intersurf \
  irspecgui \
  membrane \
  mergestructs \
  molefacture \
  moltoptools \
  multimolanim \
  multiplot \
  multitext \
  mutator \
  namdenergy \
  namdgui \
  namdplot \
  namdserver \
  nanotube \
  navfly \
  navigate \
  optimization \
  palettetool \
  paratool \
  pbctools \
  pdbtool \
  pmepot \
  psfgen \
  qmtool \
  ramaplot \
  readcharmmpar \
  readcharmmtop \
  resptool \
  rnaview \
  rmsd \
  rmsdtt \
  ruler \
  runante \
  saltbr \
  signalproc \
  solvate \
  ssrestraints \
  stingtool \
	symmetrytool \
  timeline \
  topotools \
  trunctraj \
  utilities \
  vdna \
  viewmaster \
  volmapgui \
  vmdmovie \
  vmdtkcon \
  zoomseq \
  $(MSEQBUILDDIRS)

INCLUDEDIR = include
PLUGINHEADERS = vmdplugin.h molfile_plugin.h vmdconio.h

distrib: 
	@echo "Populating distribution directory with compiled plugins"
	-mkdir -p $(PLUGINDIR)/include
	for header in $(PLUGINHEADERS); do cp -p $(INCLUDEDIR)/$$header $(PLUGINDIR)/include || exit 1 ; done
	for dir in molfile_plugin $(BUILDDIRS); do cd $$dir && $(MAKE) distrib && cd .. || exit 1 ; done

include Make-arch

molfilelibs:
	cd molfile_plugin; $(MAKE) dynlibs && cd .. || exit 1
	cd molfile_plugin; $(MAKE) staticlibs && cd .. || exit 1

win32molfilelibs:
	cd molfile_plugin; $(MAKE) dynlibs && cd .. || exit 1
	cd molfile_plugin; $(MAKE) win32staticlibs && cd .. || exit 1

dynlibs: molfilelibs
	for dir in $(BUILDDIRS); do cd $$dir && $(MAKE) dynlibs && cd .. || exit 1 ; done

staticlibs: molfilelibs
	for dir in $(BUILDDIRS); do cd $$dir && $(MAKE) staticlibs && cd .. || exit 1 ; done

win32staticlibs: win32molfilelibs
	for dir in $(BUILDDIRS); do cd $$dir && $(MAKE) win32staticlibs && cd .. || exit 1 ; done

bins:
	for dir in molfile_plugin $(BUILDDIRS); do cd $$dir && $(MAKE) bins && cd .. || exit 1 ; done

win32bins:
	for dir in molfile_plugin $(BUILDDIRS); do cd $$dir && $(MAKE) win32bins && cd .. || exit 1 ; done

clean:
	find compile \( -name \*.o -o -name \*.a -o -name \*.so -o -name \*.exp -o -name \*.lib -o -name \*.h \) -print | xargs rm -f
	find compile \( -name lib_\* \) -print | xargs rm -rf
	rm -f log.*
	rm -f psfgen/so_locations
	rm -f molfile_plugin/so_locations

checkperms:
	@echo "Looking for bad file permissions..."
	find . ! -perm +004

