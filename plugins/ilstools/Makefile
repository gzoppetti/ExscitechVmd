.SILENT:

COPYFILES = ilstools.tcl pkgIndex.tcl oxygen.xyz nitricoxide.xyz \
            carbonmonoxide.xyz carbondioxide.xyz ethene.xyz methane.xyz \
            nitrousoxide.xyz nitrogendioxide.xyz nitrogen.xyz \
            acetylene.xyz hydrogen.xyz

DIR = $(PLUGINDIR)/noarch/tcl/ilstools1.2

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib: 
	@echo "Copying ilstools 1.2 files to $(DIR)"
	mkdir -p $(DIR) 
	cp -rf $(COPYFILES) $(DIR) 
