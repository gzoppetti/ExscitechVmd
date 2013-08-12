.SILENT:

SOLVFILES = wat.top wat.pdb wat.psf solvate.tcl pkgIndex.tcl
SOLVVERSION = 1.3
DIR = $(PLUGINDIR)/noarch/tcl/solvate$(SOLVVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying solvate $(SOLVVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(SOLVFILES) $(DIR) 

	
