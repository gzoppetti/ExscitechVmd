.SILENT:

VMFILES = readcharmmtop.tcl pkgIndex.tcl top_all27_prot_lipid_na.inp top_all27_hybrid.inp top_amber2charmm.inp
VMVERSION = 1.0
DIR = $(PLUGINDIR)/noarch/tcl/readcharmmtop$(VMVERSION)

bins:
win32bins:
dynlibs:
staticlibs:
win32staticlibs:

distrib:
	@echo "Copying readcharmmtop $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 

	
