
VMFILES = phylotree.tcl phylotree_analysis.tcl phylotree_data.tcl phylotree_export.tcl phylotree_je.tcl phylotree_matrixviewer.tcl phylotree_newick.tcl phylotree_nexus.tcl\
    phylotree_nodeprops.tcl phylotree_widget.tcl pkgIndex.tcl
    
VMVERSION = 1.1
DIR = $(PLUGINDIR)/noarch/tcl/phylotree$(VMVERSION)

#
# Rules
#
bins:
win32bins:
dynlibs:
staticlibs: 
win32staticlibs: 

distrib:
	echo "Copying phylotree $(VMVERSION) files to $(DIR)"
	mkdir -p $(DIR) 
	cp $(VMFILES) $(DIR) 

	
