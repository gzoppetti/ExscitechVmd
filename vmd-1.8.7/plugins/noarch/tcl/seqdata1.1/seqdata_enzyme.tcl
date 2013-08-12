############################################################################
#cr
#cr            (C) Copyright 1995-2004 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################

############################################################################
# RCS INFORMATION:
#
#       $RCSfile: seqdata_enzyme.tcl,v $
#       $Author: kvandivo $        $Locker:  $             $State: Exp $
#       $Revision: 1.2 $       $Date: 2009/03/31 17:00:47 $
#
############################################################################

# This file provides functions for obtaining information about Swiss Prot sequences.

package provide seqdata 1.1

# Declare global variables for this package.
namespace eval ::SeqData::Enzyme {

    # The map of Swisprot organism identification codes
    variable map
    array unset map 

    proc getDescription {ecNumber} {
        
        # Import global variables.
        variable map
    
        # See if we have a common name.
        if {[info exists map($ecNumber,description)]} {
            return $map($ecNumber,description)
        }
        
        return ""
    }
    
    proc addEnzyme {ecNumber description} {
        
        # Import global variables.
        variable map
        
        set map($ecNumber,description) $description
    }
}

