#!/usr/bin/tclsh
# This file is part of TopoTools, a VMD package to simplify 
# manipulating bonds other topology related properties.
#
# Copyright (c) 2009 by Axel Kohlmeyer <akohlmey@cmm.chem.upenn.edu>
#

# return info about impropers
# we list and count only impropers that are entirely within the selection.
proc ::TopoTools::improperinfo {infotype sel {flag none}} {

    set numimpropers 0
    array set impropertypes {}
    set atidxlist [$sel list]
    set improperlist {}

    foreach improper [join [molinfo [$sel molid] get impropers]] {
        lassign $improper t a b c d

        if {([lsearch -sorted -integer $atidxlist $a] >= 0)          \
                && ([lsearch -sorted -integer $atidxlist $b] >= 0)   \
                && ([lsearch -sorted -integer $atidxlist $c] >= 0)   \
                && ([lsearch -sorted -integer $atidxlist $d] >= 0) } {
            set impropertypes($t) 1
            incr numimpropers
            lappend improperlist $improper
        }
    }
    switch $infotype {

        numimpropers      { return $numimpropers }
        numimpropertypes  { return [array size impropertypes] }
        impropertypenames { return [array names impropertypes] }
        getimproperlist   { return $improperlist }
        default        { return "bug! shoot the programmer?"}
    }
}

# delete all contained impropers of the selection.
proc ::TopoTools::clearimpropers {sel} {
    set mol [$sel molid]
    set atidxlist [$sel list]
    set improperlist {}

    foreach improper [join [molinfo $mol get impropers]] {
        lassign $improper t a b c d

        if {([lsearch -sorted -integer $atidxlist $a] < 0)          \
                || ([lsearch -sorted -integer $atidxlist $b] < 0)   \
                || ([lsearch -sorted -integer $atidxlist $c] < 0)   \
                || ([lsearch -sorted -integer $atidxlist $d] < 0) } {
            lappend improperlist $improper
        }
    }
    molinfo $mol set impropers [list $improperlist]
}

# reset impropers to data in improperlist
proc ::TopoTools::setimproperlist {sel improperlist} {

    set mol [$sel molid]
    set atidxlist [$sel list]
    set newimproperlist {}

    # set defaults
    set t unknown; set a -1; set b -1; set c -1; set d -1

    # preserve all impropers definitions that are not contained in $sel
    foreach improper [improperinfo getimproperlist $sel] {
        lassign $improper t a b c d

        if {([lsearch -sorted -integer $atidxlist $a] < 0)          \
                || ([lsearch -sorted -integer $atidxlist $b] < 0)   \
                || ([lsearch -sorted -integer $atidxlist $c] < 0)   \
                || ([lsearch -sorted -integer $atidxlist $d] < 0) } {
            lappend newimproperlist $improper
        }
    }

    # append new ones, but only those contained in $sel
    foreach improper $improperlist {
        lassign $improper t a b c d

        if {([lsearch -sorted -integer $atidxlist $a] >= 0)          \
                && ([lsearch -sorted -integer $atidxlist $b] >= 0)   \
                && ([lsearch -sorted -integer $atidxlist $c] >= 0)   \
                && ([lsearch -sorted -integer $atidxlist $d] >= 0) } {
            lappend newimproperlist $improper
        }
    }

    molinfo $mol set impropers [list $newimproperlist]
}

# reset impropers to data in improperlist
proc ::TopoTools::retypeimpropers {sel} {

    set mol [$sel molid]
    set improperlist [improperinfo getimproperlist $sel]
    set atomtypes [$sel get type]
    set atomindex [$sel list]
    set newimpropers {}
    
    foreach improper $improperlist {
        lassign $improper type i1 i2 i3 i4

        set idx [lsearch -sorted -integer $atomindex $i1]
        set a [lindex $atomtypes $idx]
        set idx [lsearch -sorted -integer $atomindex $i2]
        set b [lindex $atomtypes $idx]
        set idx [lsearch -sorted -integer $atomindex $i3]
        set c [lindex $atomtypes $idx]
        set idx [lsearch -sorted -integer $atomindex $i4]
        set d [lindex $atomtypes $idx]

        if { ([string compare $b $c] > 0) \
                 || ( [string equal $b $c] && [string compare $a $d] > 0 ) } {
            set t $a; set a $d; set d $t 
            set t $b; set b $c; set c $t 
        }
        set type [join [list $a $b $c $d] "-"]

        lappend newimpropers [list $type $i1 $i2 $i3 $i4]
    }
    setimproperlist $sel $newimpropers
}


# define a new improper or change an existing one.
proc ::TopoTools::addimproper {mol id1 id2 id3 id4 {type unknown}} {
    if {[catch {atomselect $mol "index $id1 $id2 $id3 $id4"} sel]} {
        vmdcon -error "topology addimproper: Invalid atom indices: $sel"
        return
    }

    # canonicalize indices
    if {$id2 > $id3} {
        set t $id2 ; set id2 $id3 ; set id3 $t 
        set t $id1 ; set id1 $id4 ; set id4 $t 
    }

    set impropers [join [molinfo $mol get impropers]]
    lappend impropers [list $type $id1 $id2 $id3 $id4]
    molinfo $mol set impropers [list $impropers]
}

# delete a improper.
proc ::TopoTools::delimproper {mol id1 id2 id3 id4 {type unknown}} {
    if {[catch {atomselect $mol "index $id1 $id2 $id3 $id4"} sel]} {
        vmdcon -error "topology delimproper: Invalid atom indices: $sel"
        return
    }

    # canonicalize indices
    if {$id2 > $id3} {
        set t $id2 ; set id2 $id3 ; set id3 $t 
        set t $id1 ; set id1 $id4 ; set id4 $t 
    }

    set newimpropers {}
    foreach improper [join [molinfo $mol get impropers]] {
        lassign $improper t a b c d
        if { ($a != $id1) || ($b != $id2) || ($c != $id3) || ($d != $id4) } {
            lappend newimpropers $improper
        }
    }
    molinfo $mol set impropers [list $newimpropers]
}
