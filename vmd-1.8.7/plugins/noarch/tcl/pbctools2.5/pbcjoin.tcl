############################################################
#
#   This file contains procedures to join compounds of atoms that are
# wrapped around unit cell boundaries.
#
# $Id: pbcjoin.tcl,v 1.11 2009/05/14 21:47:35 johns Exp $
#

package provide pbctools 2.5

namespace eval ::PBCTools:: {
    namespace export pbc*
    ############################################################
    #
    # pbcjoin $compound [OPTIONS...]
    #
    #   Joins compounds of type $compound of atoms that have been
    # split due to wrapping around the unit cell boundaries, so that
    # they are not split anymore. $compound must be one of the values
    # "residue", "chain", "segment" or "fragment".
    # 
    # OPTIONS:
    #   -molid $molid|top
    #   -first $first|first|now 
    #   -last $last|last|now
    #   -all|allframes
    #   -now
    #   -sel $sel
    #   -noborder|-border $depth
    #   -noref|-ref $sel
    #   -[no]verbose
    #
    # AUTHOR: Olaf
    #
    proc pbcjoin { compoundtype args } {
	# Set the defaults
	set molid "top"
	set first "now"
	set last "now"
	set seltext "all"
	set border 0
	set ref "all"
	set verbose 0

	# Normalize compoundtype
	switch -- $compoundtype {
	    "seg" -
	    "segid" { 
		set compoundtype "segid" 
		set compoundseltext "segid %s"
	    }
	    "res" -
	    "resid" -
	    "residue" { 
		set compoundtype "residue" 
		set compoundseltext "residue %s"
	    }
	    "chain" { 
		set compoundtype "chain" 
		set compoundseltext "chain %s"
	    }
	    "bonded" -
	    "fragment" { 
		set compoundtype "fragment" 
		set compoundseltext "fragment %s"
	    }
	    default { error "ERROR: pbcjoin: unknown compound type $compoundtype" }
	}

	# Parse options
	for { set argnum 0 } { $argnum < [llength $args] } { incr argnum } {
	    set arg [ lindex $args $argnum ]
	    set val [ lindex $args [expr {$argnum + 1}]]
	    switch -- $arg {
		"-molid" { set molid $val; incr argnum }
		"-first" { set first $val; incr argnum }
		"-last" { set last $val; incr argnum }
		"-allframes" -
		"-all" { set last "last"; set first "first" }
		"-now" { set last "now"; set first "now" }
		"-sel" { set seltext $val; incr argnum }
		"-border" { set border $val; incr argnum }
		"-noborder" { set border 0 }
		"-ref" { set ref $val; incr argnum }
		"-noref" { set ref "all" }
		"-verbose" { set verbose 1 }
		"-noverbose" { set verbose 0 }
		default { error "pbcjoin: unknown option: $arg" }
	    }
	}
	    
	if { $molid eq "top" } then { set molid [ molinfo top ] }
	    
	# Save the current frame number
	set frame_before [ molinfo $molid get frame ]

	if { $first eq "now" }   then { set first $frame_before }
	if { $first eq "first" || $first eq "start" || $first eq "begin" } then { 
	    set first 0 
	}
	if { $last eq "now" }    then { set last $frame_before }
	if { $last eq "last" || $last eq "end" } then {
	    set last [expr {[molinfo $molid get numframes]-1}]
	}

	if { $border != 0 } then {
	    set borderseltext "x<$border or y<$border or z<$border"
	    if { $seltext ne "all" } then {
		set seltext "($seltext) and $borderseltext"
	    } else {
		set seltext $borderseltext
	    }
	}

	if { $verbose } then {
	    set numframes [expr $last - $first + 1]
	    set start_time [clock clicks -milliseconds]
	    set next_time [clock clicks -milliseconds]
	    set show_step 1000
	    vmdcon -info "Will join $numframes frames."
	}

	set totalcnt 0
	set framecnt 0
	for {set frame $first} { $frame <= $last } { incr frame } {
	    if { $verbose } then {
		vmdcon -info "Joining compounds in frame $frame ($framecnt/$numframes)."
	    }
	    molinfo $molid set frame $frame
	    
	    # get the current cell 
	    set cell [lindex [pbc get -molid $molid -namd] 0]
	    set A [lindex $cell 0]
	    set B [lindex $cell 1]
	    set C [lindex $cell 2]
	    
	    set cell [lindex [pbc get -molid $molid -vmd] 0]
	    pbc_check_cell $cell

	    # create the selection
	    set sel [atomselect $molid $seltext frame $frame]
	    # create a list of all compounds in sel: these compounds need to be tested
	    set compoundlist {}
	    switch -- $compoundtype {
		"segid" { set compoundlist [lsort -unique [$sel get segid]] }
		"residue" { set compoundlist [lsort -integer -unique [$sel get residue]] }
		"chain" { set compoundlist [lsort -unique [$sel get chain]] }
		"fragment" { set compoundlist [lsort -unique [$sel get fragment]] }
	    }
	    $sel delete

	    if { [llength $compoundlist] == 0 } then {
		vmdcon -warn "Did not find any compounds to join in frame $frame!"
		continue
	    }
	
	    if { $verbose } then {
		set numcompounds  [llength $compoundlist]
		vmdcon -info "Testing $numcompounds compounds."
	    }

	    # determine half the box size
	    # if a compound is larger than half the box size, it can
	    # be assumed that it needs to be joined
	    set a [expr 0.5 * [lindex $cell 0]]
	    set b [expr 0.5 * [lindex $cell 1]]
	    set c [expr 0.5 * [lindex $cell 2]]

	    set joincompounds {}
	    set xs {}
	    set ys {}
	    set zs {}
	    set rxs {}
	    set rys {}
	    set rzs {}

	    set compoundcnt 0
	    # loop over all compounds
	    foreach compoundid $compoundlist {
		# select the next compound
		set compound [atomselect $molid [format $compoundseltext $compoundid] frame $frame]

		# now test whether the compound needs to be joined
		set minmax [measure minmax $compound]

		set d [vecsub [lindex $minmax 1] [lindex $minmax 0]]
		set dx [lindex $d 0]
		set dy [lindex $d 1]
		set dz [lindex $d 2]
		if { $dx > $a || $dy > $b || $dz > $c } then {
		    set x [$compound get x]
		    set y [$compound get y]
		    set z [$compound get z]

		    if { $ref ne "all" } then {
			# get the coordinates of the reference atom in the compound
			set refsel [atomselect $molid [format "$compoundseltext and ($ref)" $compoundid] frame $frame]
			set r [lindex [$refsel get { x y z }] 0]
			$refsel delete
			set rx [lindex $r 0]
			set ry [lindex $r 1]
			set rz [lindex $r 2]
		    } else {
			# otherwise get the first atom in the compound
			set rx [lindex $x 0]
			set ry [lindex $y 0]
			set rz [lindex $z 0]
		    }

		    # append the coordinates of the compounds atoms
		    # and its reference atom to the result list
		    lappend joincompounds $compoundid
		    foreach xv $x yv $y zv $z {
			lappend xs $xv
			lappend ys $yv
			lappend zs $zv
			lappend rxs $rx
			lappend rys $ry
			lappend rzs $rz
		    }

		}
		$compound delete

		if {$verbose} then {
		    set time [clock clicks -milliseconds]
		    if { $time >= $next_time} then {
			set progress [expr $totalcnt / (1.0*$numcompounds*$numframes)]
			set elapsed [expr ($time-$start_time)/1000.0]

			set percentage [format "%4.1f" [expr 100.0*$progress]]
			set elapseds [format "%4.1f" $elapsed]
			set eta [format "%4.1f" [expr $elapsed/$progress]]

			vmdcon -info "$percentage% complete (frame $framecnt/$numframes, compound $compoundcnt/$numcompounds) $elapseds s / $eta s"
			set next_time [expr $time + $show_step]
		    }
		}
		incr compoundcnt
		incr totalcnt
	    } 
	    # END foreach compound $compoundlist

	    if { $verbose } then {
		vmdcon -info "Joining [llength $joincompounds] compounds."
	    }
	    if { [llength $joincompounds] > 0 } then {

		set joinsel [atomselect $molid [format $compoundseltext $joincompounds] frame $frame]

		# wrap the coordinates
		pbcwrap_coordinates $A $B $C xs ys zs $rxs $rys $rzs

		# set the new coordinates
		$joinsel set x $xs
		$joinsel set y $ys
		$joinsel set z $zs

		$joinsel delete
	    }

	    incr framecnt
	}

	if {$verbose} then {
	    set percentage 100
	    vmdcon -info "100.0% complete (frame $frame, compoundid $compoundid)"
	}

	# Rewind to original frame
	if { $verbose } then { vmdcon -info "Rewinding to frame $frame_before." }
	animate goto $frame_before
    }

    # > pbcwrap -compound $compound -compundref $ref
    # is equivalent to
    # > pbcwrap -sel $ref
    # > pbcjoin $compound -ref $ref

}
