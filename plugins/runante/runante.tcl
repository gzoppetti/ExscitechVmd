package require exectool 1.2

package provide runante 0.1

namespace eval ::ANTECHAMBER:: {

  variable acpath "";# path to antechamber executable
  variable electypes [list resp cm1 esp gas wc bcc cm2 mul rc]
  variable molefmode 0 ;# are we running inside of molefacture?

}

proc ::ANTECHAMBER::run_ac_typing {selection {chargetype bcc} {totcharge 0.0} {atomtypes gaff} {spinmult 1} {resname MOL}} {
# Run a full antechamber typing run on the selection
# To do this, write a mol2 with the atoms and initial bond orders,
# and then call the antechamber command line executable

# The fully typed molecule is loaded as a new molecule in vmd, and
# the molid of this molecule is returned

# If atomtypes begins with the string CUSTOM, the CUSTOM will be stripped and
# the remainder of that string will be taken as a path to the type definition file

# Find the antechamber binary
  variable acpath
  global env

  if {$acpath == ""} {
    set  acpath \
    [::ExecTool::find -interactive \
      -description "Antechamber" \
      -path [file join $env(ACHOME) exe antechamber ] antechamber]
  }

  if {$acpath == ""} {
    error "Couldn't find antechamber executable. Please install antechamber and provide the path to your antechamber installation."
  }

# Print a banner giving credit where credit is due
  puts "************************************************************"
  puts "* Running ANTECHAMBER typing                               *"
  puts "* Please read and cite:                                    *"
  puts "*   J. Wang et al., J. Mol. Graph. Model. 25:247-260 (2006)*"
  puts "************************************************************"

### Sanity check input
  # Make sure we have hydrogens present
  set hydsel [atomselect [$selection molid] "[$selection text] and hydrogen"]
  
  if {[$hydsel num] == 0} {
    puts "WARNING: You are running antechamber on a structure with no hydrogens"
    puts "  You should build a structure with all hydrogens prior to running antechamber"
    puts "  Disregard this message if your molecule has no hydrogen"
  }

  $hydsel delete

  # Make sure we're using a valid charging method
  variable electypes

  if {[lsearch $electypes $chargetype] < 0} {
    puts "ERROR: Invalid charge method"
    puts " Valid choices are: [join $electypes]"
  }

  # see if we're using custom types
  set customtypes 0
  set customstring "CUSTOM"
  if {[string equal -length 6 $customstring $atomtypes]} {
    set customtypes 1
    set atomtypes [string range $atomtypes 6 end]
  }


# Write an input mol2 file
  $selection writemol2 antechamber-temp.mol2 

  if {$customtypes == 1} {
    set typestring "-d"
  } else {
    set typestring "-at"
  }

  # If we're keeping charges, write a charge file
  set delchargefile 0

  if {$chargetype == "rc"} {
    set charges [$selection get charge]
    set cfile [open "tmp-charges.ac.dat" "w"]
    for {set i 0} {$i < [llength $charges]} {incr i} {
      puts -nonewline $cfile "[lindex $charges $i] "
      if { [expr $i % 8] == 7 } {puts $cfile " "}
    }
    close $cfile

    set chargetype "$chargetype -cf tmp-charges.ac.dat"
    set delchargefile 1
  } 

# make sure we call divcon if needed
  set divconflag 0
  if {$chargetype == "cm1"} {set divconflag 1}


  if {$divconflag == 1} {
   puts "$acpath -fi mol2 -i antechamber-temp.mol2 -fo mol2 -o antechamber-temp.out.mol2 -nc $totcharge -s 2 -c $chargetype $typestring $atomtypes -j 4 -df 1 -m $spinmult 2> molefac.log"
   exec $acpath -fi mol2 -i antechamber-temp.mol2 -fo mol2 -o antechamber-temp.out.mol2 -nc $totcharge -s 2 -c $chargetype $typestring $atomtypes -j 4 -df 1 -m $spinmult 2> molefac.log
  } else {
  puts "$acpath -fi mol2 -i antechamber-temp.mol2 -fo mol2 -o antechamber-temp.out.mol2 -nc $totcharge -s 2 -c $chargetype $typestring $atomtypes -j 4 -m $spinmult 2>molefac.log"
  exec $acpath -fi mol2 -i antechamber-temp.mol2 -fo mol2 -o antechamber-temp.out.mol2 -nc $totcharge -s 2 -c $chargetype $typestring $atomtypes -j 4 -m $spinmult 2>molefac.log
  }

 if {$delchargefile > 0} {
   file delete "tmp-charges.ac.dat"
 }

# Load the output mol2 file
  set newmolid [mol new antechamber-temp.out.mol2]

  set sel [atomselect top all]

# clean up
 file delete [glob antechamber-temp*]

  return $newmolid
}

proc ::ANTECHAMBER::ac_type_in_place {selection {chargetype bcc} {totalcharge 0.0} {atomtypes gaff} {spinmult 1} {resname MOL}} {
# Wrapper around run_ac_typing that will apply the atom types, charges, 
#  and bonding pattern from antechamber to the selection in the original molecule
# In this case, the newly created molecule is then deleted

  set newmolid [run_ac_typing $selection $chargetype $totalcharge $atomtypes $spinmult $resname]
  puts $newmolid
  set newsel [atomselect $newmolid all]

  # Store the old names for use in tracking down bonds
  # This would be much easier if we could assume that the input is an isolated
  # fragment, but we can't/shouldn't 
  set oldnames [$selection get name]
  set oldids [$selection get index]
  set oldbonds [$selection getbonds]


  # Set the trivial properties
  puts [$selection get charge]
  puts [$newsel get charge]
  $selection set charge [$newsel get charge]
  $selection set type [$newsel get type]
  $selection set resname [$newsel get resname]
  $selection set name [$newsel get name]


  ### now work out the bonds
  set newnames [$newsel get name]
  set newids [$newsel get index]

  array set equivinds {};# array of oldindex->newindex pairs

  foreach oldname $oldnames oldid $oldids {
    # Find the equivalent in the new molecule
    set equivind [lindex $newids [lsearch -exact $oldname $newnames] ]
    array set equivinds {$oldid $equivind}
  }

  set fixedbonds [list]

  foreach oldbond $oldbonds newbond [$newsel getbonds] oldbo [$selection getbondorders] newbo [$newsel getbondorders] oldid $oldids {

  # If we have the same number of bonds, assume the order matches up
    if { [llength $oldbond] == [llength $newbond] } {
      lappend fixedbonds $newbo
    } else {
      # otherwise some bonds go outside of the selection
      #  note that oldbonds must then be a superset of newbonds
      set smalllist [list]
      set j 0
      for {set i 0} {$i < [llength $oldbond]} {incr i} {
        set myind [lindex $oldbond $i]
        set eqind $equivinds($myind)
        if { [lindex $newbond $j] == $eqind } {
          lappend smalllist [lindex $newbo $j]
          incr j
        } else {
          lappend smalllist [lindex $oldbo $j]
        }
      }

      lappend fixedbonds $smalllist
    }

  }

  $selection setbondorders $fixedbonds

  mol delete $newmolid
}

proc ::ANTECHAMBER::init_gui {} {
  variable atomsel all
  variable totcharge 0.0
  variable spinmult 1
  variable resname MOL
  variable inplace 0
  variable ante_type gaff
  variable ante_qtype bcc
  variable outfile ""
}


proc ::ANTECHAMBER::antechamber_gui { {molefacturemode 0}} {
# Just a simple gui for running antechamber in place on a selection
# This should be callable from most other plugins

# if molefacturemode is nonzero, only atoms with occupancy > 0.5 are used

  variable w
  variable molefmode
  variable inplace
  variable ante_qtype
  variable ante_type
  set molefmode $molefacturemode
  set inplace 1

  if { [winfo exists .antechambergui] } {
    wm deiconify .antechambergui
    return
  }

  init_gui
  
  set w [toplevel ".antechambergui"]
  wm title $w "Antechamber"

  set rownum 0

  frame $w.settings

  grid [label $w.settings.sellabel -text "Selection:"] -row $rownum -column 0 -sticky w
  grid [entry $w.settings.selection -width 30 -textvar ::ANTECHAMBER::atomsel] \
    -row $rownum -column 1 -columnspan 3 -sticky ew
  incr rownum

  grid [label $w.settings.chargelabel -text "Charge:"] -row $rownum -column 0 -sticky w
  grid [entry $w.settings.charge -width 5 -textvar ::ANTECHAMBER::totcharge] \
    -row $rownum -column 1 -sticky ew
  grid [label $w.settings.multlabel -text "Multiplicity:"] -row $rownum -column 2 -sticky ew
  grid [entry $w.settings.mult -width 5 -textvar ::ANTECHAMBER::spinmult] \
    -row $rownum -column 3 -sticky ew
  incr rownum

  grid [label $w.settings.rnlabel -text "Resname:"] -row $rownum -column 0 -sticky w
  grid [entry $w.settings.resname -width 6 -textvar ::ANTECHAMBER::resname] \
    -row $rownum -column 1 -sticky ew
  grid [label $w.settings.inplacelabel -text "Operate in place:"] -row $rownum -column 2 -sticky ew
  grid [checkbutton $w.settings.inplacebutton -variable ::ANTECHAMBER::inplace] \
    -row $rownum -column 3 -sticky ew
  incr rownum

  grid [label $w.settings.types -text "Atom types:"] -row $rownum -column 0 -sticky w
  grid [menubutton $w.settings.typemenu -menu $w.settings.typemenu.menu -textvar ::ANTECHAMBER::ante_type -relief raised] \
    -row $rownum -column 1 -columnspan 3 -sticky ew
  menu $w.settings.typemenu.menu -tearoff no
  $w.settings.typemenu.menu add radiobutton -label "GAFF" -variable ::ANTECHAMBER::ante_type -value "gaff"
  $w.settings.typemenu.menu add radiobutton -label "Amber" -variable ::ANTECHAMBER::ante_type -value "amber"
  $w.settings.typemenu.menu add radiobutton -label "BCC" -variable ::ANTECHAMBER::ante_type -value "bcc"
  $w.settings.typemenu.menu add radiobutton -label "Sybyl" -variable ::ANTECHAMBER::ante_type -value "sybyl"
  incr rownum

  grid [label $w.settings.charges -text "Atom charges:"] -row $rownum -column 0 -sticky w
  grid [menubutton $w.settings.chargemenu -menu $w.settings.chargemenu.menu -textvar ::ANTECHAMBER::ante_qtype -relief raised] \
    -row $rownum -column 1 -columnspan 3 -sticky ew
  menu $w.settings.chargemenu.menu -tearoff no
  $w.settings.chargemenu.menu add radiobutton -label "RESP" -variable ::ANTECHAMBER::ante_qtype -value "resp"
  $w.settings.chargemenu.menu add radiobutton -label "CM1" -variable ::ANTECHAMBER::ante_qtype -value "cm1"
  $w.settings.chargemenu.menu add radiobutton -label "ESP" -variable ::ANTECHAMBER::ante_qtype -value "esp"
  $w.settings.chargemenu.menu add radiobutton -label "Gasteiger" -variable ::ANTECHAMBER::ante_qtype -value "gas"
  $w.settings.chargemenu.menu add radiobutton -label "AM1-BCC" -variable ::ANTECHAMBER::ante_qtype -value "bcc"
  $w.settings.chargemenu.menu add radiobutton -label "CM2" -variable ::ANTECHAMBER::ante_qtype -value "cm2"
  $w.settings.chargemenu.menu add radiobutton -label "Mulliken" -variable ::ANTECHAMBER::ante_qtype -value "mul"
  $w.settings.chargemenu.menu add radiobutton -label "Keep current" -variable ::ANTECHAMBER::ante_qtype -value "rc"
  incr rownum

#  grid [label $w.settings.outflabel -text "Output file:"] -row $rownum -column 0 -sticky w
#  grid [entry $w.settings.outf -width 30 -textvar ::ANTECHAMBER::outfile] \
#    -row $rownum -column 1 -columnspan 3 -sticky ew
#  incr rownum

  grid [button $w.settings.rotf -text "Run ANTECHAMBER" -command [namespace current]::run_ante_gui] -row $rownum -column 0 -columnspan 4

  pack $w.settings
}

proc ::ANTECHAMBER::run_ante_gui {} {
  variable atomsel
  variable totcharge
  variable spinmult
  variable resname
  variable inplace
  variable ante_type
  variable ante_qtype
  variable outfile
  variable molefmode

  set atomselold $atomsel
  if {$molefmode == 1} {
    set atomsel "$atomsel and occupancy >= 0.8"
    mol top $::Molefacture::tmpmolid
  }
  set mysel [atomselect top "$atomsel"]
  set atomsel $atomselold


  if {$inplace == 1} {
    puts "typing in place"
    [namespace current]::ac_type_in_place $mysel $ante_qtype $totcharge $ante_type $spinmult $resname
  } else {
    [namespace current]::run_ac_typing $mysel $ante_qtype $totcharge $ante_type $spinmult $resname
  }

  $mysel delete

  if {$molefmode == 1} {
    ::Molefacture::update_openvalence
  }
}








  





