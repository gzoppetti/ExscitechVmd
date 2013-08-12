#Generate a single wall carbon nanotube

package require topotools 1.0

namespace eval ::Nanotube:: {
  variable w            ; # handle of the GUI's toplevel widget
  variable version  1.0 ; # version number of this plugin

  variable l
  variable n
  variable m
}
package provide nanotube $::Nanotube::version

proc nanotube { args } {return [eval ::Nanotube::nanotube_core $args] }

proc ::Nanotube::nanotube_usage { } {
  vmdcon -info "Usage: nanotube -l <length> -n <n> -m <m>"
  vmdcon -info "  <length> is length of nanotube in nanometers"
  vmdcon -info "  <n> and <m> are the chiral indices of the nanotube"
  error ""
}

proc ::Nanotube::nanotube_core { args } {
  # Check if proper #arguments was given
  set n_args [llength $args]
  if { [expr fmod($n_args,2)] } { nanotube_usage }
  if { $n_args < 6 && $n_args > 8 } { nanotube_usage }
   
  for { set i 0} {$i < $n_args} {incr i 2} {
    set key [lindex $args $i]
    set val [lindex $args [expr $i + 1]]
    set cmdline($key) $val
  }

  # Check if mandatory options are defined
  if { ![info exists cmdline(-l)] \
           || ![info exists cmdline(-n)] \
           || ![info exists cmdline(-m)] } {
    nanotube_usage
  }
  
  # Set nanotube parameters
  set length $cmdline(-l)
  set n $cmdline(-n)
  set m $cmdline(-m)
  set a 1.418
  set pi 3.14159265358979323846

  #Check that input is reasonable
  if { $n < 0 || $m < 0 || int($n) != $n || int($m) != $m} {error "n and m must be positive integers"}
  if {$m==0 && $n==0} {error "n and m can not both be zero"}
  if {$length <= 0} {error "Nanotube length must be a positive value"}

  #Calculate greatest common divisor d_R
  set num1 [expr 2*$m + $n]
  set num2 [expr 2*$n + $m]
  while { $num1 != $num2 } {
    if { $num1 > $num2 } {
      set num1 [expr $num1 - $num2]
    } else { 
      set num2 [expr $num2 - $num1] 
    }
  }
  set d_R $num1

  #Compute geometric properties
  set C [expr $a*sqrt(3*($n*$n + $m*$n + $m*$m))]
  set R [expr $C/(2*$pi)]
  set L_cell [expr sqrt(3)*$C/$d_R]

  #Number of unit cells
  set N_cell [expr ceil($length*10/$L_cell)]

  #Index min/max
  set pmin 0
  set pmax [expr int(ceil($n + ($n + 2*$m)/$d_R))]
#  set pmax [expr int(ceil($C*$n + $L_cell*($n + 2*$m)/sqrt(3)))]
#  set qmin [expr int(floor(-$L_cell*(2*$n + $m)/($C*sqrt(3))))]
  set qmin [expr int(floor(-(2*$n + $m)/$d_R))]
  set qmax $m
  set i 0

  #Generate unit cell coordinates
  for {set q $qmin} {$q <= $qmax} {incr q} {
    for {set p $pmin} {$p <= $pmax} {incr p} {

      #First basis atom
      set xprime1 [expr 3*$a*$a*($p*(2*$n + $m) + $q*($n + 2*$m))/(2*$C)]
      set yprime1 [expr 3*sqrt(3)*$a*$a*($p*$m - $q*$n)/(2*$C)]

      #Second basis atom
      set xprime2 [expr $xprime1 + 3*$a*$a*($n + $m)/(2*$C)]
      set yprime2 [expr $yprime1 - $a*$a*sqrt(3)*($n - $m)/(2*$C)]

      set phi1 [expr $xprime1/$R]
      set phi2 [expr $xprime2/$R]

      set x1 [expr $R*cos($phi1)]
      set x2 [expr $R*cos($phi2)]
      set y1 [expr $R*sin($phi1)]
      set y2 [expr $R*sin($phi2)]
      set z1 $yprime1
      set z2 $yprime2

      #Store coordinates of unit cell in an array
      #   0 <= xprime1 < C and 0 <= yprime1 < L_cell
      if {0 <= $xprime1 \
              && $p*(2*$n + $m) + $q*($n + 2*$m) < 2*($n*$n + $n*$m + $m*$m) \
              && 0 <= $yprime1 \
              && $d_R*($p*$m - $q*$n) < 2*($n*$n + $n*$m + $m*$m) } {
        set coord1($i,0) $x1
        set coord1($i,1) $y1
        set coord1($i,2) $z1
        
        set coord2($i,0) $x2
        set coord2($i,1) $y2
        set coord2($i,2) $z2

        incr i
      }
    }
  }

  set num_atom $i
  set k 0

  #Create new molecule with one frame
  set mol [mol new atoms [expr int(2 * $num_atom * $N_cell + 0.4999)]]
  animate dup $mol
  set sel [atomselect $mol all]
  #Set default values for all atoms
  foreach key {name resname element type mass radius} value {C CNT C CA 12.0107 1.7} {
    $sel set $key $value
  }
  molinfo $mol set {a b c} [list 100.0 100.0 [expr $N_cell*$L_cell]]

  #Generate nanotube
  set xyzlist {}
  for {set j 0} { $j < $N_cell } {incr j} {
    for {set i 0} {$i < $num_atom} {incr i} {
      lappend xyzlist [list $coord1($i,0) $coord1($i,1) [expr $coord1($i,2) + $j*$L_cell]]
      lappend xyzlist [list $coord2($i,0) $coord2($i,1) [expr $coord2($i,2) + $j*$L_cell]]
    }
  }
  $sel set {x y z} $xyzlist
  $sel delete

  #Add representation for molecule
  mol rename $mol "($m,$n) Carbon Nanotube"
  mol bondsrecalc $mol
  mol reanalyze $mol
  ::TopoTools::adddefaultrep $mol
  # have better look at the CNT.
  rotate x by -90

  return $mol
}


proc ::Nanotube::nanotube_gui {} {
    variable w
    variable l
    variable n
    variable m

    if { [winfo exists .nanotube] } {
        wm deiconify $w
        return
    }

    set w [toplevel ".nanotube"]
    wm title $w "Nanotube"
    wm resizable $w no no
    set row 0

    set ::Nanotube::l 5
    set ::Nanotube::n 5
    set ::Nanotube::m 10

    #Add a menubar
    frame $w.menubar -relief raised -bd 2 -padx 10
    grid  $w.menubar -padx 1 -column 0 -columnspan 4 -row $row -sticky ew
    menubutton $w.menubar.help -text "Help" -underline 0 \
    -menu $w.menubar.help.menu
    $w.menubar.help config -width 5
    pack $w.menubar.help -side right
    menu $w.menubar.help.menu -tearoff no
    $w.menubar.help.menu add command -label "About" \
    -command {tk_messageBox -type ok -title "About Nanotube Builder" \
    -message "Carbon nanotube building tool.\n\nVersion $::Nanotube::version\n(c) 2009 by Robert R. Johnson <robertjo@physics.upenn.edu>"}
    $w.menubar.help.menu add command -label "Help..." \
    -command "vmd_open_url [string trimright [vmdinfo www] /]/plugins/nanotube"
    incr row

    grid [label $w.nlabel -text "Nanotube chiral index n: "] \
    -row $row -column 0 -columnspan 3 -sticky w
    grid [entry $w.n -width 7 -textvariable ::Nanotube::n] -row $row -column 3 -columnspan 1 -sticky ew
    incr row

    grid [label $w.mlabel -text "Nanotube chiral index m: "] \
    -row $row -column 0 -columnspan 3 -sticky w
    grid [entry $w.m -width 7 -textvariable ::Nanotube::m] -row $row -column 3 -columnspan 1 -sticky ew
    incr row

    grid [label $w.llabel -text "Nanotube length (nm): "]\
    -row $row -column 0 -columnspan 3 -sticky w
    grid [entry $w.l -width 7 -textvariable ::Nanotube::l] -row $row -column 3 -columnspan 1 -sticky ew
    incr row

    grid [button $w.gobutton -text "Generate Nanotube" \
      -command [namespace code {
        vmdcon -info "calling nanotube_core -l $l -n $n -m $m"
        nanotube_core -l "$l" -n "$n" -m "$m"
      } ]] -row $row -column 0 -columnspan 4 -sticky nsew
}

proc nanotube_tk {} {
  ::Nanotube::nanotube_gui
  return $::Nanotube::w
}

