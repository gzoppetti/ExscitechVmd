#timeline.tcl  -- VMD script to list/select 2D trajectory info
# of a protein molecule
#
# Copyright (c) 2008 The Board of Trustees of the University of Illinois
#
# Barry Isralewitz  barryi@ks.uiuc.edu    
# vmd@ks.uiuc.edu
#
#
# $Id: timeline.tcl,v 1.58 2009/07/24 17:44:33 barryi Exp $

package provide timeline 2.0

proc timeline {} {
  return [::timeline::startTimeline]
}

#####
#
#  Timeline programming notes
#  
#  The dataname array provides names
# of the columns of $dataVal(column,row)
# The column names are called fields,
# but (except for 0 through 2) represent
# data for animation frames. 
#  Here any column from ($dataOrigin) to ($numFrames + $dataOrigin -1)
# contains the same kind of data, with frame 0 in ($dataOrigin), frame 1 in ($dataOrigin+1).
#
# dataname(0) resid
# dataname(1) resname
# dataname(2) chain
# dataname(3) (whatever most recently was calculated,
#              may soon be 'user' always
# dataname(4) and higher: (not initialized)
#            but the associated dataVal() columns
#            contain whatever is in dataname(3) 
#
#    The dataname array is a leftover from
# when timeline's vertical columns were fields
# for multiple columns of differnt sorts of data
# (as seen in the related Sequence Viewer plugin)
#  Currently does have some uses in decided how to color data,
# but does not seem necessary for every frame, just per molec.
# 
#
#- 
#  xcol() could hold the pixel location of every
#column, but since all are the same, we only assign xcol($dataOrigin).
# and calculate from there.  If used for all columns, could track resiable winows.
#
#

#######################
#create the namespace
######################t#
namespace eval ::timeline  {
    variable clicked "-1"
    #XXX can we eliminate these next 4 lines by using local vars?
    variable oldmin "0.0"
    variable oldmax "2.0"
    variable oldAnyResFuncDesc ""
    variable oldAnyResFuncName ""
    variable oldFirstAnalysisFrame ""
    variable oldLastAnalysisFrame ""
    variable lastCalc "0"
       # last calculation, see ::recalc switch statement for other codes.
       #XXX is lastCalc doing any good right now?
       #XXX lastCalc _should_ be used (even in calcTestHbonds) after Appearance:Set Scaling.
}


####################/
#define the procs
####################
proc ::timeline::tlPutsDebug {theText} {
 #puts "*TL DEBUG* $theText"
}

proc ::timeline::recalc {} {
  
  variable lastCalc
  variable firstAnalysisFrame
  variable lastAnalysisFrame
  variable currentMol
  variable dataMin
  variable dataMax
  set dataMin(all) null
  set dataMax(all) null
  #XXX get rid of integer-associations here nd in menu calls, too error prone
  switch $lastCalc {
  -1  { showall 1}
  0   { clearData; showall 1}
  1   { calcDataStruct; showall 1 }
  2   { calcDataX; showall 1 }
  3   { calcDataY; showall 1 }
  4   { calcDataZ; showall 1 }
  5   { calcDataPhi; showall 1 }
  6   { calcDataDeltaPhi; showall 1 }
  7   { calcDataPsi; showall 1 }
  8   { calcDataDeltaPsi; showall 1 }
  9   { ::rmsdtool ; showall 1 }
  10  { calcTestFreeSel 10;showall 1 }
  11  { calcTestHbonds 11;showall 1 }
  12  { calcDataUser; showall 1 }
  13  { calcDataAnyResFunc; showall 1 }
  14  { calcDisplacement; showall 1 }
  15  { calcDispVelocity; showall 1 }
  16  { calcSaltBridge 16; showall 1 }
  }
}

proc ::timeline::canvasScrollY {args} { 
  variable w

  eval $w.can yview $args
  eval $w.vertScale yview $args 
}     
proc ::timeline::canvasScrollX {args} { 
  variable w

  eval $w.can xview $args
  eval $w.horzScale xview $args 
  eval $w.threshGraph xview $args 
  
  return
}


proc ::timeline::lookupCode {resname} {
  variable codes

  set result ""
  if {[catch { set result $codes($resname) } ]} {
    set result $resname
  } else {
    set result " $result "
  }
  return $result
}

proc ::timeline::stopZoomSeq {} {
  menu timeline off
}

proc ::timeline::chooseColor {field intensity} {
  variable dataName
  set field_color_type 4 
  #hack to default to struct field type coloring
  if {$dataName($field) != "struct"} {
    if {$intensity < 0} {set intensity 0}
    if {$intensity > 255} {set intensity 255}
    set intensity [expr int($intensity)]
    #set field_color_type $field 
    #check color mapping
    set field_color_type 3 
  }
  #super hacky here
  switch -exact $field_color_type {         
    #temporaily diable so greyscale color  only
    3333 {   
      set red $intensity
      set green [expr 255 - $intensity]
      set blue 150 
    }
    4 {
      #the field_color_type hack sends all structs to here 
      if { [catch {
        switch $intensity {
          
######
## CMM 08/28/06 mmccallum@pacific.edu
##
# modify the colors displayed in order to better match what shows up in the
# "structure" representation.  Please note that I have set 3_{10} 
# helices to be blue, to provide more contrast between the purple (alpha)
# and default mauve/pinkish for 3_{10} helices from the "structure" rep
####
#  This gives blue = 3_{10}, purple = alpha, red = pi helix
###################################################
          B {set red 180; set green 180; set blue 0}
          C {set red 255; set green 255; set blue 255}
          E {set red 255; set green 255; set blue 100}
          T {set red 70; set green 150; set blue 150}
          # G = 3_{10}
          G {set red 20; set green 20; set blue 255}
          # H = alpha;  this was fine-tuned a bit to match better.
          H {set red 235; set green 130; set blue 235}
          I {set red 225; set green 20; set blue 20}
          default {set red 100; set green 100; set blue 100}
        }
        
      } ] 
         } { #badly formatted file, intensity may be a number
        set red 0; set green 0; set blue 0 
      }
    }
    default {
      #set red [expr 200 - int (200.0 * ($intensity / 255.0) )]
      #set green $red
      #set red 140; set blue 90; set green 90;
      set red $intensity
      set green $intensity
      set blue $intensity
    }
  }
  
  #convert red blue green 0 - 255 to hex
  set hexred     [format "%02x" $red]
  set hexgreen   [format "%02x" $green]
  set hexblue    [format "%02x" $blue]
  set hexcols [list $hexred $hexgreen $hexblue]

  return $hexcols
}


proc ::timeline::redraw {name func op} {
  
  variable x1 
  variable y1 
  variable so
  variable w 
  variable monoFont
  variable xcanwindowmax 
  variable ycanwindowmax 
  variable xcanmax 
  variable ycanmax
  variable ybox 
  variable xsize 
  variable ysize 
  variable resnamelist 
  variable structlist 
  variable betalist 
  variable sel 
  variable canvasnew 
  variable scalex 
  variable scaley 
  variable dataVal 
  variable dataValNum 
  variable dataOrigin 
  variable dataName 
  variable ytopmargin 
  variable ybottommargin 
  variable vertTextSkip   
  variable xcolbond_rad 
  variable bond_res 
  variable rep 
  variable xcol 
  variable vertTextRight
  variable vertHighLeft
  variable vertHighRight
  variable resCodeShowOneLetter 
  variable dataWidth 
  variable dataMargin 
  variable dataMin
  variable dataMax 
  variable trajMin
  variable trajMax
  variable xPosScaleVal
  variable usableMolLoaded
  variable rectCreated
  variable prevScalex
  variable prevScaley
  variable numFrames

  if { ($usableMolLoaded) && ($dataValNum >=0 ) } {
    set ysize [expr $ytopmargin+ $ybottommargin + ($scaley *  $ybox * ($dataValNum + 1) )]  

    set xsize [expr  $xcol($dataOrigin) +  ($scalex *  $dataWidth *  $numFrames)  ] 

    set ycanmax(data) $ysize
    set ycanmax(vert) $ycanmax(data)
    set xcanmax(data) $xsize
    set xcanmax(horz) $xcanmax(data)
    if {$ycanmax(data) < $ycanwindowmax} {
      set ycanmax(data) $ycanwindowmax
    }


    if {$xcanmax(data) < $xcanwindowmax} {
      set xcanmax(data) $xcanwindowmax
    }

    $w.can configure -scrollregion "0 0 $xcanmax(data) $ycanmax(data)"
    $w.vertScale configure -scrollregion "0 0 $xcanmax(vert) $ycanmax(data)"
    $w.horzScale configure -scrollregion "0 0 $xcanmax(data) $ycanmax(horz)"
    $w.threshGraph configure -scrollregion "0 0 $xcanmax(data) $ycanmax(horz)"
    drawVertScale
    drawHorzScale
    
    
    #for example, if we have 2 frames of data, frame 0 and frame 1,
    #then numFrames = 2.  Since dataOrigin =3, fieldLast is 4, since data
    # is in field 3 (frame 0), field 4 (frame 1). Formula is...
    set fieldLast [expr $dataOrigin + $numFrames -1 ]

    #draw data on can
    #loop over all data fields

    if {! $rectCreated} {
      #this until separate data and scale highlighting
      $w.threshGraph delete xScalable
      $w.horzScale delete xScalable
      $w.vertScale delete yScalable
      $w.can delete dataScalable
      #puts "drawing rects, scalex is $scalex"
      #hack here -- for now skip B-field stuff, so minimal stuff drawn
      tlPutsDebug ": setting min/max, dataOrigin= $dataOrigin" 
      for {set field [expr $dataOrigin ]} {$field <= $fieldLast} {incr field} {
        
        
        set xPosFieldLeft [expr int  ( $xcol($dataOrigin) + ($scalex * $dataWidth * ($field - $dataOrigin)  ) ) ]
        set xPosFieldRight [expr int ( $xcol($dataOrigin) + ($scalex * $dataWidth * ($field - $dataOrigin + 1 - $dataMargin)  ) ) ]
        
        #now draw data rectangles
        #puts "drawing field $field at xPosField $xPosField" 
        #yipes, does this redraw all rects (even non visible) every timeXXX
        set y 0.0
        
        set intensity 0
        
        for {set i 0} {$i<=$dataValNum} {incr i} { 
          set val $dataVal($field,$i)
          if {$val != "null"} {
            #calculate color and create rectange
            
            set ypos [expr $ytopmargin + ($scaley * $y)]
            
            #should Prescan  to find range of values!   
            #this should be some per-request-method range / also allow this to be adjusted
            
            #set intensity except if field 4 (indexed struct)
            #puts "field = $field, dataName($field) = $dataName($field),i= $i" 
            if {$dataName($field) != "struct"} {
              ##if { ( ($field != 4)  ) } open brace here 
              #set range [expr $dataMax($field) - $dataMin($field)]
              set range [expr $trajMax - $trajMin ]
              if { ($range > 0)  && ([string is double $val] )} {
                set intensity  [expr int (255. * ( (0.0 + $val - $trajMin ) / $range)) ]
                #tlPutsDebug ": $val $dataMin($field) $range $field $intensity"
              }
              
              
              
              set hexcols [chooseColor $field $intensity]
            } else {
              #horrifyingly, sends string for data, tcl is typeless
              set hexcols [chooseColor $field $val ]
            }
            foreach {hexred hexgreen hexblue} $hexcols {} 

            
            #draw data rectangle
            $w.can create rectangle  [expr $xPosFieldLeft] [expr $ypos ] [expr $xPosFieldRight]  [expr $ypos + ($scaley * $ybox)]  -fill "\#${hexred}${hexgreen}${hexblue}" -outline "" -tags dataScalable
          }
          
          set y [expr $y + $ybox]
        }
      }

      drawVertHighlight 
    }  else {

      #$w.can scale dataRect $xcol($firstdata) $ytopmargin 1 $scaley
      #$w.can scale dataScalable $xcol($dataOrigin) [expr $ytopmargin] 1 [expr $scaley / $prevScaley ]

      $w.can scale dataScalable $xcol($dataOrigin) [expr $ytopmargin] [expr $scalex / $prevScalex]  [expr $scaley / $prevScaley ]
      #now for datarect
      $w.vertScale scale yScalable 0 [expr $ytopmargin] 1  [expr $scaley / $prevScaley ]
      $w.horzScale scale xScalable $xcol($dataOrigin) 0 [expr $scalex / $prevScalex ] 1
      $w.threshGraph scale xScalable $xcol($dataOrigin) 0 [expr $scalex / $prevScalex ] 1

    } 
   tlPutsDebug "redraw: about to set rectCreated" 
    
    set rectCreated 1
    set prevScaley $scaley
    set prevScalex $scalex
  }

  tlPutsDebug "done with redraw"
  return
}



proc ::timeline::makecanvas {} {

  variable xcanmax 
  variable ycanmax
  variable w
  variable xsize
  variable ysize 
  variable xcanwindowmax 
  variable ycanwindowmax
  variable threshGraphHeight
  variable horzScaleHeight
  variable vertScaleWidth 
  set xcanmax(data) $xsize 
  set ycanmax(data) $ysize
  
  
  #make main canvas




  
  canvas $w.spacer1 -width [expr $vertScaleWidth+20] -height [expr $threshGraphHeight + $horzScaleHeight + 25] -bg #A0A0A0
  canvas $w.spacer2 -width [expr $vertScaleWidth+20] -height [expr $threshGraphHeight + $horzScaleHeight + 25] -bg #C0C0E0
  canvas $w.can -width [expr $xcanwindowmax] -height $ycanwindowmax -bg #E9E9D9 -xscrollcommand "$w.xs set" -yscrollcommand "$w.ys set" -scrollregion  "0 0 $xcanmax(data) $ycanmax(data)" 
  canvas $w.vertScale -width $vertScaleWidth -height $ycanwindowmax -bg #C0D0C0 -yscrollcommand "$w.ys set" -scrollregion "0 0 $vertScaleWidth $ycanmax(data)" 

  canvas $w.threshGraph -width $xcanwindowmax -height  $threshGraphHeight  -scrollregion "0 0 $xcanmax(data) $threshGraphHeight" -bg #DDDDDD -xscrollcommand "$w.xs set"
  canvas $w.horzScale -width $xcanwindowmax -height  $horzScaleHeight  -scrollregion "0 0 $xcanmax(data) $horzScaleHeight" -bg #A9A9A9 -xscrollcommand "$w.xs set"
  #pack the horizontal (x) scrollbar
  pack $w.spacer1 -in $w.cfr -side left  -anchor e  
  pack $w.spacer2 -in $w.cfr -side bottom -anchor s  
  pack $w.can  -in $w.cfr -side left -anchor sw 
  #vertical scale/labels
  place $w.vertScale -in $w.can -relheight 1.0 -relx 0.0 -rely 0.5 -bordermode outside -anchor e
  #now place the vertical (y) scrollbar
  place $w.ys -in $w.vertScale -relheight 1.0 -relx 0.0 -rely 0.5 -bordermode outside -anchor e
  #now place the horizontal threshold Graph
  place $w.threshGraph -in $w.can -relwidth 1.0 -relx 0.5 -rely 1.0 -width 1 -bordermode outside -anchor n
  # horizontal scale/labels
  place $w.horzScale -in $w.threshGraph -relwidth 1.0 -relx 0.5 -rely 1.0 -bordermode outside -anchor n
  #now place the horizontal (x) scrollbar
  place $w.xs -in $w.horzScale -relwidth 1.0 -relx 0.5 -rely 1.0 -bordermode outside -anchor n

  # may need to specify B1-presses shift/nonshift separately...
  bind $w.can <ButtonPress-1>  [namespace code {getStartedMarquee %x %y 0 1 data}]
  bind $w.can <ButtonPress-3>  [namespace code {getStartedMarquee %x %y 0 3 data}]
  bind $w.can <Shift-ButtonPress-1>  [namespace code {getStartedMarquee %x %y 1 1 data}]
  bind $w.can <Control-ButtonPress-1>  [namespace code {timeBarJumpPress %x %y 0 data}]
  bind $w.can <Control-ButtonRelease-1>  [namespace code {timeBarJumpRelease %x %y 0 data}]
  bind $w.can <ButtonPress-2>  [namespace code {timeBarJumpPress %x %y 0 data}]
  bind $w.can <ButtonRelease-2>  [namespace code {timeBarJumpRelease %x %y 0 data}]
  bind $w.can <B1-Motion>  [namespace code {keepMovingMarquee %x %y 1 data}]
  bind $w.can <B3-Motion>  [namespace code {keepMovingMarquee %x %y 3 data}]
  bind $w.can <B2-Motion>  [namespace code {timeBarJump %x %y 0 data}]
  bind $w.can <Control-B2-Motion>  [namespace code {timeBarJump %x %y 0 data}]
  bind $w.can <ButtonRelease-1> [namespace code {letGoMarquee %x %y 1 data}]
  bind $w.can <ButtonRelease-3> [namespace code {letGoMarquee %x %y 3 data}]

  bind $w.vertScale <ButtonPress-1>  [namespace code {getStartedMarquee %x %y 0 1 vert}]
  bind $w.vertScale <ButtonPress-3>  [namespace code {getStartedMarquee %x %y 0 3 vert}]
  bind $w.vertScale <Shift-ButtonPress-1>  [namespace code {getStartedMarquee %x %y 1 1 vert}]
  bind $w.vertScale <ButtonPress-2>  [namespace code {timeBarJumpPress %x %y 0 vert}]
  bind $w.vertScale <ButtonRelease-2>  [namespace code {timeBarJumpRelease %x %y 0 data}]
  bind $w.vertScale <B1-Motion>  [namespace code {keepMovingMarquee %x %y 1 vert}]
  bind $w.vertScale <B3-Motion>  [namespace code {keepMovingMarquee %x %y 3 vert}]
  bind $w.vertScale <B2-Motion>  [namespace code {timeBarJump %x %y 0 vert}]
  bind $w.vertScale <ButtonRelease-1> [namespace code {letGoMarquee %x %y 1 vert}]
  bind $w.vertScale <ButtonRelease-3> [namespace code {letGoMarquee %x %y 3 vert}]

  bind $w.horzScale <ButtonPress-1>  [namespace code {getStartedMarquee %x %y 0 1 horz}]
  bind $w.horzScale <ButtonPress-3>  [namespace code {getStartedMarquee %x %y 0 3 horz}]
  bind $w.horzScale <Shift-ButtonPress-1>  [namespace code {getStartedMarquee %x %y 1 1 horz}]
  bind $w.horzScale <ButtonPress-2>  [namespace code {timeBarJumpPress %x %y 0 horz}]
  bind $w.horzScale <ButtonRelease-2>  [namespace code {timeBarJumpRelease %x %y 0 data}]
  bind $w.horzScale <B1-Motion>  [namespace code {keepMovingMarquee %x %y 1 horz}]
  bind $w.horzScale <B3-Motion>  [namespace code {keepMovingMarquee %x %y 3 horz}]
  bind $w.horzScale <B2-Motion>  [namespace code {timeBarJump %x %y 0 horz}]
  bind $w.horzScale <ButtonRelease-1> [namespace code {letGoMarquee %x %y 1 horz}]
  bind $w.horzScale <ButtonRelease-3> [namespace code {letGoMarquee %x %y 3 horz}]
  lower $w.spacer1 $w.cfr
  lower $w.spacer2 $w.cfr
  
  return
} 


proc ::timeline::reconfigureCanvas {} {
  variable xcanmax
  variable ycanmax
  variable w
  variable ysize 
  variable xcanwindowmax 
  variable ycanwindowmax
  variable threshGraphHeight
  variable horzScaleHeight
  variable vertScaleWidth
  variable xcanwindowStarting
  variable xcanwindowmax 
  variable dataOrigin
  variable xcol

  #in future, add to xcanwindowstarting if we widen window
  set xcanwindowmax  $xcanwindowStarting 


  #check if can cause trouble if no mol loaded...
  $w.can configure  -height $ycanwindowmax -width $xcanwindowmax 
  $w.horzScale configure  -height  $horzScaleHeight  -scrollregion "0 0 $xcanmax(data) $horzScaleHeight"
  $w.threshGraph configure -height  $threshGraphHeight  -scrollregion "0 0 $xcanmax(data) $horzScaleHeight"

  $w.vertScale configure  -width $vertScaleWidth -scrollregion "0 0 $vertScaleWidth $ycanmax(data)" 
  $w.horzScale delete all
  $w.vertScale delete all
  $w.can delete all

}

proc ::timeline::draw_traj_highlight {xStart xFinish} {

  variable w 
  variable dataVal 
  variable dataValNum 
  variable dataOrigin
  variable xcol 
  variable ytopmargin 
  variable ytopmargin 
  variable scaley
  variable ybox  
  variable currentMol 
  variable rep 
  variable bond_rad 
  variable bond_res
  variable repColoring
  variable rectCreated

  tlPutsDebug " now in draw_traj_highlight, xStart = $xStart, rectCreated = $rectCreated"
  $w.can delete trajHighlight 
  for {set i 0} {$i<=$dataValNum} {incr i} {
    if  {$dataVal(picked,$i) == 1} {
      set ypos [expr $ytopmargin+ ($scaley * $i *$ybox)]
      
      set red 0 
      set green 0 
      set blue 255 
      #convert red blue green 0 - 255 to hex
      set hexred     [format "%02x" $red]
      set hexgreen   [format "%02x" $green]
      set hexblue    [format "%02x" $blue]
      

      ###draw highlight only if not yet drawn -- if rectCreated is 0, we may just cleared the rects
      ###     to redraw free of accumulated scaling errors
      ###if {($dataVal(pickedId,$i) == "null") || ($rectCreated == 0)} 
      
      #always draw trajBox
      #after prototype, merge this with normal highlight draw method
      #set trajBox [$w.can create rectangle  $xStart $ypos $xFinish [expr $ypos + ($scaley * $ybox)]  -fill "\#${hexred}${hexgreen}${hexblue}" -stipple gray25 -outline "" -tags [list dataScalable trajHighlight ] ]
      set trajBox [$w.can create rectangle  $xStart $ypos $xFinish [expr $ypos + ($scaley * $ybox)]  -fill "" -outline "\#${hexred}${hexgreen}${hexblue}" -tags [list dataScalable trajHighlight ] ]
      #puts "trajBox is $trajBox, xStart = $xStart, $xFinish = $xFinish"
      
      #$w.can lower $dataVal(pickedId,$i) vertScaleText 
      
      
      
    }
  }
}

proc ::timeline::drawVertHighlight  {} {

  variable w 
  variable dataVal 
  variable dataValNum 
  variable dataOrigin
  variable xcol 
  variable ytopmargin 
  variable scaley
  variable ybox  
  variable currentMol 
  variable rep 
  variable bond_rad 
  variable bond_res
  variable repColoring
  variable rectCreated
  variable vertHighLeft
  variable vertHighRight
  variable usesFreeSelection

  set red 255
  set green 0
  set blue 255
  #convert red blue green 0 - 255 to hex
  set hexred     [format "%02x" $red]
  set hexgreen   [format "%02x" $green]
  set hexblue    [format "%02x" $blue]
  set highlightColorString    "\#${hexred}${hexgreen}${hexblue}" 

  for {set i 0} {$i<=$dataValNum} {incr i} {
    if  {$dataVal(picked,$i) == 1} {
      set ypos [expr $ytopmargin+ ($scaley * $i *$ybox)]
      
      
      #draw highlight only if not yet drawn -- if rectCreated is 0, we may  just cleared the rects
      #     to redraw free of accumulated scaling errors
      if {($dataVal(pickedId,$i) == "null") || ($rectCreated == 0)} {

        set dataVal(pickedId,$i)  [$w.vertScale create rectangle  $vertHighLeft $ypos $vertHighRight [expr $ypos + ($scaley * $ybox)]  -fill $highlightColorString -outline "" -tags yScalable]
        
        
        $w.vertScale lower $dataVal(pickedId,$i) vertScaleText 
        
      }
      
    }
  }

  
  #make selection string to display in VMD 
  set ll "" 
  set prevChain "Empty" 


  #altered change for multi free selections
  #Cannot be held by chain  

  for {set i 0} {$i <= $dataValNum} {incr i} {
    if {$dataVal(picked,$i) == 1} {
      if $usesFreeSelection {
          append ll ") or ($dataVal(freeSelString,$i)"
      } else {
        if { [string compare $prevChain $dataVal(2,$i)] != 0} {
          #chain is new or has changed
          append ll ") or (chain $dataVal(2,$i)  and resid $dataVal(0,$i)"
        } else {
          append ll " $dataVal(0,$i)"
        }
        set prevChain $dataVal(2,$i)
      }
    }  
   }
  append ll ")"
  set ll [string trimleft $ll ") or " ]
  
  #check for the state when mol first loaded
  if {$ll ==""} {
    set ll "none"
  } 
  
  
  if {($rep($currentMol) != "null")} {

    if { [expr [molinfo $currentMol get numreps] -1] >= $rep($currentMol) } {

      mol modselect $rep($currentMol) $currentMol $ll
    } else {
      createHighlight  $ll      
    }
  } else {
    createHighlight  $ll        
    #mol selection $ll
    #mol modstyle $rep($currentMol)  $currentMol Bonds $bond_rad $bond_res
    #mol color ColorID 11 
    #get info about this
  }
  return
}

proc ::timeline::showCursorHighlight {selText} {

  variable currentMol
  variable cursor_bond_rad
  variable cursor_bone_res
  variable cursorRepColor
  variable cursorRep

  if {($cursorRep($currentMol) != "null")} {

    if { [expr [molinfo $currentMol get numreps] -1] >= $cursorRep($currentMol) } {

      mol modselect $cursorRep($currentMol) $currentMol $selText
    } else {
      createCursorHighlight  $selText      
    }
  } else {
    createCursorHighlight  $selText        
  }

}

proc ::timeline::hideCursorHighlight {selText} {

  variable currentMol
  variable cursor_bond_rad
  variable cursor_bone_res
  variable cursorRepColor
  variable cursorRep

  if {($cursorRep($currentMol) != "null")} {

    if { [expr [molinfo $currentMol get numreps] -1] >= $cursorRep($currentMol) } {

      mol showrep $currentMol $cursorRep($currentMol) 0
 
    } else {
      createCursorHighlight  $selText      
      mol showrep $currentMol $cursorRep($currentMol) 0
    }
  } else {
    createCursorHighlight  $selText        
    mol showrep $currentMol $cursorRep($currrentMol) 0
  }

}

proc ::timeline::revealCursorHighlight {selText} {
  tlPutsDebug "revealCursorHighlight"
  #code copy from showCursorHighlight XXX 
  variable currentMol
  variable cursor_bond_rad
  variable cursor_bone_res
  variable cursorRepColor
  variable cursorRep

  if {($cursorRep($currentMol) != "null")} {

    if { [expr [molinfo $currentMol get numreps] -1] >= $cursorRep($currentMol) } {

      mol showrep $currentMol $cursorRep($currentMol) 1
 
    } else {
      createCursorHighlight  $selText      
      mol showrep $currentMol $cursorRep($currentMol) 1
    }
  } else {
    createCursorHighlight  $selText        
    mol showrep $currentMol $cursorRep($currentMol) 1
  }

}


proc ::timeline::listPick {name element op} {
  
  global vmd_pick_atom 
  global vmd_pick_mol 
  global vmd_pick_shift_state  

  variable w 
  variable xcanmax
  variable ycanmax
  variable xcanwindowmax 
  variable ycanwindowmax
  variable ybox
  variable ytopmargin 
  variable ybottommargin 
  variable vertTextSkip 
  variable scaley 
  variable dataVal 
  variable dataValNum 
  variable dataOrigin
  variable dataName 
  variable bond_rad 
  variable bond_res 
  variable repColoring
  variable rep 
  variable xcol 
  variable ysize 
  variable dataOrigin
  variable currentMol
  # get the coordinates



  #later deal with top (and rep)  etc. for multi-mol use


  
  if {$vmd_pick_mol == $currentMol} {
    
    set sel [atomselect $currentMol "index $vmd_pick_atom"]
    
    set pickedresid [lindex [$sel get {resid}] 0] 
    set pickedchain  [lindex [$sel get {chain}] 0] 
    set pickedresname [lindex  [$sel get {resname}] 0]
    
    
    set pickedOne -1
    #XXX must be changed for free Selections
    for {set i 0} {$i <= $dataValNum} {incr i} {
      
      if {($dataVal(0,$i) == $pickedresid) && ($dataVal(1,$i) == $pickedresname) &&  ($dataVal(2,$i) == $pickedchain)} {
        set pickedOne $i
        
        break
      }
    }
    
    if {$pickedOne >= 0} {
      set ypos [expr $ytopmargin+ ($scaley * $i *$ybox)]
      
      #do bitwise AND to check for shift-key bit
      if {$vmd_pick_shift_state & 1} {
        set shiftPressed 1
      } else {
        set shiftPressed 0
      }
      

      
      if {$shiftPressed == 0 } {
        #delete all from canvas

        for {set i 0} {$i <= $dataValNum} {incr i} {
          set dataVal(picked,$i) 0
          if {$dataVal(pickedId,$i) != "null"} {
            $w.can delete $dataVal(pickedId,$i)
            set dataVal(pickedId,$i) "null"
          }
        }
      }
      
      
      set dataVal(picked,$pickedOne) 1
      
      drawVertHighlight 
      
      #scroll to picked
      set center [expr $ytopmargin + ($ybox * $scaley * $pickedOne) ] 
      set top [expr $center - 0.5 * $ycanwindowmax]
      
      if {$top < 0} {
        set top 0
      }
      set yfrac [expr $top / $ysize]
      $w.can yview moveto $yfrac
      $w.vertScale yview moveto $yfrac
    }
    
  }
  return
}



proc ::timeline::timeLineMain {} {
#------------------------
  #------------------------
  # main code starts here
  #vars initialized a few lines down
  

  #puts "in timeLineMain.."
  variable w 
  variable monoFont
  variable eo 
  variable x1 
  variable y1 
  variable startShiftPressed 
  variable startCanvas
  variable vmd_pick_shift_state 
  variable resCodeShowOneLetter 
  variable so 
  variable xcanwindowmax 
  variable ycanwindowmax 
  variable xcanmax 
  variable ycanmax
  variable ybox 
  variable ysize 
  variable resnamelist 
  variable structlist 
  variable betalist 
  variable sel 
  variable canvasnew 
  variable scaley 
  variable dataVal 
  variable dataOrigin
  variable dataHash
  variable rectId
  #dataValNum is -1 if no data present, 
  variable dataValNum 
  variable dataValNumResSel
  variable dataName 
  variable ytopmargin 
  variable ybottommargin 
  variable xrightmargin
  variable vertTextSkip   
  variable xcolbond_rad 
  variable bond_res 
  variable repColoring
  variable cursor_bond_res
  variable cursor_bond_rad
  variable prevCursorObject
  variable prevCursorFrame
  variable bond_rad 
  variable bond_res
  variable rep 
  variable cursorRep
  variable cursorShown
  variable xcol 
  variable resCodeShowOneLetter 
  variable dataWidth 
  variable dataMargin 
  variable dataMin 
  variable dataMax 
  variable xPosScaleVal
  variable currentMol
  variable fit_scalex
  variable fit_scaley
  variable usableMolLoaded 
  variable initializedVars
  variable prevScalet
  variable rectCreated
  variable windowShowing
  variable needsDataUpdate 
  variable numFrames
  variable firstAnalysisFrame
  variable lastAnalysisFrame

  # check for usable molecule loaded
  set uml 0
  foreach mm [molinfo list] {
    if {([molinfo $mm get numatoms] > 0 )} {
      set uml 1
    }
  }
  set usableMolLoaded $uml
      

  #Init vars and draw interface
  if {$initializedVars == 0} {
    initVars
    draw_interface
    makecanvas
    set initializedVars 1
    #watch the slider value, tells us when to redraw
    #this sets a trace for ::timeline::scaley
    
  } else {
    #even if no molecule is present
    reconfigureCanvas
  }   
  
  
  #-----
  #Now load info from the current molecule, must reload for every molecule change
  
  if {$usableMolLoaded} {
    #get info for new mol
    #set needsDataUpdate 0
    
    #The number of dataNames
    
    #lets fill  a ((dataOrigin-1)+numFrames) x (dataValNumResSel +1) array
    #dataValNumResSel we'll be the number of objects we found with VMD search
    #if doing proteins and DNA, likely all residues, found with 'name CA' or 'name C3*;, etc.
    #the items 0 through dataOrigin-1 (count=dataOrigin) are the 3 identifiers of residue
    #the items dataOrigin through dataOrigin+(numFrames-1) (count=numFrames) is the data for the frames.
    # The more general term (for both per-residue and free selections) will be dataValNum
    set dataValNumResSel -1
    #if no data is available, dataValNum will remain -1 
    #we are looking for dataVal when only a single res sel per line 

    # set  a new  trace below, only if dataValNum > -1  
    # following check likely is no longer necessary
    if {[molinfo $currentMol get numatoms] >= 1} {

      
      set currentMol_name [molinfo $currentMol get name]
      wm title $w "VMD Timeline  $currentMol_name (mol $currentMol) "
      set sel [atomselect $currentMol "(all and name CA) or ( (not protein) and  ( (name \"C3\\'\") or (name \"C3\\*\") ) ) "] 
      # gets 1 atom per protein or nucleic acid residue
      #below assumes sel retrievals in same order each time, fix this 
      #by changing to one retreival and chopping up result
      set datalist  [$sel get {resid resname chain}]
      puts "Info) Timeline is checking sequence info. for molecule $currentMol..."

      catch {unset dataHash}
      
      foreach elem $datalist {
        
        incr dataValNumResSel
        #set picked state to false -- 'picked' is only non-numerical field
        set dataVal(picked,$dataValNumResSel) 0
        set dataVal(pickedId,$dataValNumResSel) "null"
        set theResid [ lindex [split $elem] 0]
        set dataVal(0,$dataValNumResSel) $theResid 
        
        set dataVal(1,$dataValNumResSel) [ lindex [split $elem] 1]
        set dataVal(1code,$dataValNumResSel) [lookupCode $dataVal(1,$dataValNumResSel)]
        set theChain [ lindex [split $elem] 2]
        set dataVal(2,$dataValNumResSel) $theChain 
        #for fast index searching later
        set dataHash($theResid,$theChain) $dataValNumResSel
      }
      #if datalist is length 0 (empty), dataValNum is still -1, 
      #So we must check before each use of dataValNum     
      
      #set the molec. structure so nothing is highlighted yet
      set rep($currentMol) "null"
      set cursorRep($currentMol) "null"
      set prevCursorObject($currentMol) "null"
      set cursorShown($currentMol) 0
      set prevCursorFrame($currentMol) "null"
   } 
   # so dataValNum <= -1 if no sequence or atoms at all
    
    set dataValNum $dataValNumResSel   
    if {$dataValNum <= -1 } {
      puts "Info) Timeline couldn't find a sequence in this molecule.\n"
       return
    }
   
    
    
    
    #So dataValNum is number of the last dataVal.  It is also #elements -1, 
    
    #numFrames (and routines that use it)  will eventualy be changed
    # to reflect loaded data, and  multi-frame-data groups
    set numFrames [molinfo $currentMol get numframes]
    if {$numFrames >= 1} then {
           set fitNF $numFrames
    } else {
           set fitNF 1
    }
    set firstAnalysisFrame 0
    set  lastAnalysisFrame [expr $numFrames - 1]
    if {$lastAnalysisFrame < $firstAnalysisFrame} { 
      set lastAnalysisFrame $firstAnalysisFrame
    } 

    set fit_scalex [expr (0.0 + $xcanwindowmax - $xcol($dataOrigin) ) / ($dataWidth * $fitNF ) ]
    set fit_scaley [expr (0.0 + $ycanwindowmax - $ytopmargin - $ybottommargin) / ($ybox * ($dataValNum + 1) ) ]
    #since we zero-count dataValNum.

    set scaley 1.0
    set scalex $fit_scalex 
    tlPutsDebug "Timeline: Restarting data, scalex = $scalex, scaley= $scaley"
    #this trace only set if dataValNum != -1

    #Other variable-adding methods
    #should not change this number.  We trust $sel to always
    #give dataValNum elems, other methods might not work as well.
    
    
    #handle if this value is 0 or -1
    
    
    #don't need datalist anymore
    unset datalist 
    
    
    
    #now lets fill in some data/
    
    #new data, so need to redraw rects when time comes
    set rectCreated 0 
    #also set revScaley back to 1 
    set prevScaley scaley
    set prevScalex scalex 
    #value of dataNameNum is 2. last is numbered (dataNameLast) = 2
    
    


    #fill in traj data with X position (very fast) 
    tlPutsDebug "Timeline about to fill in with calcDataX, may not have cleared if first time" 
    calcDataX
    
  } 
  
  #puts "time for first redraw, scales, min/max not calced"
  #redraw first time
  redraw name func ops
  
  #now draw the scales (after the data, we may need to extract min/max 
  #------
  #draw color legends, loop over all data fields
  #puts "dataName(0) is $dataName(0) dataName(1) is $dataName(1)"
  

  return
}


proc ::timeline::molChooseMenu {name function op} {
  variable w

  variable usableMolLoaded
  variable currentMol
  variable prevMol
  variable nullMolString
  variable dataOrigin
  $w.mol.menu delete 0 end



  set molList ""
  foreach mm [molinfo list] {
    if {([molinfo $mm get numatoms] > 0 )} {
      lappend molList $mm
      #add a radiobutton, but control via commands, not trace,
      #since if this used a trace, the trace's callback
      #would delete that trace var, causing app to crash.
      #variable and value only for easy button lighting
      ##$w.mol.menu add radiobutton -variable [namespace current]::currentMol -value $mm -label "$mm [molinfo $mm get name]" -command [namespace code "molChoose name function op"]
      $w.mol.menu add radiobutton -variable [namespace current]::currentMol -value $mm -label "$mm [molinfo $mm get name]"
    }
  }

  #set if any non-Graphics molecule is loaded
  if {$molList == ""} {
    set usableMolLoaded  0
    if {$prevMol != $nullMolString} {
      set currentMol $nullMolString
    }
  } else {

    #deal with first (or from-no mol state) mol load
    # and, deal with deletion of currentMol, if mols present
    # by setting the current mol to first usable mol in list
    if {($usableMolLoaded == 0) || [lsearch -exact $molList $currentMol]== -1 } {
      set usableMolLoaded 1
      #  
      # old line was: set currentMol [molinfo top]: works with auto-yop
      # but top could be an unsable molec, instrad use first usable in list
      set currentMol [lindex $molList 0] 
    }

  }


  
  
  return
}

proc ::timeline::setThresholdBounds {args} {
  variable clicked
  variable thresholdBoundMin 
  variable thresholdBoundMax
  variable oldmin
  variable oldmax

  # save old values 
  set oldmin $thresholdBoundMin 
  set oldmax $thresholdBoundMax

  set d .thresholdboundsdialog
  catch {destroy $d}
  toplevel $d -class Dialog
  wm title $d {Set threshold bounds for Timeline}
  wm protocol $d WM_DELETE_WINDOW {set ::timeline::clicked -1}
  wm minsize  $d 220 120  

  # only make the dialog transient if the parent is viewable.
  if {[winfo viewable [winfo toplevel [winfo parent $d]]] } {
      wm transient $d [winfo toplevel [winfo parent $d]]
  }

  frame $d.bot
  frame $d.top
  $d.bot configure -relief raised -bd 1
  $d.top configure -relief raised -bd 1
  pack $d.bot -side bottom -fill both
  pack $d.top -side top -fill both -expand 1

  # dialog contents:
  label $d.head -justify center -relief raised -text {Set threshold bounds for Timeline:}
    pack $d.head -in $d.top -side top -fill both -padx 6m -pady 6m
    grid $d.head -in $d.top -column 0 -row 0 -columnspan 2 -sticky snew 
    label $d.la  -justify left -text {Bottom value:}
    label $d.lb  -justify left -text {Top value:}
    set i 1
    grid columnconfigure $d.top 0 -weight 2
    foreach l "$d.la $d.lb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 0 -row $i -sticky w 
        incr i
    }

    entry $d.ea  -justify left -textvariable ::timeline::thresholdBoundMin
    entry $d.eb  -justify left -textvariable ::timeline::thresholdBoundMax
    set i 1
    grid columnconfigure $d.top 1 -weight 2
    foreach l "$d.ea $d.eb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 1 -row $i -sticky w 
        incr i
    }
    
    # buttons
    button $d.ok -text {OK} -command {::timeline::thresholdMakeGraph ; set ::timeline::clicked 1}
    grid $d.ok -in $d.bot -column 0 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 0
    button $d.cancel -text {Cancel} -command {set ::timeline::thresholdBoundMin $timeline::oldmin ; set ::timeline::thresholdBoundMax $::timeline::oldmax ; set ::timeline::clicked 1}
    grid $d.cancel -in $d.bot -column 1 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 1

    bind $d <Destroy> {set ::timeline::clicked -1}
    set oldFocus [focus]
    set oldGrab [grab current $d]
    if {[string compare $oldGrab ""]} {
        set grabStatus [grab status $oldGrab]
    }
    grab $d
    focus $d

    # wait for user to click
    vwait ::timeline::clicked
    catch {focus $oldFocus}
    catch {
        bind $d <Destroy> {}
        destroy $d
    }
    if {[string compare $oldGrab ""]} {
      if {[string compare $grabStatus "global"]} {
            grab $oldGrab
      } else {
          grab -global $oldGrab
        }
    }
    return
}

proc ::timeline::setAnalysisFrames {args} {
  variable clicked
  variable firstAnalysisFrame
  variable lastAnalysisFrame
  variable oldFirstAnalysisFrame
  variable oldLastAnalysisFrame
   #the "any" refers to any function

  # save old values 
  set oldFirstAnalysisFrame $firstAnalysisFrame
  set oldLastAnalysisFrame $lastAnalysisFrame


  set d .vmd_timeline_setanalysisframesialog
  catch {destroy $d}
  toplevel $d -class Dialog
  wm title $d {Set analysis frames for Timeline}
  wm protocol $d WM_DELETE_WINDOW {set ::timeline::clicked -1}
  wm minsize  $d 320 120  

  # only make the dialog transient if the parent is viewable.
  if {[winfo viewable [winfo toplevel [winfo parent $d]]] } {
      wm transient $d [winfo toplevel [winfo parent $d]]
  }

  frame $d.bot
  frame $d.top
  $d.bot configure -relief raised -bd 1
  $d.top configure -relief raised -bd 1
  pack $d.bot -side bottom -fill both
  pack $d.top -side top -fill both -expand 1

  # dialog contents:
  label $d.head -justify center -relief raised -text {Analysis frame range:}
    pack $d.head -in $d.top -side top -fill both -padx 6m -pady 6m
    grid $d.head -in $d.top -column 0 -row 0 -columnspan 2 -sticky snew 
    label $d.la  -justify left -text {First frame:}
    label $d.lb  -justify left -text {Last frame:}
    set i 1
    grid columnconfigure $d.top 0 -weight 2
    foreach l "$d.la $d.lb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 0 -row $i -sticky w 
        incr i
    }

    entry $d.ea  -justify left -textvariable ::timeline::firstAnalysisFrame
    entry $d.eb  -justify left -textvariable ::timeline::lastAnalysisFrame
    set i 1
    grid columnconfigure $d.top 1 -weight 2
    foreach l "$d.ea $d.eb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 1 -row $i -sticky w 
        incr i
    }
    
    # buttons
    button $d.ok -text {OK} -command {::timeline::recalc ; set ::timeline::clicked 1}
    grid $d.ok -in $d.bot -column 0 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 0
    button $d.cancel -text {Cancel} -command {set ::timeline::anyResFuncName $timeline::oldAnyResFuncName ; set ::timeline::anyResFuncDesc $::timeline::oldAnyResFuncDesc ; set ::timeline::clicked 1}
    grid $d.cancel -in $d.bot -column 1 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 1

    bind $d <Destroy> {set ::timeline::clicked -1}
    set oldFocus [focus]
    set oldGrab [grab current $d]
    if {[string compare $oldGrab ""]} {
        set grabStatus [grab status $oldGrab]
    }
    grab $d
    focus $d

    # wait for user to click
    vwait ::timeline::clicked
    catch {focus $oldFocus}
    catch {
        bind $d <Destroy> {}
        destroy $d
    }
    if {[string compare $oldGrab ""]} {
      if {[string compare $grabStatus "global"]} {
            grab $oldGrab
      } else {
          grab -global $oldGrab
        }
    }
    return
}


proc ::timeline::setAnyResFunc {args} {
  variable clicked
  variable anyResFuncDesc
  variable anyResFuncName
  variable oldAnyResFuncDesc
  variable oldAnyResFuncName
   #the "any" refers to any function

  # save old values 
  set oldAnyResFuncDesc $anyResFuncDesc
  set oldAnyResFuncName $anyResFuncName

  set d .vmd_timeline_setanyresdialog
  catch {destroy $d}
  toplevel $d -class Dialog
  wm title $d {Set Every-Residue Function for Timeline}
  wm protocol $d WM_DELETE_WINDOW {set ::timeline::clicked -1}
  wm minsize  $d 320 120  

  # only make the dialog transient if the parent is viewable.
  if {[winfo viewable [winfo toplevel [winfo parent $d]]] } {
      wm transient $d [winfo toplevel [winfo parent $d]]
  }

  frame $d.bot
  frame $d.top
  $d.bot configure -relief raised -bd 1
  $d.top configure -relief raised -bd 1
  pack $d.bot -side bottom -fill both
  pack $d.top -side top -fill both -expand 1

  # dialog contents:
  label $d.head -justify center -relief raised -text {Set per-residue function:}
    pack $d.head -in $d.top -side top -fill both -padx 6m -pady 6m
    grid $d.head -in $d.top -column 0 -row 0 -columnspan 2 -sticky snew 
    label $d.la  -justify left -text {Function (TCL proc)}
    label $d.lb  -justify left -text {Label for the function:}
    set i 1
    grid columnconfigure $d.top 0 -weight 2
    foreach l "$d.la $d.lb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 0 -row $i -sticky w 
        incr i
    }

    entry $d.ea  -justify left -textvariable ::timeline::anyResFuncName
    entry $d.eb  -justify left -textvariable ::timeline::anyResFuncDesc
    set i 1
    grid columnconfigure $d.top 1 -weight 2
    foreach l "$d.ea $d.eb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 1 -row $i -sticky w 
        incr i
    }
    
    # buttons
    button $d.ok -text {OK} -command {::timeline::recalc ; set ::timeline::clicked 1}
    grid $d.ok -in $d.bot -column 0 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 0
    button $d.cancel -text {Cancel} -command {set ::timeline::anyResFuncName $::timeline::oldAnyResFuncName ; set ::timeline::anyResFuncDesc $::timeline::oldAnyResFuncDesc ; set ::timeline::clicked 1}
    grid $d.cancel -in $d.bot -column 1 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 1

    bind $d <Destroy> {set ::timeline::clicked -1}
    set oldFocus [focus]
    set oldGrab [grab current $d]
    if {[string compare $oldGrab ""]} {
        set grabStatus [grab status $oldGrab]
    }
    grab $d
    focus $d

    # wait for user to click
    vwait ::timeline::clicked
    catch {focus $oldFocus}
    catch {
        bind $d <Destroy> {}
        destroy $d
    }
    if {[string compare $oldGrab ""]} {
      if {[string compare $grabStatus "global"]} {
            grab $oldGrab
      } else {
          grab -global $oldGrab
        }
    }
    return
}


proc ::timeline::setScaling {args} {
  variable clicked
  variable trajMin
  variable trajMax 

  # save old values 
  set oldmin $trajMin
  set oldmax $trajMax

  set d .scalingdialog
  catch {destroy $d}
  toplevel $d -class Dialog
  wm title $d {Set Scaling for Timeline}
  wm protocol $d WM_DELETE_WINDOW {set ::timeline::clicked -1}
  wm minsize  $d 220 120  

  # only make the dialog transient if the parent is viewable.
  if {[winfo viewable [winfo toplevel [winfo parent $d]]] } {
      wm transient $d [winfo toplevel [winfo parent $d]]
  }

  frame $d.bot
  frame $d.top
  $d.bot configure -relief raised -bd 1
  $d.top configure -relief raised -bd 1
  pack $d.bot -side bottom -fill both
  pack $d.top -side top -fill both -expand 1

  # dialog contents:
  label $d.head -justify center -relief raised -text {Set scaling for timeline:}
    pack $d.head -in $d.top -side top -fill both -padx 6m -pady 6m
    grid $d.head -in $d.top -column 0 -row 0 -columnspan 2 -sticky snew 
    label $d.la  -justify left -text {Bottom value:}
    label $d.lb  -justify left -text {Top value:}
    set i 1
    grid columnconfigure $d.top 0 -weight 2
    foreach l "$d.la $d.lb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 0 -row $i -sticky w 
        incr i
    }

    entry $d.ea  -justify left -textvariable ::timeline::trajMin
    entry $d.eb  -justify left -textvariable ::timeline::trajMax
    set i 1
    grid columnconfigure $d.top 1 -weight 2
    foreach l "$d.ea $d.eb" {
        pack $l -in $d.top -side left -expand 1 -padx 3m -pady 3m
        grid $l -in $d.top -column 1 -row $i -sticky w 
        incr i
    }
    
    # buttons
    button $d.ok -text {OK} -command {::timeline::recalc ; set ::timeline::clicked 1}
    grid $d.ok -in $d.bot -column 0 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 0
    button $d.cancel -text {Cancel} -command {set ::timeline::trajMin $timeline::oldmin ; set ::timeline::trajMax $::timeline::oldmax ; set ::timeline::clicked 1}
    grid $d.cancel -in $d.bot -column 1 -row 0 -sticky ew -padx 10 -pady 4
    grid columnconfigure $d.bot 1

    bind $d <Destroy> {set ::timeline::clicked -1}
    set oldFocus [focus]
    set oldGrab [grab current $d]
    if {[string compare $oldGrab ""]} {
        set grabStatus [grab status $oldGrab]
    }
    grab $d
    focus $d

    # wait for user to click
    vwait ::timeline::clicked
    catch {focus $oldFocus}
    catch {
        bind $d <Destroy> {}
        destroy $d
    }
    if {[string compare $oldGrab ""]} {
      if {[string compare $grabStatus "global"]} {
            grab $oldGrab
      } else {
          grab -global $oldGrab
        }
    }
    return
}


proc ::timeline::printCanvas {} {
  variable w
  #extract first part of molecule name to print here?
  set filename "VMD_Timeline_Window.ps"
  set filename [tk_getSaveFile -initialfile $filename -title "VMD Timeline Print" -parent $w -filetypes [list {{Postscript Files} {.ps}} {{All files} {*} }] ]
  if {$filename != ""} {
    $w.can postscript -file $filename
  }
  
  return
}





proc ::timeline::getStartedMarquee {x y shiftState whichButtonPressed whichCanvas} {

  variable w 
  variable x1 
  variable y1 
  variable so
  variable str 
  variable eo 
  variable g 
  variable startCanvas 
  variable startShiftPressed
  variable xcanmax
  variable ycanmax
  variable usableMolLoaded
  variable marqueeButton
  
  
  if {$usableMolLoaded} {

    #calculate offset for canvas scroll
    set startShiftPressed $shiftState   
    set marqueeButton $whichButtonPressed
    set startCanvas $whichCanvas 
    #get actual name of canvas
    switch -exact $startCanvas {
      data {set drawCan can}
      vert {set drawCan vertScale}
      horz {set drawCan horzScale}
      default {
          #puts "problem with finding canvas..., startCanvas= >$startCanvas<"
      } 
    }   
    set x [expr $x + $xcanmax($startCanvas) * [lindex [$w.$drawCan xview] 0]] 
    set y [expr $y + $ycanmax($startCanvas) * [lindex [$w.$drawCan yview] 0]] 
    tlPutsDebug "getStarted, y= $y, yview =  [$w.$drawCan yview]" 
    set x1 $x
    set y1 $y
    

    #puts "getStartedMarquee x= $x  y= $y, startCanvas= $startCanvas" 
    #Might have other canvas tools in future..   
    # Otherwise, start drawing rectangle for selection marquee
    
    
   if {$marqueeButton==1} {
     set outlineColor "blue"
   } else {
     set outlineColor "green"
   }   
    set so [$w.$drawCan create rectangle $x $y $x $y -fill {} -outline $outlineColor]
    set eo $so
  } 
  return
}


proc ::timeline::molChoose {name function op} {

  variable scaley
  variable w
  variable currentMol
  variable prevMol
  variable nullMolString
  variable rep 
  variable usableMolLoaded
  variable needsDataUpdate
  variable windowShowing
  variable dataOrigin 
  variable usesFreeSelection

  #this does complete restart
  #can do this more gently...
  
  #trace vdelete scaley w [namespace code redraw]
  #trace vdelete ::vmd_pick_atom w  [namespace code listPick] 
  
  #if there's a mol loaded, and there was an actual non-graphic mol last
  #time, and if there has been a selection, and thus a struct highlight
  #rep made, delete the highlight rep.
  if {($usableMolLoaded)  && ($prevMol != $nullMolString) && ($rep($prevMol) != "null")} {



    #catch this since currently is exposed to user, so 
    #switching/reselecting  molecules can fix problems.
    ##puts "About to delete rep=$rep($prevMol) for prevMol= $prevMol"
    #determine if this mol exists...
    if  {[lsearch -exact [molinfo list] $prevMol] != -1}  {
      #determine if this rep exists (may have been deleted by user)
      if { [expr [molinfo $prevMol get numreps] -1] >= $rep($prevMol) } { 
        
        mol delrep $rep($prevMol) $prevMol 
      }
    }
    
  }

  set prevMol $currentMol

  #can get here when window is not displayed if:
  #   molecule is loaded, other molecule delete via Molecule GUI form.
  # So, we'll only redraw (and possible make a length (wallclock) call
  # to chosen analysis method) if timeline window is showing
  
  set needsDataUpdate 1
  set usesFreeSelection 0

  if {$windowShowing} {
    set needsDataUpdate 0
    #set this immediately, so other  calls can see this
    
    [namespace current]::timeLineMain
  }


  
  #reload/redraw stuff, settings (this may elim. need for above lines...)
  
  
  #change molecule choice and redraw if needed (visible && change) here...
  #change title of window as well
  ##wm title $w "VMD Timeline  $currentMol_name (mol $currentMol) "
  
  #reload sutff (this may elim. need for above lines...)

  return
}
proc ::timeline::keepMovingMarquee {x y whichButtonPressed whichCanvas} {

  variable x1 
  variable y1 
  variable so 
  variable w 
  variable xcanmax 
  variable ycanmax
  variable startCanvas
  variable usableMolLoaded
  #get actual name of canvas
  switch -exact $startCanvas {
    data {set drawCan can}
    vert {set drawCan vertScale}
    horz {set drawCan horzScale}
    default {tlPutsDebug "Info) Timeline: had problem with finding canvas (moving marquee)..., startCanvas= $startCanvas"}
  } 

  
  if {$usableMolLoaded} {

    #next two lines for debugging only
    set windowx $x
    set windowy $y 
    #calculate offset for canvas scroll
    set x [expr $x + $xcanmax($startCanvas) * [lindex [$w.$drawCan xview] 0]] 
    set y [expr $y + $ycanmax($startCanvas) * [lindex [$w.$drawCan yview] 0]] 
    
    
    
    
    $w.$drawCan coords $so $x1 $y1 $x $y
  }
  return
}

proc ::timeline::initPicked {} {
  variable dataVal
  variable dataValNum
  variable w
  for {set i 0} {$i <= $dataValNum} {incr i} {
    set dataVal(picked,$i) 0
    set dataVal(pickedId,$i) "null"
  }
}

proc ::timeline::clearAllPicked {} {
  variable dataVal
  variable dataValNum
  variable w
  for {set i 0} {$i <= $dataValNum} {incr i} {
    set dataVal(picked,$i) 0
    if {$dataVal(pickedId,$i) != "null"} {
      $w.vertScale delete $dataVal(pickedId,$i)
      set dataVal(pickedId,$i) "null"
    }
  }
}
proc ::timeline::letGoMarquee {x y whichButtonPressed whichCanvas} {


  variable x1 
  variable y1 
  variable startShiftPressed 
  variable marqueeButton
  variable startCanvas
  variable so 
  variable eo 
  variable w 
  variable xsize
  variable ysize
  variable xcanmax
  variable ycanmax
  variable ySelStart 
  variable ySelFinish 
  variable ybox 
  variable ytopmargin 
  variable ybottommargin 
  variable xcanwindowmax
  variable ycanwindowmax
  variable vertTextSkip 
  variable scalex 
  variable scaley 
  variable dataVal 
  variable dataValNum 
  variable dataOrigin
  variable dataName 
  variable bond_rad 
  variable bond_res 
  variable repColoring
  variable rep 
  variable xcol
  variable currentMol
  variable usableMolLoaded
  variable dataOrigin
  variable dataWidth 
  variable ycanwindowmax  
  variable numFrames
  variable fit_scalex
  variable fit_scaley
  variable userScalex
  variable userScaley
  variable userScaleBoth
  #set actual name of canvas
  switch -exact $startCanvas {
    data {set drawCan can}
    vert {set drawCan vertScale}
    horz {set drawCan horzScale}
    default {puts "Info) Timeline: problem with finding canvas (moving marquee)..., startCanvas= $startCanvas"}
  }

  if {$usableMolLoaded} {
    #calculate offset for canvas scroll
    set x [expr $x + $xcanmax(data) * [lindex [$w.$drawCan xview] 0]] 
    set y [expr $y + $ycanmax(data) * [lindex [$w.$drawCan yview] 0]] 
    tlPutsDebug "yview=  [$w.$drawCan yview]    y= $y"
    #compute the frame at xSelStart
    if {$x1 < $x} {
      set xSelStart $x1
      set xSelFinish $x
    }  else {
      set xSelStart $x
      set xSelFinish $x1
    }
    #puts "xSelStart is $xSelStart xSelFinish is $xSelStart" 
    
    #in initVars we hardcode dataOrigin to be 3
    #later, there may be many field-groups that can be stretched 
    tlPutsDebug "scalex= $scalex, dataWidth= $dataWidth  xSelStart= $xSelStart"
    set selStartFrame [expr  int (($xSelStart - $xcol($dataOrigin))/ ($dataWidth * $scalex))  ]
    set selFinishFrame [expr int( ($xSelFinish - $xcol($dataOrigin))/ ($dataWidth * $scalex) ) ]
    #puts "checking limits, numFrames = $numFrames, selStartFrame= $selStartFrame   selFinishFrame= $selFinishFrame"
    if { $selStartFrame < 0} {
      set selStartFrame  0
    } 
   
    if { $selFinishFrame <  0 } {
      set selFinishFrame  0 
    }
    if { $selStartFrame >=$numFrames } {
      set selStartFrame  [expr $numFrames -1]
    } 
   
    if { $selFinishFrame >= $numFrames} {
      set selFinishFrame [expr $numFrames -1]
    }
    #puts "selected frames $selStartFrame to   $selFinishFrame"

    if {$y1 < $y} {
      set ySelStart $y1
      set ySelFinish $y}  else {
        
        set ySelStart $y
        set ySelFinish $y1
      }
    
    set startObject [expr 0.0 + ((0.0 + $ySelStart - $ytopmargin) / ($scaley * $ybox))]
    set finishObject [expr 0.0 + ((0.0 + $ySelFinish - $ytopmargin) / ($scaley * $ybox))]
    
    
    if {$startShiftPressed == 1} {
      set singleSel 0
    } else {
      set singleSel 1
    }
    
    if {$startObject < 0} {set startObject 0}
    if {$finishObject < 0} {set finishObject 0}
    if {$startObject > $dataValNum} {set startObject   $dataValNum }
    if {$finishObject > $dataValNum} {set finishObject $dataValNum }
    set startObject [expr int($startObject)]
    set finishObject [expr int($finishObject)]
    
    #optimizations obvious, much math repeated...
    set xStartFrame [expr  ( ($selStartFrame  ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    #stretch across width of ending frame
    set xFinishFrame [expr  ( ($selFinishFrame+ 1.0) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    tlPutsDebug "  xStartFrame= $xStartFrame    xFinishFrame= $xFinishFrame xsize= $xsize"
 

    if {$marqueeButton==1}  {
      #highlight for animation
      
      #clear all if click/click-drag, don't clear if shift-click, shift-click-drag
      
      if {$singleSel == 1} {
        clearAllPicked
      } else {
        
        #just leave alone 
      }
      
      
      
      
      #set flags for selection
      for {set i $startObject} {$i <= $finishObject} {incr i} {
        set dataVal(picked,$i) 1
      }
      
      
      
      set field 0
      #note that the column will be 0, but the data will be from picked
      
      drawVertHighlight 
      
      
      #puts "now to delete outline, eo= $eo" 
      $w.$drawCan delete $eo
      $w.can delete timeBarRect 
      #now that highlight changed, can animate
      #if single selection in frame area, animate, then jump to that frame

           if {$startCanvas=="data"} { 
        if {  $selStartFrame >= 0 } {
          if {$selFinishFrame > $selStartFrame} {
            #draw a box to show selected animation

            

            #puts "now to  draw_traj_highlight $xStartFrame $xFinishFrame"
            draw_traj_highlight $xStartFrame $xFinishFrame

            set xTimeBarEnd  [expr  ( ($selStartFrame + 1.0) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
            
            #set timeBar [$w.can create rectangle  $xStartFrame 1 $xTimeBarEnd [expr $ycanmax(data) ]   -fill "\#000000" -stipple gray50 -outline "" -tags [list dataScalable timeBarRect ] ]
            set timeBar [$w.can create rectangle  $xStartFrame 1 $xTimeBarEnd [expr $ycanmax(data) ]   -fill "" -outline "\#000000"  -tags [list dataScalable timeBarRect ] ]
            display update ui
            #maybe store top, and restore it afterwards
            mol top $currentMol
            #make this controllable, and animation repeatable without
            #need to reselect
            #soon, we move this loop, and anim will happen only at button push XXX
            for {set r 1} {$r <= 1} {incr r} {
              for {set f $selStartFrame} {$f <= $selFinishFrame} {incr f} {
                #puts "time for anim. goto is [time {animate goto $f} ]"
                #puts "time for draw = [time { drawTimeBar $f}]"
                #puts "time for disp update = [time {display update ui}]" 
                animate goto $f
                drawTimeBar $f
                display update ui 
              }
            }
            $w.can delete timeBarRect 

          } 
          animate goto $selStartFrame
          drawTimeBar $selStartFrame 
          puts "now jumped to frame $selStartFrame for molecule $currentMol"
        }
      } 

    } else {
      $w.$drawCan delete $eo
      # zoom to requested position
      tlPutsDebug "  START scale calcs, scalex= $scalex  scaley= $scaley"
     
      if { ($x1==$x) && ($y1==$y) } {
         #zoom out
        tlPutsDebug "zoom out -- x= $x  x1= $x1  ySelStart= $ySelStart startObject= $startObject ySelFinish= $ySelFinish finishObject=$finishObject startShiftPressed= $startShiftPressed"
         set scaleFacX 0.8
         set scaleFacY 0.8
         # place these in middle

         set leftborder [expr $x - 0.5 * $xcanwindowmax]
         if {$leftborder < 0}  {
           set leftborder 0
         } 

         set topborder [expr $y - 0.5 * $ycanwindowmax]
         if {$topborder < 0} {
           set topborder 0
         }
         set xf_low [expr $leftborder/$xsize]
         set yf_low [expr $topborder/$ysize]
      } else { 
        tlPutsDebug "zoom in x= $x  x1= $x1 ySelStart= $ySelStart startObject= $startObject ySelFinish= $ySelFinish finishObject=$finishObject startShiftPressed= $startShiftPressed"
        set marqueeBoxesHeight [expr $finishObject - $startObject]
        set marqueeBoxesWidth [expr $selFinishFrame- $selStartFrame]
     
       if {$marqueeBoxesWidth<= 3} then {set marqueeBoxesWidth 3}
       if {$marqueeBoxesHeight<=3} then {set marqueeBoxesHeight 3}
       tlPutsDebug "marqueeBoxesWidth= $marqueeBoxesWidth  marqueeBcxesHeight= $marqueeBoxesHeight\n    dataWidth= $dataWidth   ybox= $ybox   xsize= $xsize ysize= $ysize"
       #set scaleFacX  [expr $xcanmax(data)/( $marqueeBoxesWidth* $dataWidth)]
       #set scaleFacY  [expr $ycanmax(data)/($ytopmargin+$ybottommargin+ $marqueeBoxesHeight* $ybox)]
       set scaleFacX  [expr $xcanwindowmax/( $marqueeBoxesWidth* $dataWidth *$scalex)]
       set scaleFacY  [expr $ycanwindowmax/( $marqueeBoxesHeight* $ybox * $scaley)]
       #set xf_low [expr  ($fit_scalex * $newScaleX* ($xcol($dataOrigin) +($selStartFrame* $dataWidth)))/$xcanmax(data)]
       #set xf_high [expr  $newScaleX* ($xcol($dataOrigin) +( ($selStartFrame+ $marqueeBoxesWidth) * $dataWidth))]
       #set yf_low [expr  ($fit_scaley* $newScaleY * ($ytopmargin + ($startObject * $marqueeBoxesHeight)))/$ycanmax(data)]
       #set yf_high [expr  $newScaleY * ($ytopmargin + ($startObject * $marqueeBoxesHeight))]
       
       tlPutsDebug "scalex= $scalex  scaley= $scaley"
       set xf_low [expr $xSelStart/$xsize]
       set yf_low [expr $ySelStart/$ysize]
     }
     #ignore zoom if already zoomed in too far
     if { (2 * $scalex * $dataWidth) < $xcanwindowmax} {
        set scalex [expr $scaleFacX * $scalex]
        tlPutsDebug "setting scalex"
     }
     if {(2 * $scaley * $ybox )< $ycanwindowmax} {
       set scaley [expr $scaleFacY * $scaley]
       tlPutsDebug "setting scaley"
     }

     set userScalex [expr $scalex/ $fit_scalex ]
     set userScaley [expr $scaley/$fit_scaley ]
     set userScaleBoth $userScalex 
     tlPutsDebug "letGoMarquee: scalex= $scalex  scaley= $scaley scaleFacX= $scaleFacX  scaleFacY= $scaleFacY  userScalex= $userScalex   userScaley=$userScaley  fit_scaley= $fit_scaley **"
     #XXX find why causes odd problems
     #$w.zoomXlevel set $userScalex
     #$w.panl.zoomlevel set $userScaley
     redraw name func ops
     tlPutsDebug "xf_low= $xf_low  yf_low=$yf_low" 
     canvasScrollX moveto $xf_low 
     canvasScrollY moveto $yf_low 
    ### samp set fit_scalex [expr (0.0 + $xcanwindowmax - $xcol($dataOrigin) ) / ($dataWidth * ( $numFrames) ) ]
    ### sample  set fit_scaley [expr (0.0 + $ycanwindowmax - $ytopmargin - $ybottommargin) / ($ybox * ($dataValNum + 1) ) ]
    }
  }
  return
}

proc ::timeline::showall { do_redraw} {



  variable scalex 
  variable scaley 
  variable fit_scalex
  variable fit_scaley
  variable usableMolLoaded
  variable rectCreated 
  variable userScalex
  variable userScaley
  variable usesFreeSelection
 
  #only redraw once...
  if {$usableMolLoaded} {
    if {$do_redraw == 1} {
      set rectCreated 0
    }   
    
    set scalex $fit_scalex        
    set scaley $fit_scaley
    set userScalex 1.0
    set userScaley 1.0 

    redraw name func ops
  }

  return
}


proc ::timeline::every_res {} {

  variable usableMolLoaded
  variable rectCreated
  variable fit_scalex
  variable fit_scaley
  variable userScalex
  variable userScaley
  #this forces redraw, to cure any scaling floating point errors
  #that have crept in 
  set rectCreated 0

  variable scaley
  variable scalex

  if {$usableMolLoaded} {
    #redraw, set x and y  at once
    set scalex $fit_scalex 
    set userScalex 1.000 
    set scaley 1.0
    set userScaley [expr $scaley/$fit_scaley]
    redraw name func ops
  }
  
  return
}


proc ::timeline::residueCodeRedraw {} {

  variable w 
  variable resCodeShowOneLetter
  variable usableMolLoaded
  tlPutsDebug ": now in residueCodeRedraw, resiude_code_toggle is $resCodeShowOneLetter"
  
  if {$usableMolLoaded} {

    redraw name function op
  }
  return
}



proc ::timeline::initVars {} {

  variable dataFileVersion "1.3"
  variable usableMolLoaded 0
  variable windowShowing 0
  variable needsDataUpdate 0
  variable dataValNum -1
  variable dataValNumResSel -1
  variable eo 0
  variable x1 0 
  variable y1 0
  variable startCanvas ""
  variable startShiftPressed 0
  variable vmd_pick_shift_state 0
  variable resCodeShowOneLetter 0
  variable bond_rad 0.5
  variable bond_res 10
  variable repColoring "name"
  variable cursor_bond_rad 0.6 
  variable cursor_bond_res 10
  variable cursorRepColor 1
  variable cursorRep
  variable so ""
  variable marqueeButton -1
  #better if nullMolString is 'null', alter pop-up drawing to accomodate XXX
  variable nullMolString ""
  variable currentMol $nullMolString
  variable prevMol $nullMolString

  variable  userScalex 1
  variable  userScaley 1
  variable  userScaleBoth 1
  variable  scalex 1
  variable  scaley 1
  variable prevScalex 1
  variable prevScaley 1
  
  variable ytopmargin 5
  variable ybottommargin 10
  variable xrightmargin 8

  #variable xcanwindowStarting 780 
  variable xcanwindowStarting 685 
  variable ycanwindowStarting 574 

  
  variable numFrames 0
  variable xcanwindowmax  $xcanwindowStarting
  variable ycanwindowmax $ycanwindowStarting 
  variable xcanmax
  set xcanmax(data) 610
  set xcanmax(vert) 95
  set xcanmax(horz) $xcanmax(data)
  #make this sensible!
  variable ycanmax
  set ycanmax(data) 400
  set ycanmax(vert) $ycanmax(data) 
  set ycanmax(horz) 46 
  variable codes
  variable trajMin -100 
  variable trajMax 100 
  variable dataMin
  set dataMin(all) null
  variable dataMax
  set dataMax(all) null
  variable ONDist 3.2
  #distance cutoff in Angstroms
  variable hbondDistCutoff 3.0
  #angle cutoff in degrees
  variable  hbondAngleCutoff 20
  #set boolean false 
  variable usesFreeSelection 0
  variable firstAnalysisFrame 0
  variable lastAnalysisFrame 0
  variable anyResFuncDesc "count contacts"
  variable anyResFuncName "::myCountContacts"
  variable thresholdBoundMin 0
  variable thresholdBoundMax 0
  array set codes {ALA A ARG R ASN N ASP D ASX B CYS C GLN Q GLU E
    GLX Z GLY G HIS H ILE I LEU L LYS K MET M PHE F PRO P SER S
    THR T TRP W TYR Y VAL V}
   
  


  #tests if rects for current mol have been created (should extend 
  #so memorize all rectIds in 3dim array, and track num mols-long 
  #vector of rectCreated. Would hide rects for non-disped molec,
  #and remember to delete the data when molec deleted.
  
  variable rectCreated 0

  #the box height
  variable ybox 15.0
  #text skip doesn't need to be same as ybox (e.g. if bigger numbers than boxes in 1.0 scale)
  variable vertTextSkip $ybox

  
  # For vertical scale appearance
  variable vertHighLeft 2
  variable vertHighRight 100
  variable vertTextRight 96
  #The first 3 fields, 0 to 2 are printed all together, they are text
  variable xcol
  #set xcol(0) 10.0
  variable horzScaleHeight 30
  variable threshGraphHeight 40 
  variable vertScaleWidth 100
  variable dataWidth 85
  variable dataMargin 0
  variable xPosScaleVal 32
  #so rectangle of data is drawn at width $dataWidth - $dataMargin (horizontal measures)
  #
  #residuie name data is in umbered entires numbered less than 3
  variable dataOrigin 3
  #puts "dataOrigin is $dataOrigin"
  #column that multi-col data first  appears in

  #old setting from when vertscale and data were on same canvas
  #set xcol($dataOrigin)  96 
  set xcol($dataOrigin)  1 
  #The 4th field (field 3) is the "first data field"
  #we use same data structure for labels and data, but now draw in separate canvases 
  
  # the names for  three fields of data 
  
  #just for self-doc
  # dataVal(picked,n) set if the elem is picked
  # dataVal(pickedId,n) contains the canvas Id of the elem's highlight rectangle
  

  variable dataName

  set dataName(picked) "picked" 
  set dataName(pickedId) "pickedId"
  #not included in count of # datanames
  
  set dataName(0) "resid"
  set dataName(1) "resname"
  set dataName(1code) "res-code"
  set dataName(2) "chain"
  ###set dataName(3) "check error.." 
  
  
}


proc ::timeline::Show {} {
  variable windowShowing
  variable needsDataUpdate
  set windowShowing 1

  
  if {$needsDataUpdate} {
    set needsDataUpdate 0
    #set immmediately, so other binding callbacks will see
    [namespace current]::timeLineMain
  }

}

proc ::timeline::Hide {} {
  variable windowShowing 
  set windowShowing 0

}

proc ::timeline::createCursorHighlight { theSel} {
  tlPutsDebug ": in create CursorHighlight"

  variable currentMol
  variable cursor_bond_rad
  variable cursor_bond_res
  variable cursorRep
  variable cursorRepColor
  variable nullMolString
  tlPutsDebug "var block done, in create CursorHighlight"
 
  if {$currentMol == $nullMolString} {
     return
   }

  #draw first selection, as first residue 
  set cursorRep($currentMol) [molinfo $currentMol get numreps]
  mol selection $theSel
  mol material Opaque
  mol addrep $currentMol
  mol modstyle $cursorRep($currentMol)  $currentMol  Bonds $cursor_bond_rad $cursor_bond_res
  mol modcolor $cursorRep($currentMol) $currentMol ColorID $cursorRepColor
  tlPutsDebug "leaving create CursorHighlight"

}

proc ::timeline::createHighlight { theSel} {

  variable currentMol
  variable bond_rad
  variable bond_res
  variable rep
  variable repColoring
  variable cursorRep
  variable nullMolString
  #draw first selection, as first residue 
  tlPutsDebug "rep($currentMol)= $rep($currentMol)  currentMol= $currentMol"
  if {$currentMol == $nullMolString} {
     return
   }
  set rep($currentMol) [molinfo $currentMol get numreps]
  #next lines really needed here?
  mol selection $theSel
  mol material Opaque
  mol addrep $currentMol
  mol modstyle $rep($currentMol)  $currentMol  Bonds $bond_rad $bond_res
  mol modcolor $rep($currentMol) $currentMol $repColoring
}



proc ::timeline::draw_interface {} {
  variable w 

  variable eo 
  variable x1  
  variable y1 
  variable startCanvas
  variable startShiftPressed 
  variable vmd_pick_shift_state 
  variable resCodeShowOneLetter 
  variable bond_rad 
  variable bond_res
  variable so 
  variable xcanwindowmax 
  variable ycanwindowmax 
  variable xcanmax 
  variable ycanmax
  variable ybox 
  variable xsize 
  variable ysize 
  variable resnamelist 
  variable structlist 
  variable betalist 
  variable sel 
  variable canvasnew 
  variable userScalex
  variable userScaley
  variable scalex 
  variable scaley 
  variable dataValNum 
  variable dataVal 
  variable dataName 
  variable dataOrigin 
  variable ytopmargin 
  variable ybottommargin 
  variable vertTextSkip   
  variable xcolbond_rad 
  variable bond_res 
  variable bond_rad 
  variable rep 
  variable cursorRep
  variable repColoring
  variable cursor_bond_res
  variable cursor_bond_rad
  variable xcol 
  variable resCodeShowOneLetter 
  variable dataWidth 
  variable dataMargin 
  variable dataMin 
  variable dataMax 
  variable xPosScaleVal
  variable currentMol
  variable fit_scalex 1.0
  variable fit_scaley 1.0
  variable usableMolLoaded 
  variable numFrames 
  variable userScaleBoth

  frame $w.menubar -height 30 -relief raised -bd 2
  pack $w.menubar -in $w -side top -anchor nw -padx 1 -fill x

  #frame $w.fr -width 700 -height 810 -bg #FFFFFF -bd 2 ;#main frame

  #pack $w.fr

  label $w.txtlab -text "Zoom "
   tlPutsDebug " before selinfo label make"
  frame $w.panl -width 170 -height [expr $ycanwindowmax + 80] -bg #C0C0D0 -relief raised -bd 1 
  frame $w.cfr -width 350 -height [expr $ycanwindowmax + 85] -borderwidth 1  -bg #606060 -relief raised -bd 3
  tlPutsDebug " after frames"
  pack $w.panl -in $w -side left -padx 2  -fill y
  #pack $w.cfr -in $w.fr -side left -padx 2 -expand yes -fill both 
  pack $w.cfr -in $w -side left -padx 2 -expand yes -fill both 
   tlPutsDebug ": after selinfo label make"

  scale $w.panl.zoomlevel -from 0.01 -to 8.01 -length 150 -sliderlength 30  -resolution 0.01 -tickinterval 0.5 -repeatinterval 30 -showvalue true -variable [namespace current]::userScaley -command [namespace code userScaleyChanged] 
   label $w.selInfo -text "Resid Name Chain\nValue\nFrame" -width 16
  scale $w.zoomBothlevel -orient horizontal -from 0.001 -to 4.000 -length 120 -sliderlength 30  -resolution 0.001 -tickinterval 3.998 -repeatinterval 30 -showvalue true -variable [namespace current]::userScaleBoth -command [namespace code userScaleBothChanged] 
  scale $w.zoomXlevel -orient horizontal -from 0.001 -to 4.000 -length 120 -sliderlength 30  -resolution 0.001 -tickinterval 3.998 -repeatinterval 30 -showvalue true -variable [namespace current]::userScalex -command [namespace code userScalexChanged] 
  scale $w.threshMinScale -orient horizontal -from -180 -to 180 -length 100 -sliderlength 20  -resolution 0.01 -tickinterval 359.98  -repeatinterval 30 -showvalue true -variable [namespace current]::thresholdBoundMin -command [namespace code threshChanged] 
  scale $w.threshMaxScale -orient horizontal -from -180 -to 180.000 -length 100 -sliderlength 20  -resolution 0.01  -repeatinterval 30 -showvalue true -variable [namespace current]::thresholdBoundMax  -command [namespace code threshChanged] 
 tlPutsDebug " after scales"
  #pack $w.panl $w.cfr -in $w.fr -side left -padx 2
  pack $w.panl.zoomlevel -in $w.panl -side right -ipadx 5 -padx 3 -anchor e
 tlPutsDebug " after pack panlzomm"
  button $w.showall  -text "fit all" -command [namespace code {showall 0}]
  button $w.every_res  -text "every residue" -command [namespace code every_res]

 tlPutsDebug " after buttons" 
  #trace for molecule choosing popup menu 
  trace variable ::vmd_initialize_structure w  [namespace code molChooseMenu]
  
  menubutton $w.mol -relief raised -bd 2 -textvariable [namespace current]::currentMol -direction flush -menu $w.mol.menu
  menu $w.mol.menu -tearoff no


  molChooseMenu name function op
  

  label $w.molLab -text "Molecule:"

  scrollbar $w.ys -command [namespace code {canvasScrollY}]
  
  scrollbar $w.xs -orient horizontal -command [namespace code {canvasScrollX}]

 tlPutsDebug ": now to fill the top menu"

  #fill the  top menu
  menubutton $w.menubar.file -text "File" -underline 0 -menu $w.menubar.file.menu
  # XXX - set menubutton width to avoid truncation in OS X
  $w.menubar.file config -width 5
  menubutton $w.menubar.calculate -text "Calculate" -underline 0 -menu $w.menubar.calculate.menu
  # XXX - set menubutton width to avoid truncation in OS X
  $w.menubar.calculate config -width 10
  menubutton $w.menubar.threshold -text "Threshold" -underline 0 -menu $w.menubar.threshold.menu
  # XXX - set menubutton width to avoid truncation in OS X
  $w.menubar.threshold config -width 12
  menubutton $w.menubar.graphics -text "Appearance" -underline 0 -menu $w.menubar.graphics.menu
  # XXX - set menubutton width to avoid truncation in OS X
  $w.menubar.graphics config -width 11
  
  menubutton $w.menubar.analysis -text "Analysis" -underline 0 -menu $w.menubar.analysis.menu
  # XXX - set menubutton width to avoid truncation in OS X
  $w.menubar.analysis config -width 10

  menubutton $w.menubar.data -text "Data" -underline 0 -menu $w.menubar.data.menu
  # XXX - set menubutton width to avoid truncation in OS X
  $w.menubar.data config -width 5 


  pack $w.menubar.file  $w.menubar.calculate $w.menubar.threshold  $w.menubar.analysis $w.menubar.graphics  $w.menubar.data -side left

  menubutton $w.menubar.help -text Help -underline 0 -menu $w.menubar.help.menu
  menu $w.menubar.help.menu -tearoff no

  $w.menubar.help.menu add command -label "Timeline Help" -command "vmd_open_url [string trimright [vmdinfo www] /]/plugins/timeline"
  $w.menubar.help.menu add command -label "Structure codes..." -command  [namespace code {tk_messageBox -parent $w  -type ok -message "Secondary Structure Codes\n\nT        Turn\nE        Extended conformation\nB        Isolated bridge\nH        Alpha helix\nG        3-10 helix\nI         Pi-helix\nC        Coil (none of the above)\n" } ]

  pack $w.menubar.help -side right 
  
  #File menu
  menu $w.menubar.file.menu -tearoff no
  $w.menubar.file.menu add command -label "Print to file..." -command [namespace code {printCanvas} ] 
  $w.menubar.file.menu add command -label "Load data file..." -command [namespace code {readDataFile ""}  ] 

  $w.menubar.file.menu add command -label "Write data file..." -command [namespace code {writeDataFile ""}  ] 
  
  #Calculate menu
  
  menu $w.menubar.calculate.menu  -tearoff no

 tlPutsDebug ": about to register calc menus"

  $w.menubar.calculate.menu add command -label "Clear data"  -command  [namespace code clearData] 
  $w.menubar.calculate.menu add command -label "Calc. Sec. Struct"  -command [namespace code {calcDataStruct; showall 1}] 
  $w.menubar.calculate.menu add command -label "Calc. X position"  -command [namespace code {calcDataX; showall 1}] 
  $w.menubar.calculate.menu add command -label "Calc. Y position"  -command [namespace code {calcDataY; showall 1}] 
  $w.menubar.calculate.menu add command -label "Calc. Z position"  -command [namespace code {calcDataZ; showall 1}] 
  $w.menubar.calculate.menu add command -label "Calc. Phi"  -command [namespace code {calcDataPhi; showall 1}] 
  $w.menubar.calculate.menu add command -label "Calc. Delta Phi"  -command [namespace code {calcDataDeltaPhi; showall 1}] 
  $w.menubar.calculate.menu add command -label "Calc. Psi"  -command [namespace code {calcDataPsi; showall 1}] 
  $w.menubar.calculate.menu add command -label "Calc. Delta Psi"  -command [namespace code {calcDataDeltaPsi; showall 1}] 
  #$w.menubar.calculate.menu add command -label "Load User Data" -command  [namespace code {calcDataUser; showall 1}] 
  #$w.menubar.calculate.menu add command -label "Test Free Select" -command  [namespace code {calcTestFreeSel 10; showall 1}] 
  $w.menubar.calculate.menu add command -label "Calc. H-bonds" -command  [namespace code {calcTestHbonds 11; showall 1}] 
  $w.menubar.calculate.menu add command -label "Calc. Salt Bridges" -command  [namespace code {calcSaltBridge 16; showall 1}] 
  $w.menubar.calculate.menu add command -label "Calc. displacement" -command  [namespace code {calcDisplacement; showall 1}] 
  $w.menubar.calculate.menu add command -label "Calc. disp. velocity " -command  [namespace code {calcDispVelocity; showall 1}] 
  $w.menubar.calculate.menu add command -label "Calc. User-def. Per-Res Function" -command  [namespace code {calcDataAnyResFunc; showall 1}] 
 tlPutsDebug ": done with calc menus"
  #Threshold menu
  menu $w.menubar.threshold.menu -tearoff no
  $w.menubar.threshold.menu add command -label "Set bounds..." -command  [namespace code setThresholdBounds]
  $w.menubar.threshold.menu add command -label "Make threshold graph" -command  [namespace code thresholdMakeGraph]
  $w.menubar.threshold.menu add command -label "Reset graph" -command  [namespace code thresholdClearGraph]
  tlPutsDebug ": Timeline: starting graphics menus"
  
  #Graphics menu
  menu $w.menubar.graphics.menu -tearoff no
  $w.menubar.graphics.menu add command -label "Set scaling..." -command  [namespace code setScaling]
  $w.menubar.graphics.menu add checkbutton -label "Show 1-letter codes" -variable ::timeline::resCodeShowOneLetter -onvalue 1 -offvalue 0 -command  [namespace code residueCodeRedraw]
  $w.menubar.graphics.menu add cascade -label "Highlight color/style" -menu $w.menubar.graphics.menu.highlightMenu 
  #Second level menu for highlightColor 
tlPutsDebug ": Timeline: starting Highlight menu"
  set dummyHighlight 1 
  #set dummyHighlight so drawn selected first time, we use -command for actual var change
  menu $w.menubar.graphics.menu.highlightMenu -tearoff no
  $w.menubar.graphics.menu.highlightMenu add radiobutton -label "Yellow" -command {set highlightColor yellow} -variable dummyHighlight -value 0 
  $w.menubar.graphics.menu.highlightMenu add radiobutton -label "Purple" -command {set highlightColor purple} -variable dummyHighlight -value 1 

tlPutsDebug ": Timeline: starting Analysis menu"
  #Functions menu
  menu $w.menubar.analysis.menu -tearoff no
  $w.menubar.analysis.menu add command -label "Define every-residue function..." -command  [namespace code setAnyResFunc]
  $w.menubar.analysis.menu add command -label "Set analysis frame range..." -command  [namespace code setAnalysisFrames]
 
  #Data menu
  menu $w.menubar.data.menu -tearoff no
  $w.menubar.data.menu add command -label "Set collection directory..." -command  [namespace code loadDataCollection]
   
tlPutsDebug ": Timeline: done with startup of Analysis menu"

  
#the w.can object made here
#XXX should decide how to deal with variable y-size (freeSel's) and  even variable x-size (if ever abstract 2D plot so works with non-trajFrame values)
set ysize [expr $ytopmargin+ $ybottommargin + ($scaley *  $ybox * ($dataValNum + 1))]    
  set xsize [expr  $xcol($dataOrigin) +  ($scalex *  $dataWidth *  $numFrames ) ]




tlPutsDebug ": before some labels and similar"
  place $w.txtlab -in $w.panl  -bordermode outside -rely 0.1 -relx 0.5 -anchor n
  #place $w.showall -in $w.panl.zoomlevel  -bordermode outside -rely 1.0 -relx 0.5 -anchor n
  place $w.showall -in $w.panl.zoomlevel  -bordermode outside -rely 1.0 -y 10.0 -relx 0.5 -anchor n
  place $w.every_res  -in $w.showall -bordermode outside -rely 1.0 -y 3  -relx 0.5 -relwidth 1.6 -anchor n
  #place $w.every_res  -in $w.showall -bordermode outside -rely 1.0 -relx 0.5 -anchor n
  tlPutsDebug ": halfway"
  place $w.threshMinScale -in $w.every_res -bordermode outside -rely 1.0 -y 20 -relx 0.5 -width 90 -anchor n
  place $w.threshMaxScale -in $w.threshMinScale -bordermode outside -rely 1.0 -y 10 -relx 0.5 -width 90 -anchor n
  
  place $w.mol -in $w.panl.zoomlevel  -bordermode outside -rely -.1 -relx 0.85 -anchor s
  place $w.molLab -in $w.mol -bordermode outside -rely 0.5 -relx 0 -anchor e
  place  $w.zoomBothlevel -in $w.panl.zoomlevel    -bordermode  outside -rely 0.0 -y -55  -relx 0.5  -width 87 -anchor s 
  tlPutsDebug ":just before zoomXlevel"
  place $w.zoomXlevel -in $w.zoomBothlevel -bordermode inside -rely 0.0 -y -15 -relx 0.5  -width 87 -anchor s 
  place $w.selInfo -in $w.cfr -bordermode inside -rely 1.0 -y -10  -relx 0.0 -x 3 -anchor sw
  #done with interface elements     
  tlPutsDebug ":done with interface"

  #ask window manager for size of window

  #turn traces  on (initialize_struct trace comes later)
  #trace variable userScalex w  [namespace code redraw]
  #trace variable userScaley w  [namespace code redraw]
  trace variable ::vmd_pick_atom w [namespace code listPick]
  trace variable currentMol w [namespace code molChoose]

}
  proc  ::timeline::timeBarJumpPress {x y shiftState whichCanvas} {
    variable xcol
    variable dataOrigin
    variable dataWidth
    variable scalex
    variable currentMol
    variable numFrames
    variable xcanmax
    variable ycanmax
    variable w
    #maybe store top, and restore it afterwards
    variable nullMolString
    variable ytopmargin
    variable scaley
    variable ybox
    variable dataVal
    variable dataValNum
    variable cursorShown
    variable prevCursorObject
    variable prevCursorFrame
    tlPutsDebug "timeBarJumpPress starting"
    if {$currentMol == $nullMolString} {
       return
     }
    set x [expr $x + $xcanmax(data) * [lindex [$w.can xview] 0]]
    set y [expr $y + $ycanmax(data) * [lindex [$w.can yview] 0]] 

    set cursorFrame [expr  int (($x - $xcol($dataOrigin))/ ($dataWidth * $scalex))  ]

    if {$cursorFrame >= $numFrames}  {
      set cursorFrame [expr $numFrames -1]
    }
    
    if {$cursorFrame < 0 } { 
      set cursorFrame 0
    } 

    #allow scrubbing single sels
    set cursorObject [expr int (0.0 + ((0.0 + $y - $ytopmargin) / ($scaley * $ybox)) )]
    if {$cursorObject < 0} {
      set cursorObject 0
    }
    if {$cursorObject > $dataValNum} {
      set cursorObject $dataValNum 
    }
    if {$cursorObject < 0} {return}


  #  #these are only set when pressed, so we can tell if button-2 press/relase on same object+frame XXX
  #  # (a note for when we merge the silly 3 procs for timeBarJump into 1 proc) XXX
  #  # perhaps later replace this with 'object has changed' flag to dirty in button-2-moved, so press/away/back doesn't toggle
    #set prevCursorObject($currentMol)  $cursorObject ; set prevCursorFrame($currentMol) $cursorFrame
  #
    #only record the onbject, frame clicked on  for later reveal if cursorShown == 0
    if  {! $cursorShown($currentMol)} {return}
      
     
    #yipes, code duplication from letGoMarquee
    #Should reaclly de-sel everything, but need to keep track of separate sels, perhaps in the text vs. null later XXX
    if { [molinfo $currentMol get frame] != $cursorFrame } { 
      #test, and save/restore
      mol top $currentMol
      animate goto $cursorFrame

      #puts "jumped to $cursorFrame"
      drawTimeBar $cursorFrame
      #update both GL and tk timebar
      display update
      #display update ui
      #puts "time for disp. update = [time {display update}]"
      #puts "time for disp. update ui= [time {display update ui}]"
    }
    drawCursorObjectBar $cursorObject $cursorFrame
  }

  proc  ::timeline::timeBarJumpRelease {x y shiftState whichCanvas} {
    #Yikes, entire proc (nearly) is code duped from timeBarJumpPress XXXX
    #merge these!
    variable xcol
    variable dataOrigin
    variable dataWidth
    variable scalex
    variable currentMol
    variable numFrames
    variable xcanmax
    variable ycanmax
    variable w
    #maybe store top, and restore it afterwards
    variable nullMolString
    variable ytopmargin
    variable scaley
    variable ybox
    variable dataVal
    variable dataValNum
    variable prevCursorObject
    variable prevCursorFrame
    variable cursorShown
    if {($currentMol == $nullMolString) || ($dataValNum < 1)} { return }

    set x [expr $x + $xcanmax(data) * [lindex [$w.can xview] 0]]
    set y [expr $y + $ycanmax(data) * [lindex [$w.can yview] 0]] 

    set cursorFrame [expr  int (($x - $xcol($dataOrigin))/ ($dataWidth * $scalex))  ]

    if {$cursorFrame >= $numFrames}  {
      set cursorFrame [expr $numFrames -1]
    }
    
    if {$cursorFrame < 0 } { 
      set cursorFrame 0
    } 

    #allow scrubbing single sels
    set cursorObject [expr int (0.0 + ((0.0 + $y - $ytopmargin) / ($scaley * $ybox)) )]
    if {$cursorObject < 0} {
      set cursorObject 0
    }
    if {$cursorObject > $dataValNum} {
      set cursorObject $dataValNum 
    }
    
    set exitFlag 0
    if { ($prevCursorObject($currentMol) == $cursorObject) &&  ($prevCursorFrame($currentMol) == $cursorFrame)} {
        puts "cursorShown($currentMol)=$cursorShown($currentMol)    prevCursorObject($currentMol)=$prevCursorObject($currentMol) cursorObject= $cursorObject   prevCursorFrame($currentMol)=prevCursorFrame($currentMol) cursorFrame= $cursorFrame"
        #toggle state
        if {$cursorShown($currentMol)} {
          #puts "cursor is shown, so now  hiding"
          hideCursorHighlight  $currentMol 
          $w.horzScale delete cursorObjectBarRect 
          $w.vertScale delete cursorObjectBarRect 
          $w.can delete cursorObjectBarRect 
          $w.can delete trajHighlight
          $w.can delete timeBarRect
          set cursorShown($currentMol) 0
          #for good measure, delete all the sequence selections too (seems like right place for this)
          clearAllPicked
          set exitFlag 1 

        } else {
          #puts "cursor is hidden, so now revealing"
          revealCursorHighlight $currentMol 
          set cursorShown($currentMol) 1
          #and code below will redraw all rects in proper place
        }
    }
    set prevCursorObject($currentMol)  $cursorObject ; set prevCursorFrame($currentMol) $cursorFrame

    if {$exitFlag} {return}

    #yipes, code duplication from letGoMarquee
    #Should really de-sel everything, but need to keep track of separate sels, perhaps in the text vs. null later XXX
    if { [molinfo $currentMol get frame] != $cursorFrame } { 
      #test, and save/restore
      mol top $currentMol
      animate goto $cursorFrame

      #puts "jumped to $cursorFrame"
      drawTimeBar $cursorFrame
      #update both GL and tk timebar
      display update
      display update ui
      #puts "time for disp. update = [time {display update}]"
      #puts "time for disp. update ui= [time {display update ui}]"
    }
    drawCursorObjectBar $cursorObject $cursorFrame
  }



  proc  ::timeline::timeBarJump {x y shiftState whichCanvas} {
    variable xcol
    variable dataOrigin
    variable dataWidth
    variable scalex
    variable currentMol
    variable numFrames
    variable xcanmax
    variable ycanmax
    variable w
    #maybe store top, and restore it afterwards
    variable nullMolString
    variable ytopmargin
    variable scaley
    variable ybox
    variable dataVal
    variable dataValNum
    variable cursorShown
    if {($currentMol == $nullMolString) || ($dataValNum < 1)} { return }
    
    #merge these!
    set x [expr $x + $xcanmax(data) * [lindex [$w.can xview] 0]]
    set y [expr $y + $ycanmax(data) * [lindex [$w.can yview] 0]] 

    set cursorFrame [expr  int (($x - $xcol($dataOrigin))/ ($dataWidth * $scalex))  ]

    if {$cursorFrame >= $numFrames}  {
      set cursorFrame [expr $numFrames -1]
    }
    
    if {$cursorFrame < 0 } { 
      set cursorFrame 0
    } 

    #allow scrubbing single sels
    set cursorObject [expr int (0.0 + ((0.0 + $y - $ytopmargin) / ($scaley * $ybox)) )]
    if {$cursorObject < 0} {
      set cursorObject 0
    }
    if {$cursorObject > $dataValNum} {
      set cursorObject $dataValNum 
    }
    if {! $cursorShown($currentMol) } {
      revealCursorHighlight $currentMol 
      set cursorShown($currentMol) 1
    }
    #yipes, code duplication from letGoMarquee
    #Should really de-sel everything, but need to keep track of separate sels, perhaps in the text vs. null later XXX
    if { [molinfo $currentMol get frame] != $cursorFrame } { 
      #test, and save/restore
      mol top $currentMol
      animate goto $cursorFrame

      #puts "jumped to $cursorFrame"
      drawTimeBar $cursorFrame
      #update both GL and tk timebar
      display update
      #puts "time for disp. update = [time {display update}]"
      #puts "time for disp. update ui= [time {display update ui}]"
    }
    drawCursorObjectBar $cursorObject $cursorFrame
    display update ui
  }

   

  proc  ::timeline::drawTimeBar {f} {
    variable w
    variable dataWidth
    variable scalex
    variable xcol 
    variable dataOrigin
    variable ycanmax

    #puts "showing frame $f"
    set xTimeBarStart  [expr  ( ($f + 0.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    set xTimeBarEnd  [expr  ( ($f + 1.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    #puts "xTimeBarStart= $xTimeBarStart  xTimeBarEnd = $xTimeBarEnd"
    #more efficient to re-configure x1 x2
    $w.can delete timeBarRect
    #set timeBar [$w.can create rectangle  $xTimeBarStart 1 $xTimeBarEnd [expr $ycanmax(data) ]   -fill "\#000000" -stipple gray50 -outline "" -tags [list dataScalable timeBarRect ] ]
    set timeBar [$w.can create rectangle  $xTimeBarStart 1 $xTimeBarEnd [expr $ycanmax(data) ]   -fill "" -outline "\#000000"  -tags [list dataScalable timeBarRect ] ]

    #move the time line 
  } 
  proc  ::timeline::drawCursorObjectBar {obj f} {
    variable w
    variable dataWidth
    variable scalex
    variable scaley
    variable ytopmargin 
    variable ybox
    variable xcol 
    variable dataOrigin
    variable ycanmax
    variable vertScaleWidth
    variable horzScaleHeight
    variable dataVal
    variable currentMol
    variable dataName
    variable usesFreeSelection
    #puts "showing obj o in frame $f"
    #duplicated code from draw
    set xTimeBarStart  [expr  ( ($f + 0.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    set xTimeBarEnd  [expr  ( ($f + 1.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    #puts "xTimeBarStart= $xTimeBarStart  xTimeBarEnd = $xTimeBarEnd"
    #more efficient to re-configure x1 x2
   set ypos [expr $ytopmargin + ($scaley * $ybox * int ($obj))]

    $w.can delete cursorObjectBarRect
    $w.vertScale delete cursorObjectBarRect
    $w.horzScale delete cursorObjectBarRect
    #new position of cursorObject could scale width of outline... XXX
   
    if $usesFreeSelection {
      set cursorSelText $dataVal(freeSelString,$obj)
    } else {
      set cursorSelText "resid $dataVal(0,$obj) and  chain $dataVal(2,$obj)" 
    } 
    showCursorHighlight $cursorSelText
    set cursorObjectBar [$w.can create rectangle  $xTimeBarStart $ypos $xTimeBarEnd [expr $ypos+ ($scaley * $ybox)]    -outline "\#FF0000" -width 2 -fill "" -tags [list dataScalable cursorObjectBarRect ] ] 
    set cursorObjectBarInVertScale [$w.vertScale create rectangle  1 $ypos $vertScaleWidth [expr $ypos+ ($scaley * $ybox)]    -outline "\#FF0000" -width 2 -fill "" -tags [list yScalable cursorObjectBarRect ] ] 
    set cursorObjectBarInHorzScale [$w.horzScale create rectangle  $xTimeBarStart 1 $xTimeBarEnd $horzScaleHeight    -outline "\#FF0000" -width 2 -fill "" -tags [list xScalable cursorObjectBarRect ] ] 

   configureSelInfo $obj $f
}

proc ::timeline::configureSelInfo {obj f} {
  variable w
  variable dataWidth
  variable scalex
  variable scaley
  variable ytopmargin 
  variable ybox
  variable xcol 
  variable dataOrigin
  variable ycanmax
  variable vertScaleWidth
  variable horzScaleHeight
  variable dataVal
  variable currentMol
  variable dataName
  variable usesFreeSelection

  #parameter f is frame, for cursor movement
  if {$obj=="null"} {
     $w.selInfo configure -text " $dataName($dataOrigin) \n\n\n\n"
     return
  }
      
  if $usesFreeSelection {
       
         $w.selInfo configure -text "$dataName($dataOrigin)\n\n$dataVal(freeSelLabel,$obj)\n$dataVal([expr $dataOrigin+$f],$obj)\nFrame [format "%5g" $f]"
  } else {
       if {$dataName([expr $dataOrigin+$f])=="struct"} {
         $w.selInfo configure -text "sec. structure\n\n$dataVal(0,$obj) $dataVal(1,$obj) $dataVal(2,$obj) \n$dataVal([expr $dataOrigin + $f],$obj)\nFrame [format "%5g" $f]"
       } else {
         $w.selInfo configure -text "$dataName($dataOrigin)\n\n$dataVal(0,$obj) $dataVal(1,$obj) $dataVal(2,$obj) \n[format "%8.4g" $dataVal([expr $dataOrigin + $f],$obj)]\nFrame [format "%5g" $f]"
      }
  }
} 

# fix hanging vim syntax quote""



proc ::timeline::writeDataFile {filename} {

    variable w
    variable dataName
    variable dataVal
    variable dataMin
    variable dataMax
    variable dataValNum
    variable currentMol
    variable numFrames
    variable dataOrigin
    variable usesFreeSelection
    variable dataFileVersion
    if {$filename == ""  } {
      set filename [tk_getSaveFile -initialfile $filename -title "Save Trajectory Data file" -parent $w -filetypes [list { {.tml files} {.tml} } { {Text files} {.txt}} {{All files} {*} }] ]
    }
    if {$filename == ""  } {return}
    
    set writeDataFile [open $filename w]
    puts $writeDataFile "# VMD Timeline data file"
    puts $writeDataFile "# CREATOR= $::tcl_platform(user)"
    puts $writeDataFile "# MOL_NAME= [molinfo $currentMol get name]"
    puts $writeDataFile "# DATA_TITLE= $dataName($dataOrigin)"
    puts $writeDataFile "# FILE_VERSION= $dataFileVersion"
    puts $writeDataFile "# NUM_FRAMES= $numFrames "
    puts $writeDataFile "# NUM_ITEMS= [expr $dataValNum + 1]"

    if {$usesFreeSelection} {
      puts $writeDataFile "# FREE_SELECTION= 1"
      puts $writeDataFile "#"
      for {set i 0} {$i<=$dataValNum} {incr i} {
        puts $writeDataFile "freeSelLabel $i $dataVal(freeSelLabel,$i)"
        puts $writeDataFile "freeSelString $i $dataVal(freeSelString,$i)"
          # calc min/max on read?
          #puts $writeDataFile "set dataMin($curField) $dataMin($curField)" 
          #puts $writeDataFile "set dataMax($curField) $dataMax($curField)" 
        for {set trajFrame 0} {$trajFrame < $numFrames} {incr  trajFrame} {
          set curField [expr $dataOrigin + $trajFrame]
          puts $writeDataFile "$trajFrame $i $dataVal($curField,$i)" 
        }
      }
    } else {
      puts $writeDataFile "# FREE_SELECTION= 0"
      puts $writeDataFile "#"
      set endStructs [expr $dataOrigin+ ($numFrames - 1)]
      for {set field $dataOrigin} {$field <= $endStructs} {incr field} {
        set frame [expr $field - $dataOrigin]
        for {set i 0} {$i<=$dataValNum} {incr i} {
          set val $dataVal($field,$i)
          set resid $dataVal(0,$i)
          set chain $dataVal(2,$i)
          #we are proceeding through frames here for timeline
          #this loop is over already-known frame info
          #looks backwards since inherited approach from multicolumn
          # the CA is placeholder for atom (backwards compatible?)
          puts $writeDataFile "$resid $chain CA $frame $val"
        }
      }
    }
    close $writeDataFile
    return
  }

  proc ::timeline::calcDataStruct {} {
    variable w
    variable dataName
    variable dataVal
    variable dataValNum
    variable dataValNumResSel
    variable dataOrigin
    variable currentMol
    variable firstTrajField
    variable numFrames
    variable dataMin
    variable dataMax
    variable lastCalc
    variable nullMolString
    variable usesFreeSelection
    set usesFreeSelection 0
 
    if {$currentMol == $nullMolString} {
        return 
    }
    set curDataName $dataOrigin 
    set dataValNum $dataValNumResSel
    set lastCalc 1
    set sel [atomselect $currentMol "(all and name CA) or ( (not protein) and  ( (name \"C3\\'\") or (name \"C3\\*\") ) ) "] 
    for {set trajFrame 0} {$trajFrame < $numFrames} {incr  trajFrame} {
      
      animate goto $trajFrame 
      display update ui
      $sel frame $trajFrame
      #puts "set frame to $trajFrame"
      
      #puts "now update for mol $currentMol"
      mol ssrecalc $currentMol
      #puts "updated"
      set structlist [$sel get structure]
      #puts "setting dataName([$dataOrigin+$trajFrame]) to struct..."
      set dataName([expr $dataOrigin+$trajFrame]) "struct"

      set i 0
      foreach elem $structlist {
        set dataVal([expr $dataOrigin+$trajFrame],$i) $elem
        incr i
        
      }
      
      unset structlist; #done with it

    }

    #if just setting one set of data for every frame, 
    #should clear unused dataVal()'s, etc. here.
    configureSelInfo null 0
    
    return
  }

  proc ::timeline::checkRangeLimits {elem} {
    #tlPutsDebug "checking range limits for $elem"
    variable dataMin
    variable dataMax
    variable trajMin
    variable trajMax
    if {$dataMin(all) == "null"} then { 
      set dataMin(all) $elem
    } else {
      #XX should use this code below for user-defd(not as dup)
      # faster to do sort?
      if {$elem < $dataMin(all)}  {
        set dataMin(all) $elem
      } 
    }
    if {$dataMax(all) == "null"} then {
      set dataMax(all) $elem
    } else {
      if {$elem > $dataMax(all)} {
        set dataMax(all) $elem
      }  
    }   
  }

  proc ::timeline::calcDataProperty {propertyString lastCalcVal} {
    tlPutsDebug ": in calcDataProperty"
    variable w
    variable dataName
    variable dataVal
    variable dataValNum
    variable dataValNumResSel
    variable currentMol
    variable firstTrajField
    variable numFrames
    variable dataMin
    variable dataMax
    variable trajMin
    variable trajMax 
    variable lastCalc
    variable dataOrigin
    variable nullMolString
    variable usesFreeSelection
    variable anyResFuncName
    variable firstAnalysisFrame
    variable  lastAnalysisFrame
    
    clearData

    #anyResFuncName is namespace-path name of a use -defined function (procedure that returns one value )(so is a function), to be is applied to all residues
    set firstFrame 0
    set lastFrame [expr $numFrames -1]
    #this will be externally settable later

    set usesFreeSelection 0
    set dataValNum $dataValNumResSel

    if {$currentMol == $nullMolString} {
        #should really gray out choices unless molec is seleted XXX
        puts "Timeline: select molecule before choosing Calculate method"
        return 
    }
    set lastCalc $lastCalcVal 
      #XXX next line fragile in use, since relies on label/sel info (residue, chain, etc.) to have gotten this info in same order in different call .  Not sure this is guaranteed.
      set sel [atomselect $currentMol "(all and name CA) or ( (not protein) and  ( (name \"C3\\'\") or (name \"C3\\*\") ) ) "] 


    if {$propertyString != "anyResFunc"} {    
      tlPutsDebug ": VMD Timeline--CalcDataProperty starting for simple property (NOT anyResFunc)"
      tlPutsDebug ": Timeline: propertyString= $propertyString"
      for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
        #animate goto $trajFrame
        #display update ui
        set curField [expr $dataOrigin + $trajFrame]
        $sel frame $trajFrame
        #puts "set frame to $trajFrame"
        #this is quick check, not real way to do it 
        #that it doesn't use 'same selelction' 
        # currently just depends on 'sel get' order being same
        #method as other stuff, should really get data and sort it
        #
        set trajList [$sel get $propertyString]
        #tlPutsDebug "propertyString= $propertyString; trajList= $trajList" 
        #does position detection use next line -- if not delete?
        set dataName($curField) "${propertyString}-val"
        set i 0
        foreach elem $trajList {
          set dataVal($curField,$i) $elem
          checkRangeLimits $elem
          #XX should-use-below code ends
          incr i
        }
       $w.threshMinScale configure -from $dataMin(all) -to $dataMax(all) -tickinterval [expr  $dataMax(all) - $dataMin(all)-.01]
       $w.threshMinScale configure -from $dataMin(all) -to $dataMax(all) -tickinterval [expr  $dataMax(all) - $dataMin(all)-.01]
        tlPutsDebug "setting dataMin in $curField to $trajMin (appearance scaling min)"
        #set dataMin($curField) $dataMin(all)
        #set dataMax($curField) $dataMax(all)
        set dataMin($curField) $trajMin 
        set dataMax($curField) $trajMax 
      } 
    } else {
    tlPutsDebug ": VMD Timeline--CalcDataProperty starting anyResFunc calc"
    #now for anyResFunc calc
      #should do syntax check, time check on at least one residue...
      set dataMin(all) 0
      set dataMax(all) 0
      set resAtomIndexList [$sel get index]
      # the counter i tracks residue in dataVal(curField, i)
      set i 0
      foreach resAtomIndex $resAtomIndexList {
        tlPutsDebug ": VMD Timeline: starting res-row $i, index= $resAtomIndex"
        set resAtomSel [atomselect $currentMol "index $resAtomIndex"]
        set resCompleteSel [atomselect $currentMol "same residue as index $resAtomIndex"]
        set proteinNucSel [atomselect $currentMol "protein or nucleic"]
        # XXX how to speed up as array?
        #now the user function can choose to use either a core atom (resAtomSel) or all atoms in residue (resCompleteSel) with neither penalized for selection time
        for {set trajFrame $firstAnalysisFrame} {$trajFrame <= $lastAnalysisFrame} {incr  trajFrame} {
        $resAtomSel frame $trajFrame
        $resCompleteSel frame $trajFrame
        $proteinNucSel frame $trajFrame
           #set frame for both sel options
           #XXX if either of above two is time consuming, use user-set switches to choose which of them gets frame set
        set curField [expr $dataOrigin + $trajFrame]
        #now run proc, in the current context 
        if {
          [catch { 
             #run the proc
             #value would be set for trajFrame and resAtomIndex
             #tlPutsDebug "about to run  user-defined proc"
             set val [$anyResFuncName $resAtomSel  $resCompleteSel  $proteinNucSel ]


             #tlPutsDebug ": user-defined proc has run"
            # XX replace with above code for built-ins?
                    
             set dataVal($curField,$i) $val
                
             checkRangeLimits $val
            
           # replace with above code for built-ins
               
             #tlPutsDebug ": frame= $trajFrame   atom index= $resAtomIndex i= $i  dataVal($curField,$i) = $dataVal($curField,$i)  dataMin(all)= $dataMin(all) dataMax(all)= $dataMax(all)"

            #XXX replace next with loop-thru at start end.  wasted setting mostly
            #XXX much wasted time here.  REPLACE with loop later.  Also, after first calc of this, should be min/max.  Second calc in a row should not happen, just use o;d values.
            set dataName($curField) "${anyResFuncName}"
            set dataMin($curField) $trajMin 
            set dataMax($curField) $trajMax 
            #tlPutsDebug "assigned dataName, dataMind, dataMax for user def'd func" 
            }
          ] 
       } then {
          #complain about error
          puts "ERROR VMD::Timeline: User-defined residue procedure >${anyResFuncName}<"
          puts "ERROR VMD::Timeline: had error"
          puts "ERROR VMD::Timeline: for molecule $currentMol, atom index $resAtomIndex"
          #tlPutsDebug ": frame= $trajFrame   atom index= $resAtomIndex i= $i  dataVal($curField,$i) = $dataVal($curField,$i)  dataMin(all)= $dataMin(all) dataMax(all)= $dataMax(all)"

        }  
      }
      incr i  
    }
  }
  $w.threshMinScale configure -from $dataMin(all) -to $dataMax(all)
  $w.threshMaxScale configure -from $dataMin(all) -to $dataMax(all)
  configureSelInfo null 0
  return 
} 

#XXX currently, must manually match switch statemnt.  Make #these auto-register...
proc ::timeline::calcDataAnyResFunc {} {
  calcDataProperty "anyResFunc" 13
}

proc ::timeline::calcDataX {} {
  tlPutsDebug ": VMD Timeline: in calcDataX"
  calcDataProperty "x" 2
  tlPutsDebug ": VMD Timeline: leaving calcDataX"
}
proc ::timeline::calcDataY {} {
  calcDataProperty "y" 3
}
proc ::timeline::calcDataZ {} {
  calcDataProperty "z" 4
}

proc ::timeline::calcDataPhi {} {
  calcDataProperty "phi" 5
}

proc ::timeline::calcDataDeltaPhi {} {
  calcDataDeltaProperty "phi" 6
}


proc ::timeline::calcDataPsi {} {
  calcDataProperty "psi" 7
}

proc ::timeline::calcDataDeltaPsi {} {
  calcDataDeltaProperty "phi" 8
}

proc ::timeline::test1 {} {tlPutsDebug ": Timeline: This is test routine 1."}

proc ::timeline::calcDataUser {} {
  calcDataProperty "user" 12 
}
proc ::timeline::calcDisplacement {} {
  calcDisplacementProperty  14 
}
proc ::timeline::calcDispVelocity {} {
  calcDispVelocityProperty  15 
}

proc ::timeline::calcDispVelocityProperty {lastCalcVal} {
  variable w
  variable dataName
  variable dataVal
  variable dataValNum
  variable dataValNumResSel
  variable currentMol
  variable firstTrajField
  variable numFrames
  variable dataMin
  variable dataMax
  variable trajMin
  variable trajMax
  variable lastCalc
  variable dataOrigin
  variable nullMolString
  variable usesFreeSelection
  
  set usesFreeSelection 0

  set dataValNum $dataValNumResSel

  if {$currentMol == $nullMolString} {
      #should really gray out choices unless molec is seleted XXX
      puts "WARNING: VMD Timeline: select molecule before choosing Calculate method"
      return 
  }
  set lastCalc $lastCalcVal
  set curDataName $dataOrigin
  # wrong-- set initialFrameDataName [expr $curDataName+1]
  set sel [atomselect $currentMol "(all and name CA) or ( (not protein) and  ( (name \"C3\\'\") or (name \"C3\\*\") ) ) "] 

  for {set trajFrame 0} {$trajFrame < $numFrames} {incr  trajFrame} {

    #animate goto $trajFrame
    #display update ui
    set curField [expr $dataOrigin + $trajFrame]
    $sel frame $trajFrame
    #puts "set frame to $trajFrame"

    #this is quick check, not real way to do it
    #that it doesn't use 'same selelction'
    # currently just depends on 'sel get' order being same
    #method as other stuff, should really get data and sort it
    #
    set trajList [$sel get {x y z}] 
    #does position detection use next line -- if not, delete?
    set dataName($curField) "dispVelocity-val"
    set i 0
    foreach elem $trajList {
      if {$trajFrame == 0} {
          tlPutsDebug ": trajFrame= $trajFrame, curField=$curField, i=$i"
          set dataVal($curField,$i) 0
      } else {
           #XX only does central atom, change to COM/COG of residue (via index)
          set dataVal($curField,$i) [veclength [vecsub $elem  $dataVal(referenceVal,$i)] ]
      }
      set dataVal(referenceVal,$i) $elem 
       
      #if {$curField <=4} { 
      #  tlPutsDebug ": curField is $curField; dataVal($curDataName,$i)= $dataVal($curDataName,$i) "
      #}
      incr i
      #XXX this per-column min/max setting not needed, should set more widely
      set dataMin($curDataName) $trajMin 
      set dataMax($curDataName) $trajMax
    }
  }
  set dataMin(all) 0
  set dataMax(all) 10
  $w.threshMinScale configure -from $dataMin(all) -to $dataMax(all)
  $w.threshMaxScale configure -from $dataMin(all) -to $dataMax(all)
  configureSelInfo null 0
}

proc ::timeline::calcDisplacementProperty {lastCalcVal} {
  variable w
  variable dataName
  variable dataVal
  variable dataValNum
  variable dataValNumResSel
  variable currentMol
  variable firstTrajField
  variable numFrames
  variable dataMin
  variable dataMax
  variable trajMin
  variable trajMax
  variable lastCalc
  variable dataOrigin
  variable nullMolString
  variable usesFreeSelection
  
  set usesFreeSelection 0

  set dataValNum $dataValNumResSel

  if {$currentMol == $nullMolString} {
      #should really gray out choices unless molec is seleted XXX
      puts "WARNING: VMD Timeline: select molecule before choosing Calculate method"
      return 
  }
  set lastCalc $lastCalcVal
  set curDataName $dataOrigin
  # wrong-- set initialFrameDataName [expr $curDataName+1]
  set sel [atomselect $currentMol "(all and name CA) or ( (not protein) and  ( (name \"C3\\'\") or (name \"C3\\*\") ) ) "] 

  for {set trajFrame 0} {$trajFrame < $numFrames} {incr  trajFrame} {

    #animate goto $trajFrame
    #display update ui
    set curField [expr $dataOrigin + $trajFrame]
    $sel frame $trajFrame
    #puts "set frame to $trajFrame"

    #this is quick check, not real way to do it
    #that it doesn't use 'same selelction'
    # currently just depends on 'sel get' order being same
    #method as other stuff, should really get data and sort it
    #
    set trajList [$sel get {x y z}] 
    #does position detection use next line -- if not, delete?
    set dataName($curField) "displacement-val"
    set i 0
    foreach elem $trajList {
      if {$trajFrame == 0} {
          tlPutsDebug ": trajFrame= $trajFrame, curField=$curField, i=$i"
          set dataVal($curField,$i) 0
          set dataVal(referenceVal,$i) $elem 
      } else {
           #XX only does central atom, change to COM/COG of residue (via index)
          set dataVal($curField,$i) [veclength [vecsub $elem  $dataVal(referenceVal,$i)] ]
        checkRangeLimits $dataVal($curField,$i) 
      }
       
      #if {$curField <=4} { 
      #  tlPutsDebug ": curField is $curField; dataVal($curDataName,$i)= $dataVal($curDataName,$i) "
      #}
      incr i
      #XXX this per-column min/max setting not needed, should set more widely
      set dataMin($curDataName) $trajMin 
      set dataMax($curDataName) $trajMax
    }
  }

  $w.threshMinScale configure -from $dataMin(all) -to $dataMax(all)
  $w.threshMaxScale configure -from $dataMin(all) -to $dataMax(all)
  configureSelInfo null 0

}

proc ::timeline::calcDataDeltaProperty {propertyString lastCalcVal} {
  variable w
  variable dataName
  variable dataVal
  variable dataValNum
  variable dataValNumResSel
  variable currentMol
  variable firstTrajField
  variable numFrames
  variable dataMin
  variable dataMax
  variable trajMin
  variable trajMax
  variable lastCalc
  variable dataOrigin
  variable nullMolString
  variable usesFreeSelection
  
  set usesFreeSelection 0

  set dataValNum $dataValNumResSel

  if {$currentMol == $nullMolString} {
      #should really gray out choices unless molec is seleted XXX
      puts "WARNING: VMD Timeline: select molecule before choosing Calculate method"
      return 
  }
  set lastCalc $lastCalcVal
  set curDataName $dataOrigin
  # wrong-- set initialFrameDataName [expr $curDataName+1]
  set sel [atomselect $currentMol "(all and name CA) or ( (not protein) and  ( (name \"C3\\'\") or (name \"C3\\*\") ) ) "] 

  for {set trajFrame 0} {$trajFrame < $numFrames} {incr  trajFrame} {

    #animate goto $trajFrame
    #display update ui
    set curField [expr $dataOrigin + $trajFrame]
    $sel frame $trajFrame
    #puts "set frame to $trajFrame"

    #this is quick check, not real way to do it
    #that it doesn't use 'same selelction'
    # currently just depends on 'sel get' order being same
    #method as other stuff, should really get data and sort it
    #
    set trajList [$sel get $propertyString] 
    #does position detection use next line -- if not, delete?
    set dataName($curField) "delta-${propertyString}-val"
    set i 0
    foreach elem $trajList {
      if {$trajFrame == 0} {
          tlPutsDebug ": trajFrame= $trajFrame, curField=$curField, i=$i"
          set dataVal($curField,$i) 0
          set dataVal(referenceVal,$i) $elem 
      } else {
          set dataVal($curField,$i) [expr $elem - $dataVal(referenceVal,$i)]
      }
       
      #if {$curField <=4} { 
      #  tlPutsDebug ": curField is $curField; dataVal($curDataName,$i)= $dataVal($curDataName,$i) "
      #}
      incr i
      #XXX this per-column min/max setting not needed, should set more widely
      set dataMin($curDataName) $trajMin 
      set dataMax($curDataName) $trajMax
    }
  }
}


proc ::timeline::calcTestFreeSel {lastCalcVal} {
  variable w
  variable dataName
  variable dataVal
  variable dataValNum
  variable currentMol
  variable firstTrajField
  variable numFrames
  variable dataMin
  variable dataMax
  variable trajMin
  variable trajMax 
  variable lastCalc
  variable dataOrigin
  variable nullMolString
  variable usesFreeSelection
  set usesFreeSelection 1

  set lastCalc $lastCalcVal

  if {$currentMol == $nullMolString} {
      #should really gray out choices unless molec is seleted XXX
      puts "Timeline: select molecule before choosing Calculate method"
      return 
  }


  for {set trajFrame 0} {$trajFrame < $numFrames} {incr  trajFrame} {
    #clear out all frames, in real case set all data.
    for {set displayGroup 0} {$displayGroup<7} {incr displayGroup} {
      set curField [expr $dataOrigin + $trajFrame]
      set dataVal($curField,$displayGroup) 0 
      set dataMin($curField) $trajMin 
      set dataMax($curField) $trajMax 
    }
  }
    #use next line if really extracting data from traj geom.
    #$sel frame $trajFrame
    set dataName($curField) "free-sel. test"
    #manual test set
    set dataVal(freeSelLabel,0) "res 23 / res 28" 
    set dataVal(freeSelString,0) "resid 23 28"
    set dataVal([expr $dataOrigin+5],0) 100
    set dataVal([expr $dataOrigin+23],0) -100
    set dataVal([expr $dataOrigin+24],0) -100
    set dataVal([expr $dataOrigin+25],0) -100
    set dataVal([expr $dataOrigin+26],0) -100
    set dataVal(freeSelLabel,1) "res 1-5 / res 70-80"
    set dataVal(freeSelString,1) "resid 1 to 5 70 to 80"
    set dataVal([expr $dataOrigin+2],1) -100
    set dataVal([expr $dataOrigin+3],1) -100
    set dataVal([expr $dataOrigin+4],1)  70 
    set dataVal([expr $dataOrigin+5],1)  90 
    set dataVal([expr $dataOrigin+6],1)  110 
    set dataVal([expr $dataOrigin+17],1) -50 
    set dataVal([expr $dataOrigin+50],1) 150 
    set dataVal(freeSelString,2) "resid 60 61 62"
    set dataVal(freeSelLabel,2) "resid 60 61 62"
    set dataVal($dataOrigin+27,2) -50 
    set dataVal($dataOrigin+40,2) 150 
    set dataVal($dataOrigin+41,2) 150 
    set dataVal($dataOrigin+42,2) 150 
    set dataVal($dataOrigin+43,2) 150 
    set dataVal(freeSelString,3)  "resid 70 to 75"
    set dataVal(freeSelLabel,3) "resid 70 to 75"
    set dataVal($dataOrigin+17,3) -10 
    set dataVal($dataOrigin+80,3) 100 
    set dataVal($dataOrigin+45,3) 70
    set dataVal($dataOrigin+46,3) 80
    set dataVal(freeSelString,4) "resid 20 to 25 28 67"
    set dataVal(freeSelLabel,4) "some in 20's and 60's"
    set dataVal($dataOrigin+27,5) -40 
    set dataVal(freeSelString,5) "resid 50 51 52 61 62"
    set dataVal(freeSelLabel,5) "favorites"
    set dataVal($dataOrigin+27,5) -40 
    set dataVal($dataOrigin+38,5) 150 
    set dataVal($dataOrigin+39,5) 150 
    set dataVal($dataOrigin+40,5) 150 
    set dataVal(freeSelLabel,6) "res 9 / res 20"
    set dataVal(freeSelString,6) "resid 9 20"
    set dataVal([expr $dataOrigin+12],6) -130
    set dataVal([expr $dataOrigin+13],6) -100
    set dataVal([expr $dataOrigin+14],6)  170 
    set dataVal([expr $dataOrigin+15],6)  90 
    set dataVal([expr $dataOrigin+16],6)  110 
    set dataVal([expr $dataOrigin+17],6) -140 
    set dataVal([expr $dataOrigin+30],6) 150 
     #since we have dataVal (---,0) to (---,6) we have 7, but
    # this number is elesewhere set so gives last number from 0 count.
    set dataValNum 6
} 

proc ::timeline::threshChanged { var} {
  thresholdMakeGraph
}

proc ::timeline::thresholdMakeGraph {} {
  variable w
  variable currentMol 
  variable nullMolString
  variable usableMolLoaded
  variable dataValNum

  #since trace vars head here, find way to prevent
  #arrival whiles still calculating values
  if {$usableMolLoaded && ($currentMol != $nullMolString) &&  ($dataValNum >=0)} {
  
    thresholdData
    $w.threshGraph delete xScalable
    drawThresholdGraph
  }
}

proc ::timeline::thresholdClearGraph {} {
  variable w
 $w.threshGraph delete xScalable
}

proc ::timeline::readDataFile {filename} {

  variable w
  variable dataOrigin
  variable dataMin
  variable dataMax
  variable dataVal
  variable dataHash
  variable dataValNum
  variable dataName
  variable rectCreated 
  variable dataFileVersion 
  variable usesFreeSelection
  variable dataName
  variable lastCalc  
  variable numFrames
  if {$filename == ""  } {
    set filename [tk_getOpenFile -initialfile $filename -title "Open Trajectory Data file" -parent $w -filetypes [list { {.tml files} {.tml} } { {Text files} {.txt}} {{All files} {*} }] ]

  } 
  if {$filename == ""} {return}
  clearData
  set dataFile [open $filename r]
  #get file lines into an array
  set commonName ""
  set fileLines ""
  while {! [eof $dataFile] } {
    gets $dataFile curLine
    if { (! [regexp "^#" $curLine] ) && ($curLine != "" ) } {
      lappend fileLines $curLine
      if {[llength $fileLines] < 100} {
      tlPutsDebug "lines [llength $fileLines] is >$curLine<"
      }
    } else {
       
      if { [regexp "^# FILE_VERSION=" $curLine] } { 
        set inputFileVersion [lindex [split $curLine " "] 2]
        tlPutsDebug "Loading file, file version is $commonName"
      }
      if { [regexp "^# DATA_TITLE=" $curLine] } { 
         regexp "^# DATA_TITLE= (.*)$" $curLine matchall commonName
        tlPutsDebug "Loading file, field name is >$commonName<"
      } 
      if { [regexp "^# FREE_SELECTION=" $curLine] } { 
        set usesFreeSelection [lindex [split $curLine " "] 2]
      } 
      if { [regexp "^# NUM_ITEMS=" $curLine] } { 
        set dataValNum [expr [lindex [split $curLine " "] 2] -1]
      } 
      if { [regexp "^# MOL_NAME=" $curLine] } { 
        set molName [lindex [split $curLine " "] 2]
      } 
    }
  }
  #done with the file close it 
  close $dataFile
  tlPutsDebug "inputFileVersion= $inputFileVersion   usesFreeSelection= $usesFreeSelection  dataValNum= $dataValNum  numItems= [expr $dataValNum+1]  "
  
  if {$usesFreeSelection==1} {
     #tlPutsDebug "start checking free selection"
    set dataVal($dataOrigin) $commonName
    for {set f 0} {$f <= $numFrames} {incr f} {
      set dataName([expr $f + $dataOrigin]) $commonName
    }

    foreach curLine $fileLines {
      tlPutsDebug "curLine= $curLine"
      if { [regexp "^freeSelLabel" $curLine] } { 
         #tlPutsDebug "found freeSelLabel..."
         regexp "^freeSelLabel (\\d+) (.*)$" $curLine matchall itemNum theLabel
         set  dataVal(freeSelLabel,$itemNum) $theLabel
         #tlPutsDebug "dataVal(freeSelLabel,$itemNum)= $dataVal(freeSelLabel,$itemNum)  theLabel= >$theLabel<"
      }
      if { [regexp "^freeSelString" $curLine] } { 
         regexp "^freeSelString (\\d+) (.*)$" $curLine matchall itemNum theSelString
         set  dataVal(freeSelString,$itemNum) $theSelString
      }
     
      if {[regexp "^(\\d+) (\\d+) (.+)$" $curLine matchall frameNum itemNum theVal]} {
        #proceed through lines
        set curField [expr $dataOrigin + $frameNum]
        set dataVal($curField,$itemNum) $theVal
        set dataVal($curField) $commonName
        #tlPutsDebug "framenum= >$frameNum< itemNum= $itemNum  theVal= $theVal curField= $curField   dataVal($curField,$itemNum)= >$dataVal($curField,$itemNum)<"
      }
    }
  } else {
      set frameList ""
      #data-containing frames
      foreach line $fileLines {
        #puts "the line is >$line<"
        foreach {resid chain atom frame val} [split $line " "] {}
        lappend frameList $frame
        if {[llength $frameList] < 300} then {
          tlPutsDebug "frameList has [llength $frameList] elements. resid= $resid chain= $chain atom= $atom frame= $frame val= $val" 
        }
      }  
      #puts "frameList is $frameList"
      tlPutsDebug "length of frameList is [llength $frameList]" 
      set frameList [lsort -unique -increasing -integer $frameList]
      set minFrame [lindex $frameList 0]
      set maxFrame [lindex $frameList end]
           tlPutsDebug "frameList is $frameList"
      #  no longer find frame list, since catching errors on frame assignment
      #has same effect.  Could still 
      #assign values in a new Group
      # (temporarlily, to hard-coded fields, if still in hacky version)
      tlPutsDebug "now check fileLines:\n"
      foreach line $fileLines {
        #puts "assigning data, the line is >$line<"
        foreach {resid chain atom frame val} [split $line " "] {}
        
        
        #this assumes consecutive frames, should use frameList somewhere
        # if we really want proper reverse lookup
        if { [ catch {set fieldForFrame [expr $dataOrigin + $frame ]} ] } {
          set fieldForFrame -2
          puts "couldn't read frame text \"$frame\""
        }
        # XX resid,chain should in all cases be resid,chain,segid so not ambiguous
        #now do lookup via dataHash to find index in dataVal 
        if {[catch {set theIndex $dataHash($resid,$chain)} ]} {
          puts "failed to find data for resid=$resid, chain=$chain"
        } else {
           if { [catch {set dataVal($fieldForFrame,$theIndex) $val} ]} {
           puts "didn't find data for frame $frame, field= $fieldForFrame, index= $theIndex, new_val= $val"
         } else {
           set dataName($fieldForFrame) $commonName
        }
       }
    }   
  }
  #now delete the list of data lines, no longer needed
  unset fileLines

  set lastCalc -1
  #redraw the data rects
  showall 1  
  configureSelInfo null 0
  initPicked 
  return
}

proc ::timeline::loadDataCollection {} {
 variable w
 #batch load code  here
 set ext "\[tT\]\[mM\]\[lL\]"
 set dir [tk_chooseDirectory   -title "Choose a data collection directory"]
 set fileList [lsort [glob -directory $dir -type f *.$ext]]
 $w.menubar.data.menu delete 0 end
 $w.menubar.data.menu add command -label "Set collection directory..." -command  [namespace code loadDataCollection]
 foreach f $fileList  {
   set shortf [file tail $f]
   #set cmd "\{readDataFile $f\}"
   $w.menubar.data.menu add command -label "$shortf" -command [namespace code "readDataFile $f"]
 }
}


proc ::timeline::thresholdData {} {
  variable w
  variable dataVal
  variable dataValNum
  variable dataName
  variable dataOrigin
  variable numFrames
  variable usableMolLoaded
  variable rectCreated
  variable lastCalc
  variable dataThresh
  variable dataThreshVal
  variable thresholdBoundMin
  variable thresholdBoundMax

  #puts "in thresholdData, starting"
  set endField [expr $dataOrigin + $numFrames - 1 ]
  for {set field $dataOrigin} {$field <= $endField} {incr field} {
    set dataThreshVal($field) 0
    #puts "just set dataThresVal($field) to 0"
    for {set i 0} {$i<$dataValNum} {incr i} {
      #puts "tD, started loop, i= $i"
      if {$dataName($dataOrigin) == "struct"} {
        #if { ($dataVal($field,$i) == $thresholdBoundMin) || ($dataVal($field,$i) == $thresholdBoundMax)} 
        #hack for demo, so struct at least shows some graph,  do something sane with the later
        if { ($dataVal($field,$i) == "E") || ($dataVal($field,$i) == "T")} {
            incr dataThreshVal($field) 
            set dataThresh($field,$i) 1
          } else {
            set dataThresh($field,$i) 0
          }
             
        } else {

          if { ($dataVal($field,$i) > $thresholdBoundMin) && ($dataVal($field,$i) < $thresholdBoundMax)} {
              incr dataThreshVal($field) 
              #puts "incremented  dataThreshVal($field)=  $dataThreshVal($field)" 


              set dataThresh($field,$i) 1
          } else {
              set dataThresh($field,$i) 0
          }
          #puts " dataThresh($field,$i)=  $dataThresh($field,$i)" 
        }
     }
     #puts "dataThreshVal($field)= $dataThreshVal($field)"   
 }
  return
}

proc ::timeline::drawThresholdGraph {} {
  variable dataThreshVal
  variable numFrames
  variable dataOrigin
  variable threshGraphHeight 
  variable scalex
  variable xcol
  variable w
  variable dataWidth
  variable currentMol
  variable nullMolString
  variable usableMolLoaded
  #if {!($usableMolLoaded) || ($currentMol == $nullMolString)} {
  #   return
  # }

  #find min and max of Thresholds
  #make these variables later
  set threshPlotTop [expr  4 ]
  set threshPlotBottom [expr $threshGraphHeight - 5] 
  set lastField [expr $dataOrigin + $numFrames - 1]
  set minThresh $dataThreshVal($dataOrigin)
  set minThreshField $dataOrigin
  set maxThresh $dataThreshVal($dataOrigin)
  set maxThreshField $dataOrigin
  for {set field [expr $dataOrigin+1]} {$field<=$lastField} {incr field} {
    if {$dataThreshVal($field) < $minThresh} {set minThresh $dataThreshVal($field); set minThreshField $field} 
    if {$dataThreshVal($field) > $maxThresh} {set maxThresh $dataThreshVal($field); set maxThreshField $field} 
  }
  if {$maxThresh == 0} {set depictedMaxThresh 10} else {set depictedMaxThresh $maxThresh}
  set plotFactor [expr  ($threshPlotBottom-$threshPlotTop)/(0.0+$depictedMaxThresh) ]
     #puts "threshPlotTop=$threshPlotTop  thresPlotBottom=$threshPlotBottom   plotFactor= $plotFactor" 
   #count will be 0-based
   #later can do min based
   ##set plotFactor [expr 0.0+($threshPlotTop-$threshPlotBottom)/($maxThresh-$minThresh)]

  $w.threshGraph delete threshPlotBar 
  set endField [expr $dataOrigin + $numFrames - 1 ]

  for {set field $dataOrigin} {$field <= $endField} {incr field} {
    set frame [expr $field - $dataOrigin]
    set intermed [expr $plotFactor * $dataThreshVal($field)]
    set plotY [expr $threshPlotBottom - ($plotFactor * $dataThreshVal($field))]
    #puts "val= $intermed  dataThreshVal($field)= $dataThreshVal($field)  plotY=$plotY, field= $field threshPlotBottom=$threshPlotBottom "
    set xStart  [expr  ( ($frame + 0.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    set xEnd  [expr  ( ($frame + 1.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    $w.threshGraph create rectangle  $xStart $threshPlotBottom $xEnd $plotY -fill "\#EE7070"  -tags [list xScalable threshPlotBar]
    #puts "plotted   $xStart $threshPlotBottom $xEnd $plotY"
  }
    #mark min of the thresh
    set xStart  [expr  ( ([expr $minThreshField-$dataOrigin] + 0.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    set xEnd  [expr  ( ([expr $minThreshField-$dataOrigin]  + 1.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    $w.threshGraph create rectangle  $xStart [expr $threshPlotBottom+1] $xEnd [expr $threshPlotBottom +4] -fill "\#991010" -outline "" -tags [list xScalable threshPlotBar]

     #mark max of the thresh
    set xStart  [expr  ( ( $maxThreshField-$dataOrigin + 0.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    set xEnd  [expr  ( ( $maxThreshField-$dataOrigin + 1.0 ) * ($dataWidth * $scalex)) +  $xcol($dataOrigin)]
    $w.threshGraph create rectangle  $xStart [expr $threshPlotTop-1] $xEnd [expr $threshPlotTop-4] -fill "\#109910" -outline "" -tags [list xScalable threshPlotBar]

  #graph thresholds in $w.threshGraph
}
#puts "DEBUG: VMD Timeline: about to define calcTestHbonds"

proc ::timeline::calcSaltBridge {lastCalcVal} { 

  variable w
  variable dataName
  variable dataVal
  variable dataValNum
  variable currentMol
  variable firstTrajField
  variable numFrames
  variable dataMin
  variable dataMax
  variable trajMin
  variable trajMax 
  variable lastCalc
  variable dataOrigin
  variable nullMolString
  variable usesFreeSelection
  variable ONDist
  set usesFreeSelection 1

  set lastCalc $lastCalcVal

  #XXX should allow external control of hbond params
  #angle cutoff in degrees
 if {$currentMol == $nullMolString} {
      #should really gray out choices unless molec is seleted XXX
      return 
  }

  set listOfFrames ""
  set acsel [atomselect $currentMol "(protein and acidic and oxygen and not backbone)"]
  set bassel [atomselect $currentMol "(protein and basic and nitrogen and not backbone)"]
  for {set trajFrame 0} {$trajFrame < $numFrames} {incr  trajFrame} {
   set dataName([expr $trajFrame+$dataOrigin]) "salt bridge"
   $acsel frame $trajFrame
   $bassel frame $trajFrame
   lappend listOfFrames [measure contacts $ONDist $acsel $bassel]
  }
  

  #XXX hard set here, elsewhere should be actual value
  set groupValue 1

  set frame 0
  foreach f $listOfFrames {
    set frameList($frame) ""
    tlPutsDebug ": At top, frame= $frame   frameList($frame)=$frameList($frame)"
    #next line isn't loop, just a single assignment
    foreach {oxlist nitlist} $f {
      set selString ""
      tlPutsDebug "ox= $oxlist nitlist= $nitlist"
      foreach  o $oxlist n $nitlist  {
        set selString "index $o $n"
        if {$selString != ""} {
          lappend frameList($frame) $selString      
          #no value assoc'd with each entry here, if present, assinged val =1

          #tlPutsDebug "DEBUG: frame= $frame   frameList($frame)=$frameList($frame)"

          #now count how many in each 
          #now go through current frames groups of three
          set spaceToCodeN [string map {" " %20} $selString]
          tlPutsDebug ": selString= $selString spaceToCodeN= $spaceToCodeN  frame=$frame"
          set seenData($spaceToCodeN,$frame) $groupValue
          if {[info exists seenCount($spaceToCodeN)]} {
            incr  seenCount($spaceToCodeN)
          } else {
            set seenCount($spaceToCodeN) 1
            #just to be on the safe side...
            set seenDataValGroup($spaceToCodeN) "null"

          }

        }
      }
    }
    incr frame
  }
  #tlPutsDebug ": all names = [array names seenCount]"

  #here the cutoff for being displayed is: 1
  #can set higher cutoff later
  set numDisplayGroups [llength [array names seenCount]  ]


  #clear data and set labels
  #clear out all frames, in real case set all data.
  # following line sets number of displayed groups to number of groups that have been seen, that 
  # is, showed data that met conditions
  set displayGroupTextList [array names seenCount]  
  # there are displayGroup+1 lines of data in the display (equivalent of residues)
  set displayGroup 0
  foreach displayGroupText $displayGroupTextList {
    tlPutsDebug "SaltBridge: displayGroup= $displayGroup  displayGroupText= $displayGroupText"
    set codeToSpaceN [string map {%20 " "} $displayGroupText]
    #regexp "^\\D+ (.*$)" $codeToSpaceN matchall regout1
    #set dataVal(freeSelLabel,$displayGroup) $regout1
    regexp "^index (\\d+) (\\d+)$" $codeToSpaceN matchall regout1 regout2
    set selox [atomselect $currentMol "index $regout1"]
    set selnit [atomselect $currentMol "index $regout2"]
    set dataVal(freeSelLabel,$displayGroup) "[$selox get resname][$selox get resid]--[$selnit get resname][$selnit get resid]"
    set dataVal(freeSelString,$displayGroup) "same residue as ($codeToSpaceN)"
   tlPutsDebug "selox = $selox dataVal(freeSelString,$displayGroup= $dataVal(freeSelString,$displayGroup)  dataVal(freeSelString,$displayGroup)    dataVal(freeSelString,$displayGroup= $dataVal(freeSelString,$displayGroup)    codeToSpaceN= $codeToSpaceN"
  

    #set the dataVal displayGroup that corresponds to displayGroupText, will be used when writing
    set seenDataValGroup($displayGroupText) $displayGroup
    for {set trajFrame 0} {$trajFrame < $numFrames} {incr  trajFrame} {
      set curField [expr $dataOrigin + $trajFrame]
      set dataVal($curField,$displayGroup) 0 
      # XXX shouldn't we actually be setting min/max in next two lines?
      set dataMin($curField) $trajMin 
      set dataMax($curField) $trajMax 
    }
  incr displayGroup
  }
  #use next line if really extracting data from traj geom.
  #$sel frame $trajFrame
  #set data (only the rare frames that have data)
  #clear out all frames, in real case set all data.
  #for {set displayGroup 0} {$displayGroup<numDisplayGroups} {incr displayGroup} 
  #first set labels
  set dataItems 0
  set displayGroupDataList [array names seenData]  
  foreach d $displayGroupDataList {
    foreach {itemDisplayGroupText itemFrame} [split $d ","] {
    #tlPutsDebug ": d= $d   itemDisplayGroupText=$itemDisplayGroupText  itemFrame= $itemFrame"
      #turn the name back into a label and a time
      #take the number after the final comma
      set curField [expr $dataOrigin + $itemFrame]
      set displayGroup $seenDataValGroup($itemDisplayGroupText)
      set dataVal($curField,$displayGroup) $seenData($d)
       # tcl string trick: $seenData($d) should be equivalent of $seenData($itemDisplayGroupText,$itemFrame)
      #XXXX swap seenData item-frame order, for consistenecy
      incr dataItems 
    }
    
  set dataValNum [expr $numDisplayGroups -1]
  initPicked 
  #tlPutsDebug " displayGroup= $displayGroup dataItems= $dataItems"
  #XXX the zero-base for a var named like "zzzzNum" is confusing.  Should set 
  #all things that refer to n objects have a value of n, not (n-1).
  } 

}
proc ::timeline::calcTestHbonds {lastCalcVal} { 

  variable w
  variable dataName
  variable dataVal
  variable dataValNum
  variable currentMol
  variable firstTrajField
  variable numFrames
  variable dataMin
  variable dataMax
  variable trajMin
  variable trajMax 
  variable lastCalc
  variable dataOrigin
  variable nullMolString
  variable usesFreeSelection
  set usesFreeSelection 1
  variable hbondDistCutoff
  variable  hbondAngleCutoff 

  set lastCalc $lastCalcVal

 if {$currentMol == $nullMolString} {
      #should really gray out choices unless molec is seleted XXX
      puts "Timeline: select molecule before choosing Calculate method"
      return 
  }

  set listOfFrames ""
  set sel  [atomselect $currentMol "protein or nucleic"] 
  for {set trajFrame 0} {$trajFrame < $numFrames} {incr  trajFrame} {
   set dataName([expr $trajFrame+$dataOrigin]) "H-bond"
   $sel frame $trajFrame
   lappend listOfFrames [measure hbonds $hbondDistCutoff  $hbondAngleCutoff  $sel]
  }
  

  #XXX hard set here, elsewhere should be actual value
  set groupValue 1

  set frame 0
  foreach f $listOfFrames {
    set frameList($frame) ""
    tlPutsDebug ": At top, frame= $frame   frameList($frame)=$frameList($frame)"
    #next line isn't loop, just a single assignment
    foreach {donors acceptors hydrogens} $f {
      set selString ""
      #tlPutsDebug ": donors= $donors  acceptors= $acceptors hydrogens=$hydrogens"
      foreach d $donors a $acceptors h $hydrogens {
        set selString "index $d $a $h"
        if {$selString != ""} {
          lappend frameList($frame) $selString      
          #no value assoc'd with each entry here, if present, assinged val =1

          #tlPutsDebug "DEBUG: frame= $frame   frameList($frame)=$frameList($frame)"

          #now count how many in each 
          #now go through current frames groups of three
          set spaceToCodeN [string map {" " %20} $selString]
          tlPutsDebug ": selString= $selString spaceToCodeN= $spaceToCodeN  frame=$frame"
          set seenData($spaceToCodeN,$frame) $groupValue
          if {[info exists seenCount($spaceToCodeN)]} {
            incr  seenCount($spaceToCodeN)
          } else {
            set seenCount($spaceToCodeN) 1
            #just to be on the safe side...
            set seenDataValGroup($spaceToCodeN) "null"

          }

        }
      }
    }
    incr frame
  }
  #tlPutsDebug ": all names = [array names seenCount]"

  #here the cutoff for being displayed is: 1
  #can set higher cutoff later
  set numDisplayGroups [llength [array names seenCount]  ]


  #clear data and set labels
  #clear out all frames, in real case set all data.
  # following line sets number of displayed groups to number of groups that have been seen, that 
  # is, showed data that met conditions
  set displayGroupTextList [array names seenCount]  
  # there are displayGroup+1 lines of data in the display (equivalent of residues)
  set displayGroup 0
  foreach displayGroupText $displayGroupTextList {
    tlPutsDebug ": displayGroup= $displayGroup  displayGroupText= $displayGroupText"
    set codeToSpaceN [string map {%20 " "} $displayGroupText]
    regexp "^\\D+ (.*$)" $codeToSpaceN matchall regout1
    set dataVal(freeSelLabel,$displayGroup) $regout1
    set dataVal(freeSelString,$displayGroup) $codeToSpaceN
    #set the dataVal displayGroup that corresponds to displayGroupText, will be used when writing
    set seenDataValGroup($displayGroupText) $displayGroup
    for {set trajFrame 0} {$trajFrame < $numFrames} {incr  trajFrame} {
      set curField [expr $dataOrigin + $trajFrame]
      set dataVal($curField,$displayGroup) 0 
      # XXX shouldn't we actually be setting min/max in next two lines?
      set dataMin($curField) $trajMin 
      set dataMax($curField) $trajMax 
    }
  incr displayGroup
  }
  #use next line if really extracting data from traj geom.
  #$sel frame $trajFrame
  #set data (only the rare frames that have data)
  #clear out all frames, in real case set all data.
  #for {set displayGroup 0} {$displayGroup<numDisplayGroups} {incr displayGroup} 
  #first set labels
  set dataItems 0
  set displayGroupDataList [array names seenData]  
  foreach d $displayGroupDataList {
    foreach {itemDisplayGroupText itemFrame} [split $d ","] {
    #tlPutsDebug ": d= $d   itemDisplayGroupText=$itemDisplayGroupText  itemFrame= $itemFrame"
      #turn the name back into a label and a time
      #take the number after the final comma
      set curField [expr $dataOrigin + $itemFrame]
      set displayGroup $seenDataValGroup($itemDisplayGroupText)
      set dataVal($curField,$displayGroup) $seenData($d)
       # tcl string trick: $seenData($d) should be equivalent of $seenData($itemDisplayGroupText,$itemFrame)
      #XXXX swap seenData item-frame order, for consistenecy
      incr dataItems 
    }
    
  set dataValNum [expr $numDisplayGroups -1]
  initPicked  
  #tlPutsDebug " displayGroup= $displayGroup dataItems= $dataItems"
  #XXX the zero-base for a var named like "zzzzNum" is confusing.  Should set 
  #all things that refer to n objects have a value of n, not (n-1).
  } 

}

#puts "DEBUG: VMD Timeline: about to define clearData"
proc ::timeline::clearData {} {
  variable w
  variable dataVal
  variable dataValNum
  variable dataOrigin
  variable numFrames
  variable usableMolLoaded
  variable rectCreated
  variable lastCalc
  variable dataMin
  variable dataMax
  set dataMin(all) 0
  set dataMax(all) 0 
  #XX should be null, but not set to use correctly

  set lastCalc 0
  tlPutsDebug "Info) Timeline: Clearing 2D data..."
  set endStructs [expr $dataOrigin + $numFrames - 1 ]
  for {set field $dataOrigin} {$field <= $endStructs} {incr field} {
    for {set i 0} {$i<=$dataValNum} {incr i} {

      set  dataVal($field,$i) "null"
      # for the special struct case, the 0 shold give default color
      #puts "dataVal($field,$i) is now $dataVal($field,$i)"
      #set resid $dataVal(0,$i)
      #set chain $dataVal(2,$i)
      #set frame [expr $field - $dataOrigin]
      #puts $writeDataFile "$resid $chain CA $frame $val"
      
    }
  }
  #redraw the data rects
  showall 1
  set dataValNum -1
  return
}
proc  ::timeline::userScaleBothChanged {val} {
  variable userScalex
  variable userScaley
  variable userScaleBoth
  variable scaley
  variable fit_scalex
  variable fit_scaley
  variable scalex
  set scalex [expr $userScaleBoth * $fit_scalex]
  set scaley [expr $userScaleBoth * $fit_scaley]
  set userScalex  $userScaleBoth
  set userScaley $userScaleBoth
  redraw name func op
  #puts "redrawn, userScaleBoth= $userScaleBoth, scalex= $scalex, userScalex= $userScalex, scaley= $scaley, userScaley= $userScaley"
  return
}



proc  ::timeline::userScalexChanged {val} {
  variable userScalex
  variable scalex
  variable fit_scalex
  set scalex [expr $userScalex * $fit_scalex]
  redraw name func op
  #puts "redrawn, scalex= $scalex, userScalex= $userScalex"
  return
}


proc ::timeline::userScaleyChanged {val} {
  variable userScaley
  variable scaley
  variable fit_scaley
  #until working ok, still do direct mapping
  set scaley [expr $userScaley * $fit_scaley]
  #set scaley $userScaley 
  redraw name func op
  tlPutsDebug "userScaleyChanged: redrawn, fit_scaley= $fit_scaley   scaley= $scaley   userScaley= $userScaley"
  return
}

proc ::timeline::drawVertScale {} {
  variable w
  variable ytopmargin
  variable scaley
  variable ybox
  variable dataValNum
  variable dataVal
  variable vertTextSkip
  variable verTextLeft
  variable vertTextRight
  variable resCodeShowOneLetter
  variable monoFont
  variable usesFreeSelection
  $w.vertScale delete vertScaleText 

  
  #when adding new column, add to this list (maybe adjustable later)
  #The picked fields 
  
  #Add the text...
  set field 0           

  #note that the column will be 0, but the data will be from picked
  
  
  set yDataEnd [expr $ytopmargin + ($scaley * $ybox * ($dataValNum +1))]
  set y 0.0

  set yposPrev  -10000.0

  #Add the text to vertScale...
  set field 0            



  #we want text to appear in center of the dataRect we are labeling
  set vertOffset [expr $scaley * $ybox / 2.0]

  #don't do $dataValNum, its done at end, to ensure always print last 
  for {set i 0} {$i <= $dataValNum} {incr i} {
    set ypos [expr $ytopmargin + ($scaley * $y) + $vertOffset]
    if { ( ($ypos - $yposPrev) >= $vertTextSkip) && ( ( $i == $dataValNum) || ( ($yDataEnd - $ypos) > $vertTextSkip) ) } {
      #tlPutsDebug "ypos= $ypos yposPrev= $yposPrev i= $i dataValNum= $dataValNum yDataEnd= $yDataEnd vertTextSkip= $vertTextSkip vertTextRight= $vertTextRight vertOffset= $vertOffset"
      if {$usesFreeSelection} {
        $w.vertScale create text $vertTextRight $ypos -text $dataVal(freeSelLabel,$i)  -width 200 -font $monoFont -justify right -anchor e -tags vertScaleText 
       } else {
        if {$resCodeShowOneLetter == 0} {
          set res_string $dataVal(1,$i)
        } else {
          set res_string $dataVal(1code,$i)
        }
       #for speed, we use vertScaleText instead of $dataName($field)
      $w.vertScale create text $vertTextRight $ypos -text "$dataVal(0,$i) $res_string $dataVal(2,$i)" -width 200 -font $monoFont -justify right -anchor e -tags vertScaleText 
      }
      set yposPrev  $ypos
    }        
    set y [expr $y + $vertTextSkip]
    
  } 
  
  
}


proc ::timeline::drawHorzScale {} {
  variable w
  variable ytopmargin
  variable scalex
  variable dataValNum
  variable dataVal
  variable monoFont
  variable dataOrigin
  variable xcol
  variable numFrames    
  variable dataWidth

  $w.horzScale delete horzScaleText 

  
  #when adding new column, add to this list (maybe adjustable later)
  #The picked fields 
  
  #Add the text...

  #note that the column will be 0, but the data will be from picked
  
  #we want text to appear in center of the dataRect we are labeling
  set fieldLast [expr $dataOrigin + $numFrames - 1]
  #ensure minimal horizontal spacing
  # hardcoded spacing
  set horzSpacing 27 
  set horzDataTextSkip [expr $dataWidth]
  set scaledHorzDataTextSkip [expr $scalex * $dataWidth]
  set scaledHorzDataOffset [expr $scalex * $dataWidth / 2.0]
  set ypos 20 
  set xStart [expr ($xcol($dataOrigin))]
  set xDataEnd  [expr int ($xStart +  $scalex * ($dataWidth * $numFrames ) ) ] 
  set x 0 



  #numbers are scaled for 1.0 until xpos
  #this is tied to data fields, which is produced from frames upong
  #first drawing. Should really agreee with writeDataFile, which currently uses frames, not fields
  set xposPrev -1000 
  #traj data starts at dataOrigin
  for {set frameNum 0} {$frameNum < $numFrames} {incr frameNum} {
    set field [expr $frameNum + $dataOrigin]

  ####for {set field [expr $dataOrigin]} {$field <= $fieldLast} {incr field} {}
  ####  set frameNum [expr $field - $dataOrigin -1]
    
    set xpos [expr int ($xStart + ($scalex * $x) + $scaledHorzDataOffset)]
    if { ( ($xpos - $xposPrev  ) >= $horzSpacing) && ( ( $field == $fieldLast) || ( ( $xDataEnd - $xpos) > ( 2 * $horzSpacing) ) ) } {
      # draw the frame number if there is room
      #for speed, we use horzScaleText instead of $dataName($field)
      $w.horzScale create text $xpos $ypos -text "$frameNum" -width 30 -font $monoFont -justify center -anchor s -tags horzScaleText 
      set xposPrev  $xpos
    }        
    set x [expr $x + $horzDataTextSkip]
  } 

  
}

#puts "--DEBUG--:Timeline: Completed defining drawHorzScale"

#############################################
# end of the proc definitions
############################################






####################################################
# Execution starts here. 
####################################################

#####################################################
# set traces and some binidngs, then call timeLineMain
#####################################################
proc ::timeline::startTimeline {} {
  
  ####################################################
  # Create the window, in withdrawn form,
  # when script is sourced (at VMD startup)
  ####################################################
  variable w .vmd_timeline_Window
  set windowError 0
  set errMsg ""

  #if timeline has already been started, just deiconify window
  if { [winfo exists $w] } {
    wm deiconify $w 
    return
  }

  if { [catch {toplevel $w -visual truecolor} errMsg] } {
    puts "Info) Timeline window can't find trucolor visual, will use default visual.\nInfo)   (Error reported was: $errMsg)" 
    if { [catch {toplevel $w } errMsg ]} {
      puts "Info) Default visual failed, Timeline window cannot be created. \nInfo)   (Error reported was: $errMsg)"    
      set windowError 1
    }
  }
  if {$windowError == 0} { 
    #don't withdraw, not under vmd menu control during testing
    #wm withdraw $w
    wm title $w "VMD Timeline"
    #wm resizable $w 0 0 
    wm resizable $w 1 1 

    variable w
    variable monoFont
    variable initializedVars 0
    variable needsDataUpdate 0 

    #overkill for debugging, should only need to delete once....
    trace vdelete currentMol w [namespace code molChoose]
    trace vdelete currentMol w [namespace code molChoose]
    trace vdelete ::vmd_pick_atom w  [namespace code listPick] 
    trace vdelete ::vmd_pick_atom w  [namespace code listPick] 
    trace vdelete ::vmd_initialize_structure w  [namespace code molChooseMenu]
    trace vdelete ::vmd_initialize_structure w  [namespace code molChooseMenu]


    bind $w <Map> "+[namespace code Show]"
    bind $w <Unmap> "+[namespace code Hide]"
    #specify monospaced font, 12 pixels wide
    font create tkFixedTimeline -family Courier -size -12
    #for test run tkFixedTimeline was made by normal sequence window
    #change this so plugins don't depend on eachOther:1
    set monoFont tkFixedTimeline

    #call to set up, after this, all is driven by trace and bind callbacks
    timeLineMain
  }
  return $w
}
#puts "--DEBUG--: VMD Timeline: finished defining startTimeLine, reached last line of sourcefile"

proc ::myCountContacts {resAtomSel  resCompleteSel  proteinNucSel} {
                    return [llength [lindex [measure contacts 4.0 $resCompleteSel $proteinNucSel] 0]]
}
proc ::myResX {resAtomSel  resCompleteSel  proteinNucSel} {
                    return [$resAtomSel get x]
}
proc timeline::myResPhi {resAtomSel  resCompleteSel  proteinNucSel} {
                    return [$resAtomSel get phi]
}

