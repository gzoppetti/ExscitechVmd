############################################################################
#cr
#cr            (C) Copyright 1995-2007 The Board of Trustees of the
#cr                        University of Illinois
#cr                         All Rights Reserved
#cr
############################################################################

############################################################################
# RCS INFORMATION:
#
#       $RCSfile: logfile.tcl,v $
#       $Author: johns $        $Locker:  $             $State: Exp $
#       $Revision: 1.4 $        $Date: 2007/01/12 20:11:31 $
#
############################################################################

# This file implements the logfile command, which provides logging of
# all VMD commands to a file or to the console.

# To do:
#  * Maybe allow multiple logfiles to be in progress at once
#  * Add an option to send logfile commands to a remote system a la vmdcollab



# the channel to which we are logging commands
set vmd_logfile_channel off

proc logfile { fname } {
  global vmd_logfile_channel

  if { [string equal $fname off] } {
    if { [string equal $vmd_logfile_channel off] } {
      error "Not currently logging commands."
    }
    puts $vmd_logfile_channel "# [vmdinfo versionmsg]"
    puts $vmd_logfile_channel "# end of log file."
    if { ![string equal $vmd_logfile_channel stdout] } {
      close $vmd_logfile_channel
    }
    set vmd_logfile_channel off
  } else {
    # If already logging, print error message
    if { ![string equal $vmd_logfile_channel off] } {
      error "Command logging already in progress; use 'logfile off' to stop."
    }
    # If logging to console, use stdout
    if { [string equal $fname console] } {
      set vmd_logfile_channel stdout
    } else {
      set rc [catch { open $fname w } msg]
      if { $rc } {
        error "Can't open log file '$fname' for writing: $msg"
      } else {
        set vmd_logfile_channel $msg
      }
    }
    puts "Logging commands to '$fname'."
    puts $vmd_logfile_channel "# [vmdinfo versionmsg]"
    puts $vmd_logfile_channel "# Log file '$fname', created by user $::tcl_platform(user)"
    flush $vmd_logfile_channel
  }
  return
}

proc vmd_logfile_cb { name1 name2 op } {
  global vmd_logfile_channel
  global vmd_logfile

  if { [string equal $vmd_logfile_channel off] } { return }
  if { [string equal $vmd_logfile exit] } {
    logfile off
    return
  }
  puts $vmd_logfile_channel $vmd_logfile
  flush $vmd_logfile_channel
}

trace variable vmd_logfile w vmd_logfile_cb

