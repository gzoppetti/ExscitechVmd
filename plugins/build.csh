#!/bin/csh

##
## Script for building plugins on all supported platforms
##
#setenv DATE `date +%Y-%m-%d-%T`
setenv DATE `date +%m%d-%H%M%S`

##
## BioCoRE logging (eventually, need other changes first)
##
#setenv BUILDNUM `cat /Projects/vmd/vmd/vmdbuild.number`;
#setenv LOGRUN  'biolog -f -p vmd -k "VMD plugins build $BUILDNUM, BUILD SUMMARY" -s "VMD build $BUILDNUM, BUILD SUMMARY"'
#setenv LOGGING 'biolog -f -p vmd -k "VMD plugins build $BUILDNUM, Platform: $1"  -s "VMD build $BUILDNUM, Platform: $1"'

setenv unixdir `pwd` 

##
## Check for builds on remote hosted supercomputers, etc.
##
switch ( `hostname` )
  ## NCSA 'Cobalt' SGI Altix 
  case co-login*:
    echo "Using build settings for NCSA SGI Altix..."
    setenv NETCDFINC -I/home/ac/stonej/vmd/lib/netcdf/include
    setenv NETCDFLIB -L/home/ac/stonej/vmd/lib/netcdf
    setenv TCLINC -I/home/ac/stonej/vmd/lib/tcl/include
    setenv TCLLIB -L/home/ac/stonej/vmd/lib/tcl
    cd $unixdir; gmake LINUXIA64 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_LINUXIA64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUXIA64 >& log.LINUXIA64.$DATE  < /dev/null & 
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo ""
    echo "Plugin builds done..."
    breaksw;


  ## NCSA "Blue Print", "Copper"
  case bp-login1:
  case cu12:
    echo "Using build settings for NCSA IBM Regatta..."
    setenv TCLINC -I/u/home/ac/stonej/vmd/lib/tcl/include
    setenv TCLLIB -L/u/home/ac/stonej/vmd/lib/tcl
    cd $unixdir; gmake AIX6_64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_AIX6_64 >& log.AIX6_64.$DATE < /dev/null &
#    setenv NETCDFINC -I/u/ac/stonej/vmd/lib/netcdf/include
#    setenv NETCDFLIB -L/u/ac/stonej/vmd/lib/netcdf
#    setenv TCLINC -I/u/ac/stonej/vmd/lib/tcl/include
#    setenv TCLLIB -L/u/ac/stonej/vmd/lib/tcl
#    cd $unixdir; gmake AIX6 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_AIX6 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_AIX6 >& log.AIX6.$DATE < /dev/null &
#    cd $unixdir; gmake AIX6_64 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_AIX6_64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_AIX6_64 >& log.AIX6_64.$DATE  < /dev/null &
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo ""
    echo "Plugin builds done..."
    breaksw;


   ## Indiana BigRed
  case s10c2b6:
    echo "Using build settings for IU BigRed PowerPC Linux..."
#    setenv NETCDFINC -I/N/hd03/tg-johns/BigRed/vmd/lib/netcdf/include
#    setenv NETCDFLIB -L/N/hd03/tg-johns/BigRed/vmd/lib/netcdf
    setenv TCLINC -I/N/hd03/tg-johns/BigRed/vmd/lib/tcl/include
    setenv TCLLIB -L/N/hd03/tg-johns/BigRed/vmd/lib/tcl
#    cd $unixdir; gmake LINUXPPC64 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_LINUXPPC64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUXPPC64 >& log.LINUXPPC64.$DATE < /dev/null & 
    cd $unixdir; gmake LINUXPPC64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUXPPC64 >& log.LINUXPPC64.$DATE < /dev/null &
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo "^G^G^G^G"
    echo "Plugin builds done..."
    breaksw;


  ## TCBG development machines 
  case casablanca*:
  case moline*:
    echo "Using build settings for TB network..."
    setenv NETCDFINC -I/Projects/vmd/vmd/lib/netcdf/include
    setenv NETCDFLIB -L/Projects/vmd/vmd/lib/netcdf

    setenv TCLINC -I/Projects/vmd/vmd/lib/tcl/include
    ## MacOS X framework paths
    setenv TCLLIB -F/Projects/vmd/vmd/lib/tcl

# Use our own custom Tcl framework
    ssh -x sydney "cd $unixdir; gmake MACOSX NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_MACOSX TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_MACOSX >& log.MACOSX.$DATE " < /dev/null &
# Use Apple-Provided Tcl framework
#    ssh -x sydney "cd $unixdir; gmake MACOSX NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_MACOSX TCLINC=-F/System/Library/Frameworks TCLLIB=-F/System/Library/Frameworks >& log.MACOSX.$DATE " < /dev/null &

# Use our own custom Tcl framework
    ssh -x juneau "cd $unixdir; gmake MACOSXX86 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_MACOSXX86 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_MACOSXX86 >& log.MACOSXX86.$DATE " < /dev/null &
# Use Apple-Provided Tcl framework
#    ssh -x juneau "cd $unixdir; gmake MACOSXX86 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_MACOSXX86 TCLINC=-F/System/Library/Frameworks TCLLIB=-F/System/Library/Frameworks >& log.MACOSXX86.$DATE " < /dev/null &

    ##
    ## link paths for rest of the unix platforms
    ##
    setenv TCLLIB -L/Projects/vmd/vmd/lib/tcl

# build X11/Unix style 64-bit VMD for MacOS X since Tcl/Tk use Carbon otherwise
#    ssh -x bogota "cd $unixdir; gmake MACOSXX86_64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_MACOSXX86_64 >& log.MACOSXX86_64.$DATE " < /dev/null &

#    ssh -x beirut "cd $unixdir; gmake AIX4 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_AIX4 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_AIX4 >& log.AIX4.$DATE " < /dev/null &

    ssh -x dallas "cd $unixdir; gmake LINUX NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_LINUX TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUX >& log.LINUX.$DATE " < /dev/null &

    ssh -x dallas "cd $unixdir; gmake LINUXAMD64 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_LINUXAMD64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUXAMD64 >& log.LINUXAMD64.$DATE " < /dev/null &

#    ssh -x titan "cd $unixdir; gmake IRIX6 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_IRIX6 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_IRIX6 >& log.IRIX6.$DATE" < /dev/null &

#    ssh -x titan  "cd $unixdir; gmake IRIX6_64 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_IRIX6_64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_IRIX6_64 >& log.IRIX6_64.$DATE" < /dev/null &

#    ssh -x cupertino "cd $unixdir; gmake SOLARIS2 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_SOLARIS2 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_SOLARIS2 >& log.SOLARIS2.$DATE" < /dev/null &

    ssh -x cupertino "cd $unixdir; gmake SOLARIS2_64 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_SOLARIS2_64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_SOLARIS2_64 >& log.SOLARIS2_64.$DATE" < /dev/null &

#    ssh -x cancun "cd $unixdir; gmake SOLARISX86 SOLARISX86 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_SOLARISX86 >& log.SOLARISX86.$DATE" < /dev/null &

    ssh -x cancun "cd $unixdir; gmake SOLARISX86_64 SOLARISX86_64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_SOLARISX86_64 >& log.SOLARISX86_64.$DATE" < /dev/null &

#    ssh -x ganymede "cd $unixdir; gmake HPUX11 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_HPUX11 >& log.HPUX11.$DATE" < /dev/null &

#    ssh -x galatea "cd $unixdir; gmake TRU64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_TRU64 >& log.TRU64.$DATE" < /dev/null &


    ## Win32 include/link paths
    setenv windir /cygdrive/j/plugins
    setenv TCLINC -IJ:/vmd/lib/tcl/include
    setenv TCLLIB /LIBPATH:J:/vmd/lib/tcl
    ssh -1 -x administrator@malta "cd $windir; make WIN32 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_WIN32 >& log.WIN32.$DATE" < /dev/null &

    ## Win64 include/link paths
    setenv windir /cygdrive/j/plugins
    setenv TCLINC -IJ:/vmd/lib/tcl/include
    setenv TCLLIB /LIBPATH:J:/vmd/lib/tcl
#    ssh -1 -x Administrator@honolulu "cd $windir; make WIN64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_WIN64 >& log.WIN64.$DATE" < /dev/null &
#    ssh -1 -x Administrator@honolulu "cd $windir; make WIN64 >& log.WIN64.$DATE" < /dev/null &

    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo ""
    echo "Plugin builds done..."
    breaksw;


  ## proteus/toledo CUDA test boxes
  case proteus*:
  case photon*:
    echo "Using build settings for TB network..."
    setenv NETCDFINC -I/Projects/vmd/vmd/lib/netcdf/include
    setenv NETCDFLIB -L/Projects/vmd/vmd/lib/netcdf
    setenv TCLINC -I/Projects/vmd/vmd/lib/tcl/include
    setenv TCLLIB -L/Projects/vmd/vmd/lib/tcl
    gmake LINUX NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_LINUX TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUX
    echo "Plugin builds done..."
    breaksw;
 
 
  ## Photon (John's E4500)
  case photon*:
    echo "Using build settings for Photon..."
    setenv NETCDFINC -I/home/johns/vmd/lib/netcdf/include
    setenv NETCDFLIB -L/home/johns/vmd/lib/netcdf
    setenv TCLINC -I/home/johns/vmd/lib/tcl/include
    setenv TCLLIB -L/home/johns/vmd/lib/tcl
#    ssh -x photon "cd $unixdir; gmake SOLARIS2_64 NETCDFINC=$NETCDFINC NETCDFLIB=$NETCDFLIB/lib_SOLARIS2_64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_SOLARIS2_64 >& log.SOLARIS2_64.$DATE" < /dev/null &
    cd $unixdir; gmake SOLARIS2_64 TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_SOLARIS2_64
 >& log.SOLARIS2_64.$DATE  < /dev/null &                                        
    echo "Waiting for all plugin make jobs to complete..."
    wait;
    echo "^G^G^G^G"
    echo "Plugin builds done..."
    breaksw;


  ###
  ### XXXNEWPLATFORM
  ###
  default:
    echo "Unrecognized host system, add your own switch statement to customize"
    echo "for your build environment.  Edit build.csh and change the variables"
    echo "in the section marked XXXNEWPLATFORM."
    # setenv TCLINC -I/your/tcl/include/directory
    # setenv TCLLIB -L/your/tcl/library/directory
    # cd $unixdir; gmake LINUX TCLINC=$TCLINC TCLLIB=$TCLLIB/lib_LINUX >& log.LINUX.$DATE  < /dev/null &
    # echo "Waiting for all plugin make jobs to complete..."
    # wait;
    # echo ""
    # echo "Plugin builds done..."
    breaksw;
endsw



