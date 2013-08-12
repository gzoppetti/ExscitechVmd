AR= ar
ARFLAGS = cr
RANLIB = ranlib

SRCDIR=src
INCDIR=-Isrc -I../molfile_plugin/src
COMPILEDIR = ../compile
ARCHDIR=${COMPILEDIR}/lib_${ARCH}/tcl/hesstrans1.0
VPATH = src ${ARCHDIR}

##
## Disable building hesstrans plugin on platforms that it can't be ported to
## XXX ideally we'll fix the code, but until then...
##
ifneq ($(ARCH),LINUXIA64)
ifneq ($(ARCH),WIN64)
TARGETS = ${ARCHDIR} ${ARCHDIR}/hesstrans.so
endif
endif

bins:
win32bins:
dynlibs: ${TARGETS}
staticlibs:
win32staticlibs:

distrib:
	for localname in `find ../compile -name hesstrans.so -print` ; do \
		pluginname=`echo $$localname | sed s/..\\\/compile\\\/lib_// `; \
		dir=`dirname $(PLUGINDIR)/$$pluginname`; \
		mkdir -p $$dir; \
		cp $$localname $(PLUGINDIR)/$$pluginname; \
		cp pkgIndex.tcl hesstrans.tcl hessTest.tcl $$dir ; \
	done
	
${ARCHDIR}:
	mkdir -p ${ARCHDIR}

HESSTRANSOBJS=${ARCHDIR}/newmat1.o \
	${ARCHDIR}/newmat2.o \
	${ARCHDIR}/newmat3.o \
	${ARCHDIR}/newmat4.o \
	${ARCHDIR}/newmat5.o \
	${ARCHDIR}/newmat6.o \
	${ARCHDIR}/newmat7.o \
	${ARCHDIR}/newmat8.o \
	${ARCHDIR}/newmat9.o \
	${ARCHDIR}/newmatex.o \
	${ARCHDIR}/bandmat.o \
	${ARCHDIR}/submat.o \
	${ARCHDIR}/myexcept.o \
	${ARCHDIR}/cholesky.o \
	${ARCHDIR}/evalue.o \
	${ARCHDIR}/fft.o \
	${ARCHDIR}/hholder.o \
	${ARCHDIR}/jacobi.o \
	${ARCHDIR}/newfft.o \
	${ARCHDIR}/sort.o \
	${ARCHDIR}/svd.o \
	${ARCHDIR}/newmatrm.o \
	${ARCHDIR}/hesstrans_wrap.o \
	${ARCHDIR}/hesstrans.o


${ARCHDIR}/hesstrans.so : ${HESSTRANSOBJS}
	if [ -n "${TCLSHLD}" ]; \
	then ${TCLSHLD} $(LOPTO)$@ ${HESSTRANSOBJS} ${TCLLIB} ${TCLLDFLAGS} ${LDFLAGS}; \
	else ${SHLD} $(LOPTO)$@ ${HESSTRANSOBJS} ${TCLLIB} ${TCLLDFLAGS} ${LDFLAGS}; \
	fi

#
# dyn:		$(newmat_lobj)
# 		$(CXX) -shared -o libnewmatWrap.so $(newmat_lobj)
# 
# wrapper:	hesstrans.i libnewmat.a
# 		swig -c++ -tcl hesstrans.i
# 		$(CXX) -shared -fPIC -DUSE_TCL_STUBS hesstrans_wrap.C $(newmat_lobj) -o libhessTrans.so -L. -ltcl8.4
# 


${ARCHDIR}/hesstrans.o:	hesstrans.C hesstrans.h newmatap.h newmatio.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/hesstrans.C $(COPTO)${ARCHDIR}/hesstrans.o

${ARCHDIR}/newmat1.o: newmat1.C newmat.h include.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/newmat1.C $(COPTO)${ARCHDIR}/newmat1.o

${ARCHDIR}/newmat2.o: newmat2.C include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/newmat2.C $(COPTO)${ARCHDIR}/newmat2.o

${ARCHDIR}/newmat3.o: newmat3.C include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/newmat3.C $(COPTO)${ARCHDIR}/newmat3.o

${ARCHDIR}/newmat4.o: newmat4.C include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/newmat4.C $(COPTO)${ARCHDIR}/newmat4.o

${ARCHDIR}/newmat5.o: newmat5.C include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/newmat5.C $(COPTO)${ARCHDIR}/newmat5.o

${ARCHDIR}/newmat6.o: newmat6.C include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/newmat6.C $(COPTO)${ARCHDIR}/newmat6.o

${ARCHDIR}/newmat7.o: newmat7.C include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/newmat7.C $(COPTO)${ARCHDIR}/newmat7.o

${ARCHDIR}/newmat8.o: newmat8.C include.h newmat.h newmatrc.h precisio.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/newmat8.C $(COPTO)${ARCHDIR}/newmat8.o

${ARCHDIR}/newmat9.o: newmat9.C include.h newmat.h newmatio.h newmatrc.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/newmat9.C $(COPTO)${ARCHDIR}/newmat9.o

${ARCHDIR}/newmatex.o: newmatex.C include.h newmat.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/newmatex.C $(COPTO)${ARCHDIR}/newmatex.o

${ARCHDIR}/bandmat.o: bandmat.C include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/bandmat.C $(COPTO)${ARCHDIR}/bandmat.o

${ARCHDIR}/submat.o: submat.C include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/submat.C $(COPTO)${ARCHDIR}/submat.o

${ARCHDIR}/myexcept.o: myexcept.C include.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/myexcept.C $(COPTO)${ARCHDIR}/myexcept.o

${ARCHDIR}/cholesky.o: cholesky.C include.h newmat.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/cholesky.C $(COPTO)${ARCHDIR}/cholesky.o

${ARCHDIR}/evalue.o: evalue.C include.h newmatap.h newmatrm.h precisio.h newmat.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/evalue.C $(COPTO)${ARCHDIR}/evalue.o

${ARCHDIR}/fft.o: fft.C include.h newmatap.h newmat.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/fft.C $(COPTO)${ARCHDIR}/fft.o

${ARCHDIR}/hholder.o: hholder.C include.h newmatap.h newmat.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/hholder.C $(COPTO)${ARCHDIR}/hholder.o

${ARCHDIR}/jacobi.o: jacobi.C include.h newmatap.h precisio.h newmatrm.h newmat.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/jacobi.C $(COPTO)${ARCHDIR}/jacobi.o

${ARCHDIR}/newfft.o: newfft.C newmatap.h newmat.h include.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/newfft.C $(COPTO)${ARCHDIR}/newfft.o

${ARCHDIR}/sort.o: sort.C include.h newmatap.h newmat.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/sort.C $(COPTO)${ARCHDIR}/sort.o

${ARCHDIR}/svd.o: svd.C include.h newmatap.h newmatrm.h precisio.h newmat.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/svd.C $(COPTO)${ARCHDIR}/svd.o

${ARCHDIR}/newmatrm.o: newmatrm.C newmat.h newmatrm.h include.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/newmatrm.C $(COPTO)${ARCHDIR}/newmatrm.o

${ARCHDIR}/newmatnl.o: newmatnl.C newmatap.h newmatnl.h newmat.h include.h boolean.h myexcept.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/newmatnl.C $(COPTO)${ARCHDIR}/newmatnl.o

${ARCHDIR}/solution.o: solution.C include.h boolean.h myexcept.h solution.h
	${CXX} ${CXXFLAGS} ${INCDIR} -c ${SRCDIR}/solution.C $(COPTO)${ARCHDIR}/solution.o

${ARCHDIR}/hesstrans_wrap.C : hesstrans.i
	swig -tcl8 -c++ -o hesstrans_wrap.C hesstrans.i

${ARCHDIR}/hesstrans_wrap.o : hesstrans_wrap.C
	${CXX} ${CXXFLAGS} ${TCLINC} ${INCDIR} -c ${SRCDIR}/hesstrans_wrap.C $(COPTO)${ARCHDIR}/hesstrans_wrap.o


