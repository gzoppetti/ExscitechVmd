CXX = g++
#CXXFLAGS = -O2 -Wall
CXXFLAGS = -O2 -Wall -fPIC

DIFF = ./sdiff
PRE = ./
MAJOR = 1
MINOR = 0

%.o:           	%.cpp
		$(CXX) $(CXXFLAGS) -c $*.cpp

everything:    	dyn wrapper

newmat_lobj = newmat1.o newmat2.o newmat3.o newmat4.o newmat5.o newmat6.o newmat7.o newmat8.o newmatex.o bandmat.o submat.o myexcept.o cholesky.o evalue.o fft.o hholder.o jacobi.o newfft.o sort.o svd.o newmatrm.o newmat9.o hessianTransform.o

libnewmat.a:   	$(newmat_lobj)
		$(AR) -cr $@ $(newmat_lobj)
		ranlib $@

dyn:		$(newmat_lobj)
		$(CXX) -shared -o libnewmatWrap.so $(newmat_lobj)

wrapper:	hessianTransform.i libnewmat.a
		swig -c++ -tcl hessianTransform.i
		$(CXX) -shared -fPIC -DUSE_TCL_STUBS hessianTransform_wrap.cxx $(newmat_lobj) -o libhessTrans.so -L. -ltcl8.4

hessianTransform.o:	hessianTransform.cpp hessianTransform.h newmatap.h newmatio.h

newmat1.o:     	newmat1.cpp newmat.h include.h boolean.h myexcept.h

newmat2.o:     	newmat2.cpp include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h

newmat3.o:     	newmat3.cpp include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h

newmat4.o:     	newmat4.cpp include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h

newmat5.o:     	newmat5.cpp include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h

newmat6.o:     	newmat6.cpp include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h

newmat7.o:     	newmat7.cpp include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h

newmat8.o:     	newmat8.cpp include.h newmat.h newmatrc.h precisio.h boolean.h myexcept.h controlw.h

newmatex.o:    	newmatex.cpp include.h newmat.h boolean.h myexcept.h

bandmat.o:     	bandmat.cpp include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h

submat.o:      	submat.cpp include.h newmat.h newmatrc.h boolean.h myexcept.h controlw.h

myexcept.o:    	myexcept.cpp include.h boolean.h myexcept.h

cholesky.o:    	cholesky.cpp include.h newmat.h boolean.h myexcept.h

evalue.o:      	evalue.cpp include.h newmatap.h newmatrm.h precisio.h newmat.h boolean.h myexcept.h

fft.o:         	fft.cpp include.h newmatap.h newmat.h boolean.h myexcept.h

hholder.o:     	hholder.cpp include.h newmatap.h newmat.h boolean.h myexcept.h

jacobi.o:      	jacobi.cpp include.h newmatap.h precisio.h newmatrm.h newmat.h boolean.h myexcept.h

newfft.o:      	newfft.cpp newmatap.h newmat.h include.h boolean.h myexcept.h

sort.o:        	sort.cpp include.h newmatap.h newmat.h boolean.h myexcept.h

svd.o:         	svd.cpp include.h newmatap.h newmatrm.h precisio.h newmat.h boolean.h myexcept.h

newmatrm.o:    	newmatrm.cpp newmat.h newmatrm.h include.h boolean.h myexcept.h

newmat9.o:     	newmat9.cpp include.h newmat.h newmatio.h newmatrc.h boolean.h myexcept.h controlw.h


newmatnl.o:    	newmatnl.cpp newmatap.h newmatnl.h newmat.h include.h boolean.h myexcept.h

solution.o:    	solution.cpp include.h boolean.h myexcept.h solution.h


clean:
	rm *.o *.a *.so
