
%module hessTrans
%include tclsh.i
%{
#include "hesstrans.h"
%}

int getInternalHessian(double* doubleArray, int* intArray, double* hessianInternal, int numCartesians, int numBonds, int numAngles, int numDihedrals, int numImpropers);


// SWIG helper functions for Real and int arrays
%inline %{
Real *new_Real(int size) {
	return (Real *) malloc(size*sizeof(Real));
	//return new Real[size];
}
void delete_Real(Real *a) {
	free(a);
}
Real get_Real(Real *a, int index) {
	return a[index];
}
void set_Real(Real *a, int index, Real val) {
	a[index] = val;
}

int *new_int(int size) {
	if (size > 0) {
		return (int *) malloc(size*2*sizeof(int));
	}
	else {
		return NULL;
	}
	//return new int[size];
}
void delete_int(int *a) {
	free(a);
}
int get_int(int *a, int index) {
	return a[index];
}
void set_int(int *a, int index, int val) {
	a[index] = val;
}
void check_int(int *a, int size) {
	int i=0;
	for (; i<size; i++) {
		cout << a[i] << " ";
	}
	cout << "\n";
	cout << a << "\n";
	return;
}

double *new_double(int size) {
	if (size > 0) {
		return (double *) malloc(size*2*sizeof(double));
 	}
	else {
		return NULL;
	}
	//return new double[size];
}
void delete_double(double *a) {
	free(a);
}
double get_double(double *a, int index) {
	return a[index];
}
void set_double(double *a, int index, double val) {
	a[index] = val;
}
void check_double(double *a, int size) {
	int i=0;
	for (; i<size; i++) {
		cout << a[i] << " ";
	}
	cout << "\n";
	cout << a << "\n";
	return;
}	
%}
