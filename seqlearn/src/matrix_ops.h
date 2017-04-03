#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#ifndef _MATRIX_OPS
#define _MATRIX_OPS

// Compile: R CMD SHLIB crbm.c rbm.c matrix_ops.c -lgsl -lgslcblas -o librbm.so

/*---------------------------------------------------------------------------*/
/* MATRIX OPERATIONS                                                         */
/*---------------------------------------------------------------------------*/
gsl_matrix* matrix_normal (int, int, double, double, double);
gsl_matrix* matrix_sigmoid (gsl_matrix*);
gsl_matrix* matrix_bernoulli (gsl_matrix*);
int* sequence (int, int);
int* shuffle (int);

#endif
