#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#ifndef _MATRIX_OPS
#define _MATRIX_OPS 1

/*---------------------------------------------------------------------------*/
/* MATRIX OPERATIONS                                                         */
/*---------------------------------------------------------------------------*/
gsl_matrix* matrix_normal (int, int, double, double, double);
void matrix_sigmoid (gsl_matrix*, gsl_matrix*);
void matrix_exponent (gsl_matrix*, gsl_matrix*);
void matrix_bernoulli (gsl_matrix*, gsl_matrix*);
void matrix_log (gsl_matrix*, gsl_matrix*);
double matrix_sum (gsl_matrix*);
int* sequence (int, int);
int* shuffle (int);

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

#endif
