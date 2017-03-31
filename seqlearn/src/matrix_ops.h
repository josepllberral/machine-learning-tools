#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _MATRIX_OPS
#define _MATRIX_OPS

/*---------------------------------------------------------------------------*/
/* MATRIX OPERATIONS                                                         */
/*---------------------------------------------------------------------------*/
double** matrix_product (double** A, double** B, int arow, int common, int bcol);
double** matrix_summat (double** A, double** B, int nrow, int ncol);
double** matrix_subtract (double** A, double** B, int nrow, int ncol);
double** matrix_sumvec (double** A, double* B, int nrow, int ncol);
double** matrix_subtractvec (double** A, double* B, int nrow, int ncol);
double** matrix_scale (double** A, double B, int nrow, int ncol);
double** matrix_addition (double** A, double B, int nrow, int ncol);
double** matrix_sigma (double** A, int nrow, int ncol);
double** matrix_bernoulli (double** A, int nrow, int ncol);
double** matrix_sqdiff (double** A, double** B, int nrow, int ncol);
double* matrix_sumrows (double** A, int nrow, int ncol);
double* matrix_sumcols (double** A, int nrow, int ncol);
double* matrix_meancols (double** A, int nrow, int ncol);
double** matrix_normal(int nrow, int ncol, double mean, double stdev, double scale);
double** matrix_zeros (int nrow, int ncol);
double** matrix_transpose (double** A, int nrow, int ncol);
double** matrix_copy (double** A, int nrow, int ncol);
void matrix_free (double** A, int nrow);
void matrix_cfree (double** A);
void matrix_replace (double*** A, double** B, int nrow);

double* vector_sumvec (double* A, double* B, int npos);
double* vector_zeros (int npos);
double* vector_scale (double* V, double S, int npos);
void vector_replace (double** V, double* N);
double vector_mean (double* V, int npos);
int* sequence (int offset, int limit);
int* shuffle (int limit);

double sigmoid (double x);

#endif
