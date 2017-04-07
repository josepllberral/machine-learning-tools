#include "matrix_ops.h"

/*---------------------------------------------------------------------------*/
/* MATRIX OPERATIONS                                                         */
/*---------------------------------------------------------------------------*/

// Function for Creating a Normal Matrix
//   param nrow   : 1st dimension of new Matrix
//   param ncol   : 2nd dimension of new Matrix
//   param mean   : Mean for Distribution
//   param stdev  : Standard Deviation for Distribution
//   param scale  : Scale for output values (multiplication factor)
gsl_matrix* matrix_normal (int nrow, int ncol, double mean, double stdev, double scale)
{
	gsl_matrix* N = gsl_matrix_alloc(nrow, ncol);
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
		{
			double rnd1 = (double) rand() / RAND_MAX;
			double rnd2 = (double) rand() / RAND_MAX;
			double rnum = mean + sqrt(-2 * log(rnd1)) * cos( 2 * M_PI * rnd2) * stdev;
			gsl_matrix_set(N, i, j, rnum * scale);
		}
	return N;
}

// Function for Sigma over Matrix
//   param A      : Target Matrix
gsl_matrix* matrix_sigmoid (gsl_matrix* M)
{
	int nrow = M->size1;
	int ncol = M->size2;

	gsl_matrix* sigm = gsl_matrix_calloc(nrow, ncol);
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
		{
			double s = 1.0 / (1.0 + exp(-1.0 * gsl_matrix_get(M, i, j)));
			gsl_matrix_set(sigm, i, j, s);
		}
	return sigm;
}

// Function for Bernoulli Sampling over Matrix
//   param M      : Target Matrix
gsl_matrix* matrix_bernoulli (gsl_matrix* M)
{
	int nrow = M->size1;
	int ncol = M->size2;

	gsl_matrix* bern = gsl_matrix_calloc(nrow, ncol);
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
		{
			double V = gsl_matrix_get(M, i, j);
			if (V >= 0 && V <= 1 && (rand() / (RAND_MAX + 1.0)) <= V) gsl_matrix_set(bern, i, j, 1.0);
		}

	return bern;
}

// Function to produce a sequence from 1 to limit
//   param offset : starting point
//   param limit  : length of the vector
int* sequence (int offset, int limit)
{
	int* vec = (int*) malloc(sizeof(int) * limit);
	for (int i = 0; i < limit; i++) vec[i] = offset + i;
	return vec;
}

// Function to produce a random shuffle from 1 to limit
//   param limit  : length of the vector
int* shuffle (int limit)
{
	int* vec = sequence(0, limit);
	if (limit > 1)
		for (int i = limit - 1; i > 0; i--)
		{
			int j = (int) (rand() / (RAND_MAX + 1.0) * i);
			int t = vec[j];
			vec[j] = vec[i];
			vec[i] = t;
		}

	return vec;
}
