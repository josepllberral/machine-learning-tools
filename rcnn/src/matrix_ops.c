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
	gsl_matrix* N = gsl_matrix_calloc(nrow, ncol);
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
//   param M      : Input Matrix
//   param S      : Results Matrix
void matrix_sigmoid (gsl_matrix* M, gsl_matrix* S)
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

	gsl_matrix_memcpy(S, sigm);
	gsl_matrix_free(sigm);

	return;
}

// Function for Natural Exponent over Matrix
//   param M      : Input Matrix
//   param E      : Results Matrix
void matrix_exponent (gsl_matrix* M, gsl_matrix* E)
{
	int nrow = M->size1;
	int ncol = M->size2;

	gsl_matrix* expo = gsl_matrix_calloc(nrow, ncol);
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
		{
			double e = exp(gsl_matrix_get(M, i, j));
			gsl_matrix_set(expo, i, j, e);
		}

	gsl_matrix_memcpy(E, expo);
	gsl_matrix_free(expo);

	return;
}

// Function for Bernoulli Sampling over Matrix
//   param M      : Input Matrix
//   param B      : Results Matrix
void matrix_bernoulli (gsl_matrix* M, gsl_matrix* B)
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

	gsl_matrix_memcpy(B, bern);
	gsl_matrix_free(bern);

	return;
}

// Function for Log over Matrix
//   param M      : Input Matrix
//   param L      : Results Matrix
void matrix_log (gsl_matrix* M, gsl_matrix* L)
{
	int nrow = M->size1;
	int ncol = M->size2;

	gsl_matrix* logm = gsl_matrix_calloc(nrow, ncol);
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
			gsl_matrix_set(logm, i, j, log(gsl_matrix_get(M, i, j)));

	gsl_matrix_memcpy(L, logm);
	gsl_matrix_free(logm);

	return;
}

// Function for Sum of Elements on Matrix
//   param M : Input Matrix
//   returns : sum of the matrix elements
double matrix_sum (gsl_matrix* M)
{
	int nrow = M->size1;
	int ncol = M->size2;

	double acc = 0;
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
			acc += gsl_matrix_get(M, i, j);
	return acc;
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

