/*----------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C                                                */
/*----------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including:
//
// 1. Auxiliar Functions: images, evaluation, print matrices
// 2. Matrix Operations

#include "cnn.h"

/*---------------------------------------------------------------------------*/
/* AUXILIAR & EVALUATION FUNCTIONS                                           */
/*---------------------------------------------------------------------------*/

void replace_image(gsl_matrix**** destination, gsl_matrix**** newimage, int size1, int size2)
{
	if ((*destination))
	{
		for (int b = 0; b < size1; b++)
			if ((*destination)[b])
			{
				for (int c = 0; c < size2; c++)
					if ((*destination)[b][c]) gsl_matrix_free((*destination)[b][c]);
				free((*destination)[b]);
			}
		free((*destination));
	}
	int img_h = (*newimage)[0][0]->size1;
	int img_w = (*newimage)[0][0]->size2;

	(*destination) = (gsl_matrix***) malloc(size1 * sizeof(gsl_matrix**));
	for (int b = 0; b < size1; b++)
	{
		(*destination)[b] = (gsl_matrix**) malloc(size2 * sizeof(gsl_matrix*));
		for (int c = 0; c < size2; c++)
		{
			(*destination)[b][c] = gsl_matrix_alloc(img_h, img_w);
			gsl_matrix_memcpy((*destination)[b][c], (*newimage)[b][c]);
		}
	}
}

// Function to compute the Classification Accuracy
// param predicted : matrix with results. The "predicted result" will be the MAX
// param observed  : matrix with real values. Supposing that each row is one-hot-encoded
//                   the real value is also the MAX, with expected value 1.
double classification_accuracy (gsl_matrix* predicted, gsl_matrix* observed)
{
	int nrows = predicted->size1;
	int ncols = predicted->size2;

	int correct = 0;

	for (int i = 0; i < nrows; i++)
	{
		gsl_vector* prv = gsl_vector_alloc(ncols);
		gsl_vector* obv = gsl_vector_alloc(ncols);

		gsl_matrix_get_row(prv, predicted, i);
		gsl_matrix_get_row(obv, observed, i);

		int idx_maxobs = gsl_vector_max_index(prv);
		int idx_maxpred = gsl_vector_max_index(obv);

		if (idx_maxobs == idx_maxpred) correct++;

		gsl_vector_free(prv);
		gsl_vector_free(obv);
	}

	return ((double) correct / (double) nrows);
}

// Function to produce a Confusion Matrix
gsl_matrix* classification_matrix (gsl_matrix* predicted, gsl_matrix* observed)
{
	int nrows = predicted->size1;
	int ncols = predicted->size2;

	gsl_matrix* confusion = gsl_matrix_calloc(ncols, ncols);

	for (int i = 0; i < nrows; i++)
	{
		gsl_vector* prv = gsl_vector_alloc(ncols);
		gsl_vector* obv = gsl_vector_alloc(ncols);

		gsl_matrix_get_row(prv, predicted, i);
		gsl_matrix_get_row(obv, observed, i);

		int idx_maxobs = gsl_vector_max_index(obv);
		int idx_maxpred = gsl_vector_max_index(prv);

		double value = gsl_matrix_get(confusion, idx_maxobs, idx_maxpred);
		gsl_matrix_set(confusion, idx_maxobs, idx_maxpred, value + 1);

		gsl_vector_free(prv);
		gsl_vector_free(obv);
	}

	return confusion; // Oh my...!
}

// Function to print the Confusion Matrix
void classification_matrix_print (gsl_matrix* predicted, gsl_matrix* observed)
{
	int ncols = predicted->size2;

	gsl_matrix* confusion = classification_matrix(predicted, observed);

	printf("Observed VVV \\ Predicted >>>\n");
	for (int i = 0; i < ncols; i++)
	{
		for (int j = 0; j < ncols; j++)
			printf("%d ", (int)floor(gsl_matrix_get(confusion, i, j)));
		printf("\n");
	}
	printf("--------------------------------------\n");

	gsl_matrix_free(confusion);
}

// Function to print a GSL Matrix
void print_matrix (gsl_matrix* x)
{
	for (int i = 0; i < x->size1; i++)
	{
		for (int j = 0; j < x->size2; j++)
			printf("%f ", gsl_matrix_get(x, i, j));
		printf("\n");
	}
	printf("-------------\n");
}

// Function to print a frame in a 4D image
void print_image00 (gsl_matrix*** x, int a, int b)
{
	for (int i = 0; i < x[a][b]->size1; i++)
	{
		for (int j = 0; j < x[a][b]->size2; j++)
			printf("%f ", gsl_matrix_get(x[a][b], i, j));
		printf("\n");
	}
	printf("-------------\n");
}

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
