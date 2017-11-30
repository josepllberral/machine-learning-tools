/*---------------------------------------------------------------------------*/
/* CONDITIONAL RESTRICTED BOLTZMANN MACHINES in C for R                      */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including R interface for CRBMs
// Compile using "R CMD SHLIB crbm.c crbm_r.c matrix_ops.c -lgsl -lgslcblas -o crbm.so"

#include "crbm.h"

#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>

/*---------------------------------------------------------------------------*/
/* INTERFACE TO R                                                            */
/*---------------------------------------------------------------------------*/

#define RMATRIX(m,i,j) (REAL(m)[ INTEGER(GET_DIM(m))[0]*(j)+(i) ])
#define RVECTOR(v,i) (REAL(v)[(i)])
#define RVECTORI(v,i) (INTEGER(v)[(i)])

// Interface for Training a CRBM
SEXP _C_CRBM_train(SEXP dataset, SEXP seqlen, SEXP n_seq, SEXP batch_size,
	SEXP n_hidden, SEXP training_epochs, SEXP learning_rate, SEXP momentum,
	SEXP delay, SEXP rand_seed)
{
 	int nrow = INTEGER(GET_DIM(dataset))[0];
 	int ncol = INTEGER(GET_DIM(dataset))[1];

 	int nseq = INTEGER_VALUE(n_seq);

 	int basi = INTEGER_VALUE(batch_size);
	int nhid = INTEGER_VALUE(n_hidden);
 	int trep = INTEGER_VALUE(training_epochs);
 	int rase = INTEGER_VALUE(rand_seed);
 	int dely = INTEGER_VALUE(delay);
 	double lera = NUMERIC_VALUE(learning_rate);
 	double mome = NUMERIC_VALUE(momentum);

	// Create Dataset Structure
	gsl_matrix* train_X_p = gsl_matrix_alloc(nrow, ncol);
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
			gsl_matrix_set(train_X_p, i, j, RMATRIX(dataset, i, j));

	int* seq_len_p = malloc(sizeof(int) * nseq);
	for (int i = 0; i < nseq; i++) seq_len_p[i] = RVECTORI(seqlen,i);

	// Perform Training
	CRBM crbm;
	train_crbm (&crbm, train_X_p, seq_len_p, nseq, nrow, ncol, basi, nhid, trep, lera, mome, dely, rase, NULL, NULL, NULL, NULL, NULL);

	// Return Structure
	SEXP retval = PROTECT(allocVector(VECSXP, 14));

	SET_VECTOR_ELT(retval, 0, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 0))[0] = crbm.N;

	SET_VECTOR_ELT(retval, 1, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 1))[0] = crbm.n_visible;

	SET_VECTOR_ELT(retval, 2, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 2))[0] = crbm.n_hidden;

	SET_VECTOR_ELT(retval, 3, allocMatrix(REALSXP, ncol, nhid));
	SET_VECTOR_ELT(retval, 7, allocVector(REALSXP, ncol));
	SET_VECTOR_ELT(retval, 8, allocMatrix(REALSXP, ncol, nhid));
	SET_VECTOR_ELT(retval, 12, allocVector(REALSXP, ncol));
	for (int i = 0; i < ncol; i++)
	{
		for (int j = 0; j < nhid; j++)
		{
			REAL(VECTOR_ELT(retval, 3))[j * ncol + i] = gsl_matrix_get(crbm.W, i, j);
			REAL(VECTOR_ELT(retval, 8))[j * ncol + i] = gsl_matrix_get(crbm.vel_W, i, j);
		}
		REAL(VECTOR_ELT(retval, 7))[i] = gsl_vector_get(crbm.vbias, i);
		REAL(VECTOR_ELT(retval, 12))[i] = gsl_vector_get(crbm.vel_v, i);
	}

	SET_VECTOR_ELT(retval, 4, allocMatrix(REALSXP, ncol * dely, nhid));
	SET_VECTOR_ELT(retval, 5, allocMatrix(REALSXP, ncol * dely, ncol));
	SET_VECTOR_ELT(retval, 9, allocMatrix(REALSXP, ncol * dely, nhid));
	SET_VECTOR_ELT(retval, 10, allocMatrix(REALSXP, ncol * dely, ncol));
	for (int i = 0; i < ncol * dely; i++)
	{
		for (int j = 0; j < nhid; j++)
		{
			REAL(VECTOR_ELT(retval, 4))[j * ncol * dely + i] = gsl_matrix_get(crbm.B, i, j);
			REAL(VECTOR_ELT(retval, 9))[j * ncol * dely + i] = gsl_matrix_get(crbm.vel_B, i, j);
		}
		for (int j = 0; j < ncol; j++)
		{
			REAL(VECTOR_ELT(retval, 5))[j * ncol * dely + i] = gsl_matrix_get(crbm.A, i, j);
			REAL(VECTOR_ELT(retval, 10))[j * ncol * dely + i] = gsl_matrix_get(crbm.vel_A, i, j);
		}
	}

	SET_VECTOR_ELT(retval, 6, allocVector(REALSXP, nhid));
	SET_VECTOR_ELT(retval, 11, allocVector(REALSXP, nhid));
	for (int i = 0; i < nhid; i++)
	{
		REAL(VECTOR_ELT(retval, 6))[i] = gsl_vector_get(crbm.hbias, i);
		REAL(VECTOR_ELT(retval, 11))[i] = gsl_vector_get(crbm.vel_h, i);
	}

	SET_VECTOR_ELT(retval, 13, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 13))[0] = crbm.delay;

	SEXP nms = PROTECT(allocVector(STRSXP, 14));
	SET_STRING_ELT(nms, 0, mkChar("N"));
	SET_STRING_ELT(nms, 1, mkChar("n_visible"));
	SET_STRING_ELT(nms, 2, mkChar("n_hidden"));
	SET_STRING_ELT(nms, 3, mkChar("W"));
	SET_STRING_ELT(nms, 4, mkChar("B"));
	SET_STRING_ELT(nms, 5, mkChar("A"));
	SET_STRING_ELT(nms, 6, mkChar("hbias"));
	SET_STRING_ELT(nms, 7, mkChar("vbias"));
	SET_STRING_ELT(nms, 8, mkChar("vel_W"));
	SET_STRING_ELT(nms, 9, mkChar("vel_B"));
	SET_STRING_ELT(nms, 10, mkChar("vel_A"));
	SET_STRING_ELT(nms, 11, mkChar("vel_h"));
	SET_STRING_ELT(nms, 12, mkChar("vel_v"));
	SET_STRING_ELT(nms, 13, mkChar("delay"));

	setAttrib(retval, R_NamesSymbol, nms);
	UNPROTECT(2);

	// Free Dataset Structure
	free(seq_len_p);
	free_CRBM(&crbm);

	gsl_matrix_free(train_X_p);

	return retval;
}


// Interface for Updating a CRBM (Code replica, because NULL parameters derp)
SEXP _C_CRBM_update(SEXP dataset, SEXP seqlen, SEXP n_seq, SEXP batch_size,
	SEXP n_hidden, SEXP training_epochs, SEXP learning_rate, SEXP momentum,
	SEXP delay, SEXP rand_seed, SEXP init_W, SEXP init_B, SEXP init_A,
	SEXP init_hbias, SEXP init_vbias)
{
 	int nrow = INTEGER(GET_DIM(dataset))[0];
 	int ncol = INTEGER(GET_DIM(dataset))[1];

 	int nseq = INTEGER_VALUE(n_seq);

 	int basi = INTEGER_VALUE(batch_size);
	int nhid = INTEGER_VALUE(n_hidden);
 	int trep = INTEGER_VALUE(training_epochs);
 	int rase = INTEGER_VALUE(rand_seed);
 	int dely = INTEGER_VALUE(delay);
 	double lera = NUMERIC_VALUE(learning_rate);
 	double mome = NUMERIC_VALUE(momentum);

	// Get Initial Values
	gsl_matrix* initial_A = gsl_matrix_alloc(ncol * dely, ncol);
	gsl_matrix* initial_B = gsl_matrix_alloc(ncol * dely, nhid);
	for (int i = 0; i < ncol * dely; i++)
	{
		for (int j = 0; j < ncol; j++)
			gsl_matrix_set(initial_A, i, j, RMATRIX(init_A, i, j));
		for (int j = 0; j < nhid; j++)
			gsl_matrix_set(initial_B, i, j, RMATRIX(init_B, i, j));
	}
			
	gsl_matrix* initial_W = gsl_matrix_alloc(ncol, nhid);
	gsl_vector* initial_vbias = gsl_vector_calloc(ncol);
	for (int i = 0; i < ncol; i++)
	{
		for (int j = 0; j < nhid; j++)
			gsl_matrix_set(initial_W, i, j, RMATRIX(init_W, i, j));
		gsl_vector_set(initial_vbias, i, RVECTOR(init_vbias, i));
	}
	
	gsl_vector* initial_hbias = gsl_vector_calloc(nhid);
	for (int i = 0; i < nhid; i++)
		gsl_vector_set(initial_hbias, i, RVECTOR(init_hbias, i));

	// Create Dataset Structure
	gsl_matrix* train_X_p = gsl_matrix_alloc(nrow, ncol);
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
			gsl_matrix_set(train_X_p, i, j, RMATRIX(dataset, i, j));

	int* seq_len_p = malloc(sizeof(int) * nseq);
	for (int i = 0; i < nseq; i++) seq_len_p[i] = RVECTORI(seqlen,i);

	// Perform Training
	CRBM crbm;
	train_crbm (&crbm, train_X_p, seq_len_p, nseq, nrow, ncol, basi, nhid, trep, lera, mome, dely, rase, initial_A, initial_B, initial_W, initial_hbias, initial_vbias);

	// Return Structure
	SEXP retval = PROTECT(allocVector(VECSXP, 14));

	SET_VECTOR_ELT(retval, 0, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 0))[0] = crbm.N;

	SET_VECTOR_ELT(retval, 1, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 1))[0] = crbm.n_visible;

	SET_VECTOR_ELT(retval, 2, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 2))[0] = crbm.n_hidden;

	SET_VECTOR_ELT(retval, 3, allocMatrix(REALSXP, ncol, nhid));
	SET_VECTOR_ELT(retval, 7, allocVector(REALSXP, ncol));
	SET_VECTOR_ELT(retval, 8, allocMatrix(REALSXP, ncol, nhid));
	SET_VECTOR_ELT(retval, 12, allocVector(REALSXP, ncol));
	for (int i = 0; i < ncol; i++)
	{
		for (int j = 0; j < nhid; j++)
		{
			REAL(VECTOR_ELT(retval, 3))[j * ncol + i] = gsl_matrix_get(crbm.W, i, j);
			REAL(VECTOR_ELT(retval, 8))[j * ncol + i] = gsl_matrix_get(crbm.vel_W, i, j);
		}
		REAL(VECTOR_ELT(retval, 7))[i] = gsl_vector_get(crbm.vbias, i);
		REAL(VECTOR_ELT(retval, 12))[i] = gsl_vector_get(crbm.vel_v, i);
	}

	SET_VECTOR_ELT(retval, 4, allocMatrix(REALSXP, ncol * dely, nhid));
	SET_VECTOR_ELT(retval, 5, allocMatrix(REALSXP, ncol * dely, ncol));
	SET_VECTOR_ELT(retval, 9, allocMatrix(REALSXP, ncol * dely, nhid));
	SET_VECTOR_ELT(retval, 10, allocMatrix(REALSXP, ncol * dely, ncol));
	for (int i = 0; i < ncol * dely; i++)
	{
		for (int j = 0; j < nhid; j++)
		{
			REAL(VECTOR_ELT(retval, 4))[j * ncol * dely + i] = gsl_matrix_get(crbm.B, i, j);
			REAL(VECTOR_ELT(retval, 9))[j * ncol * dely + i] = gsl_matrix_get(crbm.vel_B, i, j);
		}
		for (int j = 0; j < ncol; j++)
		{
			REAL(VECTOR_ELT(retval, 5))[j * ncol * dely + i] = gsl_matrix_get(crbm.A, i, j);
			REAL(VECTOR_ELT(retval, 10))[j * ncol * dely + i] = gsl_matrix_get(crbm.vel_A, i, j);
		}
	}

	SET_VECTOR_ELT(retval, 6, allocVector(REALSXP, nhid));
	SET_VECTOR_ELT(retval, 11, allocVector(REALSXP, nhid));
	for (int i = 0; i < nhid; i++)
	{
		REAL(VECTOR_ELT(retval, 6))[i] = gsl_vector_get(crbm.hbias, i);
		REAL(VECTOR_ELT(retval, 11))[i] = gsl_vector_get(crbm.vel_h, i);
	}

	SET_VECTOR_ELT(retval, 13, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 13))[0] = crbm.delay;

	SEXP nms = PROTECT(allocVector(STRSXP, 14));
	SET_STRING_ELT(nms, 0, mkChar("N"));
	SET_STRING_ELT(nms, 1, mkChar("n_visible"));
	SET_STRING_ELT(nms, 2, mkChar("n_hidden"));
	SET_STRING_ELT(nms, 3, mkChar("W"));
	SET_STRING_ELT(nms, 4, mkChar("B"));
	SET_STRING_ELT(nms, 5, mkChar("A"));
	SET_STRING_ELT(nms, 6, mkChar("hbias"));
	SET_STRING_ELT(nms, 7, mkChar("vbias"));
	SET_STRING_ELT(nms, 8, mkChar("vel_W"));
	SET_STRING_ELT(nms, 9, mkChar("vel_B"));
	SET_STRING_ELT(nms, 10, mkChar("vel_A"));
	SET_STRING_ELT(nms, 11, mkChar("vel_h"));
	SET_STRING_ELT(nms, 12, mkChar("vel_v"));
	SET_STRING_ELT(nms, 13, mkChar("delay"));

	setAttrib(retval, R_NamesSymbol, nms);
	UNPROTECT(2);

	// Free Dataset Structure
	free(seq_len_p);
	free_CRBM(&crbm);

	gsl_matrix_free(train_X_p);
	gsl_matrix_free(initial_A);
	gsl_matrix_free(initial_B);
	gsl_matrix_free(initial_W);
	gsl_vector_free(initial_hbias);
	gsl_vector_free(initial_vbias);

	return retval;
}

// Function to Re-assemble the CRBM
void reassemble_CRBM (CRBM* crbm, SEXP W_input, SEXP B_input, SEXP A_input,
	SEXP hbias_input, SEXP vbias_input, int nhid, int nvis,	int dely)
{
 	int wrow = INTEGER(GET_DIM(W_input))[0];
 	int wcol = INTEGER(GET_DIM(W_input))[1];

	gsl_matrix* W = gsl_matrix_alloc(wrow, wcol);
	for (int i = 0; i < wrow; i++)
		for (int j = 0; j < wcol; j++)
			gsl_matrix_set(W, i, j, RMATRIX(W_input, i, j));

 	int brow = INTEGER(GET_DIM(B_input))[0];
 	int bcol = INTEGER(GET_DIM(B_input))[1];

	gsl_matrix* B = gsl_matrix_alloc(brow, bcol);
	for (int i = 0; i < brow; i++)
		for (int j = 0; j < bcol; j++)
			gsl_matrix_set(B, i, j, RMATRIX(B_input, i, j));

 	int arow = INTEGER(GET_DIM(A_input))[0];
 	int acol = INTEGER(GET_DIM(A_input))[1];

	gsl_matrix* A = gsl_matrix_alloc(arow, acol);
	for (int i = 0; i < arow; i++)
		for (int j = 0; j < acol; j++)
			gsl_matrix_set(A, i, j, RMATRIX(A_input, i, j));

	gsl_vector* hbias = gsl_vector_calloc(nhid);
	for (int i = 0; i < nhid; i++)
		gsl_vector_set(hbias, i, RVECTOR(hbias_input, i));

	gsl_vector* vbias = gsl_vector_calloc(nvis);
	for (int i = 0; i < nvis; i++)
		gsl_vector_set(vbias, i, RVECTOR(vbias_input, i));

	create_CRBM (crbm, 0, nvis, nhid, dely, 1, A, B, W, hbias, vbias);
	
	gsl_matrix_free(A);
	gsl_matrix_free(B);
	gsl_matrix_free(W);
	gsl_vector_free(hbias);
	gsl_vector_free(vbias);
}

// Interface for Predicting and Reconstructing using a CRBM
SEXP _C_CRBM_predict (SEXP newdata, SEXP n_visible, SEXP n_hidden, SEXP W_input,
	SEXP B_input, SEXP A_input, SEXP hbias_input, SEXP vbias_input, SEXP delay)
{
	int nrow = INTEGER(GET_DIM(newdata))[0];
 	int ncol = INTEGER(GET_DIM(newdata))[1];

	int nvis = INTEGER_VALUE(n_visible);
	int nhid = INTEGER_VALUE(n_hidden);
	int dely = INTEGER_VALUE(delay);

	// Re-assemble the CRBM
	CRBM crbm;
	reassemble_CRBM (&crbm, W_input, B_input, A_input, hbias_input, vbias_input,
		nhid, nvis, dely);

	// Prepare Test Dataset
	gsl_matrix* test_X_p = gsl_matrix_alloc(nrow, ncol);
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
			gsl_matrix_set(test_X_p, i, j, RMATRIX(newdata, i, j));

	// Pass through CRBM
	gsl_matrix* reconstruction = gsl_matrix_calloc(nrow, ncol);
	gsl_matrix* activations = gsl_matrix_calloc(nrow, nhid);
	reconstruct_CRBM(&crbm, test_X_p, &activations, &reconstruction);

	// Prepare Results
	SEXP retval = PROTECT(allocVector(VECSXP, 2));

	SET_VECTOR_ELT(retval, 0, allocMatrix(REALSXP, nrow, ncol));
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
			REAL(VECTOR_ELT(retval, 0))[j * nrow + i] = gsl_matrix_get(reconstruction, i, j);

	SET_VECTOR_ELT(retval, 1, allocMatrix(REALSXP, nrow, nhid));
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < nhid; j++)
			REAL(VECTOR_ELT(retval, 1))[j * nrow + i] = gsl_matrix_get(activations, i, j);

	SEXP nms = PROTECT(allocVector(STRSXP, 2));
	SET_STRING_ELT(nms, 0, mkChar("reconstruction"));
	SET_STRING_ELT(nms, 1, mkChar("activation"));

	setAttrib(retval, R_NamesSymbol, nms);
	UNPROTECT(2);

	// Free the structures and the CRBM
	free_CRBM(&crbm);

	gsl_matrix_free(reconstruction);
	gsl_matrix_free(activations);
	gsl_matrix_free(test_X_p);

	return retval;
}

// Interface for Passing Forward an input dataset through a CRBM
SEXP _C_CRBM_forward (SEXP newdata, SEXP n_visible, SEXP n_hidden, SEXP W_input,
	SEXP B_input, SEXP A_input, SEXP hbias_input, SEXP vbias_input, SEXP delay)
{
	int nrow = INTEGER(GET_DIM(newdata))[0];
 	int ncol = INTEGER(GET_DIM(newdata))[1];

	int nvis = INTEGER_VALUE(n_visible);
	int nhid = INTEGER_VALUE(n_hidden);
	int dely = INTEGER_VALUE(delay);

	// Re-assemble the CRBM
	CRBM crbm;
	reassemble_CRBM (&crbm, W_input, B_input, A_input, hbias_input, vbias_input,
		nhid, nvis, dely);

	// Prepare Test Dataset
	gsl_matrix* test_X_p = gsl_matrix_alloc(nrow, ncol);
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
			gsl_matrix_set(test_X_p, i, j, RMATRIX(newdata, i, j));

	// Pass through CRBM
	gsl_matrix* activations = gsl_matrix_calloc(nrow, nhid);
	forward_CRBM(&crbm, test_X_p, &activations);

	// Prepare Results
	SEXP retval = PROTECT(allocMatrix(REALSXP, nrow, nhid));
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < nhid; j++)
			REAL(retval)[j * nrow + i] = gsl_matrix_get(activations, i, j);

	UNPROTECT(1);

	// Free the structures and the CRBM
	free_CRBM(&crbm);

	gsl_matrix_free(activations);
	gsl_matrix_free(test_X_p);

	return retval;
}

// Interface for Reconstructing Backwards an activation dataset through a CRBM
SEXP _C_CRBM_backward (SEXP newdata, SEXP history, SEXP n_visible, SEXP n_hidden, SEXP W_input,
	SEXP B_input, SEXP A_input, SEXP hbias_input, SEXP vbias_input, SEXP delay)
{
	int nrow = INTEGER(GET_DIM(newdata))[0];

	int nvis = INTEGER_VALUE(n_visible);
	int nhid = INTEGER_VALUE(n_hidden);
	int dely = INTEGER_VALUE(delay);

	// Re-assemble the CRBM
	CRBM crbm;
	reassemble_CRBM (&crbm, W_input, B_input, A_input, hbias_input, vbias_input,
		nhid, nvis, dely);

	// Prepare Test Dataset
	gsl_matrix* test_X_p = gsl_matrix_alloc(nrow, nhid);
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < nhid; j++)
			gsl_matrix_set(test_X_p, i, j, RMATRIX(newdata, i, j));

	// Prepare History Dataset
	gsl_matrix* test_X_h = gsl_matrix_alloc(nrow + dely - 1 , nvis * dely);
	for (int i = 0; i < nrow + dely - 1; i++)
		for (int j = 0; j < nvis * dely; j++)
			gsl_matrix_set(test_X_h, i, j, RMATRIX(history, i, j));

	// Pass through CRBM
	gsl_matrix* reconstruction = gsl_matrix_calloc(nrow, nvis);
	backward_CRBM(&crbm, test_X_p, test_X_h, &reconstruction);

	// Prepare Results
	SEXP retval = PROTECT(allocMatrix(REALSXP, nrow, nvis));
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < nvis; j++)
			REAL(retval)[j * nrow + i] = gsl_matrix_get(reconstruction, i, j);

	UNPROTECT(1);

	// Free the structures and the CRBM
	free_CRBM(&crbm);

	gsl_matrix_free(reconstruction);
	gsl_matrix_free(test_X_p);
	gsl_matrix_free(test_X_h);

	return retval;
}

// Interface for Generating a Sequence using a CRBM
SEXP _C_CRBM_generate_samples (SEXP newdata, SEXP n_visible, SEXP n_hidden,
	SEXP W_input, SEXP B_input, SEXP A_input, SEXP hbias_input,
	SEXP vbias_input, SEXP delay, SEXP n_samples, SEXP n_gibbs)
{
	int nrow = INTEGER(GET_DIM(newdata))[0];
 	int ncol = INTEGER(GET_DIM(newdata))[1];

	int nvis = INTEGER_VALUE(n_visible);
	int nhid = INTEGER_VALUE(n_hidden);
	int dely = INTEGER_VALUE(delay);

	int nsamp = INTEGER_VALUE(n_samples);
	int ngibb = INTEGER_VALUE(n_gibbs);

	// Re-assemble the CRBM
	CRBM crbm;
	reassemble_CRBM (&crbm, W_input, B_input, A_input, hbias_input, vbias_input,
		nhid, nvis, dely);

	// Prepare Test Dataset
	gsl_matrix* data_p = gsl_matrix_alloc(nrow, ncol);
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
			gsl_matrix_set(data_p, i, j, RMATRIX(newdata, i, j));

	// Pass through CRBM
	gsl_matrix* results = generate_samples(&crbm, data_p, nsamp, ngibb);

	// Prepare Results
	SEXP retval = PROTECT(allocVector(VECSXP, 1));

	SET_VECTOR_ELT(retval, 0, allocMatrix(REALSXP, nsamp, ncol));
	for (int i = 0; i < nsamp; i++)
		for (int j = 0; j < ncol; j++)
			REAL(VECTOR_ELT(retval, 0))[j * nsamp + i] = gsl_matrix_get(results, i, j);

	SEXP nms = PROTECT(allocVector(STRSXP, 1));
	SET_STRING_ELT(nms, 0, mkChar("generated"));

	setAttrib(retval, R_NamesSymbol, nms);
	UNPROTECT(2);

	// Free the structures and the CRBM
	gsl_matrix_free(data_p);
	gsl_matrix_free(results);
	free_CRBM(&crbm);

	return retval;
}

