/*---------------------------------------------------------------------------*/
/* RESTRICTED BOLTZMANN MACHINES in C for R                                  */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including R interface for RBMs
// Compile using "R CMD SHLIB rbm.c rbm_r.c matrix_ops.c -lgsl -lgslcblas -o rbm.so"

#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>

#include "rbm.h"

/*---------------------------------------------------------------------------*/
/* INTERFACE TO R                                                            */
/*---------------------------------------------------------------------------*/

#define RMATRIX(m,i,j) (REAL(m)[ INTEGER(GET_DIM(m))[0]*(j)+(i) ])
#define RVECTOR(v,i) (REAL(v)[(i)])

// Interface for Training an RBM
SEXP _C_RBM_train(SEXP dataset, SEXP batch_size, SEXP n_hidden, SEXP training_epochs,
           SEXP learning_rate, SEXP momentum, SEXP rand_seed)
{
 	int nrow = INTEGER(GET_DIM(dataset))[0];
 	int ncol = INTEGER(GET_DIM(dataset))[1];

 	int basi = INTEGER_VALUE(batch_size);
	int nhid = INTEGER_VALUE(n_hidden);
 	int trep = INTEGER_VALUE(training_epochs);
 	int rase = INTEGER_VALUE(rand_seed);
 	double lera = NUMERIC_VALUE(learning_rate);
 	double mome = NUMERIC_VALUE(momentum);

	// Create Dataset Structure
	gsl_matrix* train_X_p = gsl_matrix_alloc(nrow, ncol);
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
			gsl_matrix_set(train_X_p, i, j, RMATRIX(dataset, i, j));

	// Perform Training
	RBM rbm;
	train_rbm (&rbm, train_X_p, nrow, ncol, basi, nhid, trep, lera, mome, rase);

	// Return Structure
	SEXP retval = PROTECT(allocVector(VECSXP, 9));

	SET_VECTOR_ELT(retval, 0, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 0))[0] = rbm.N;

	SET_VECTOR_ELT(retval, 1, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 1))[0] = rbm.n_visible;

	SET_VECTOR_ELT(retval, 2, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 2))[0] = rbm.n_hidden;

	SET_VECTOR_ELT(retval, 3, allocMatrix(REALSXP, ncol, nhid));
	SET_VECTOR_ELT(retval, 5, allocVector(REALSXP, ncol));
	SET_VECTOR_ELT(retval, 6, allocMatrix(REALSXP, ncol, nhid));
	SET_VECTOR_ELT(retval, 8, allocVector(REALSXP, ncol));
	for (int i = 0; i < ncol; i++)
	{
		for (int j = 0; j < nhid; j++)
		{
			REAL(VECTOR_ELT(retval, 3))[i * nhid + j] = gsl_matrix_get(rbm.W, i, j);
			REAL(VECTOR_ELT(retval, 6))[i * nhid + j] = gsl_matrix_get(rbm.vel_W, i, j);
		}
		REAL(VECTOR_ELT(retval, 5))[i] = gsl_vector_get(rbm.vbias, i);
		REAL(VECTOR_ELT(retval, 8))[i] = gsl_vector_get(rbm.vel_v, i);
	}

	SET_VECTOR_ELT(retval, 4, allocVector(REALSXP, nhid));
	SET_VECTOR_ELT(retval, 7, allocVector(REALSXP, nhid));
	for (int i = 0; i < nhid; i++)
	{
		REAL(VECTOR_ELT(retval, 4))[i] = gsl_vector_get(rbm.hbias, i);
		REAL(VECTOR_ELT(retval, 7))[i] = gsl_vector_get(rbm.vel_h, i);
	}

	SEXP nms = PROTECT(allocVector(STRSXP, 9));
	SET_STRING_ELT(nms, 0, mkChar("N"));
	SET_STRING_ELT(nms, 1, mkChar("n_visible"));
	SET_STRING_ELT(nms, 2, mkChar("n_hidden"));
	SET_STRING_ELT(nms, 3, mkChar("W"));
	SET_STRING_ELT(nms, 4, mkChar("hbias"));
	SET_STRING_ELT(nms, 5, mkChar("vbias"));
	SET_STRING_ELT(nms, 6, mkChar("vel_W"));
	SET_STRING_ELT(nms, 7, mkChar("vel_h"));
	SET_STRING_ELT(nms, 8, mkChar("vel_v"));

	setAttrib(retval, R_NamesSymbol, nms);
	UNPROTECT(2);

	// Free Dataset Structure
	free_RBM(&rbm);

	gsl_matrix_free(train_X_p);

	return retval;
}

// Function to Re-assemble the RBM
void reassemble_RBM (RBM* rbm, SEXP W_input, SEXP hbias_input, SEXP vbias_input,
	int nhid, int nvis)
{
 	int wrow = INTEGER(GET_DIM(W_input))[0];
 	int wcol = INTEGER(GET_DIM(W_input))[1];

	gsl_matrix* W = gsl_matrix_alloc(wrow, wcol);
	for (int i = 0; i < wrow; i++)
		for (int j = 0; j < wcol; j++)
			gsl_matrix_set(W, i, j, RMATRIX(W_input, i, j));

	gsl_vector* hbias = gsl_vector_calloc(nhid);
	for (int i = 0; i < nhid; i++)
		gsl_vector_set(hbias, i, RVECTOR(hbias_input, i));

	gsl_vector* vbias = gsl_vector_calloc(nvis);
	for (int i = 0; i < nvis; i++)
		gsl_vector_set(vbias, i, RVECTOR(vbias_input, i));

	create_RBM (rbm, 0, nvis, nhid, 1, W, hbias, vbias);
}


// Interface for Predicting and Reconstructing using an RBM
SEXP _C_RBM_predict (SEXP newdata, SEXP n_visible, SEXP n_hidden, SEXP W_input, SEXP hbias_input, SEXP vbias_input)
{
 	int nrow = INTEGER(GET_DIM(newdata))[0];
 	int ncol = INTEGER(GET_DIM(newdata))[1];

	int nhid = INTEGER_VALUE(n_hidden);

	// Re-assemble the RBM
	RBM rbm;
	reassemble_RBM (&rbm, W_input, hbias_input, vbias_input, nhid, ncol);

	// Prepare Test Dataset
	gsl_matrix* test_X_p = gsl_matrix_alloc(nrow, ncol);
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
			gsl_matrix_set(test_X_p, i, j, RMATRIX(newdata, i, j));

	// Pass through RBM
	gsl_matrix* reconstruction = gsl_matrix_calloc(nrow, ncol);
	gsl_matrix* activations = gsl_matrix_calloc(nrow, nhid);
	reconstruct_RBM(&rbm, test_X_p, &activations, &reconstruction);

	// Prepare Results
	SEXP retval = PROTECT(allocVector(VECSXP, 2));

	SET_VECTOR_ELT(retval, 0, allocMatrix(REALSXP, nrow, ncol));
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
			REAL(VECTOR_ELT(retval, 0))[i * ncol + j] = gsl_matrix_get(reconstruction, i, j);

	SET_VECTOR_ELT(retval, 1, allocMatrix(REALSXP, nrow, nhid));
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < nhid; j++)
			REAL(VECTOR_ELT(retval, 1))[i * nhid + j] = gsl_matrix_get(activations, i, j);

	SEXP nms = PROTECT(allocVector(STRSXP, 2));
	SET_STRING_ELT(nms, 0, mkChar("reconstruction"));
	SET_STRING_ELT(nms, 1, mkChar("activation"));

	setAttrib(retval, R_NamesSymbol, nms);
	UNPROTECT(2);

	// Free the structures and the RBM
	free_RBM(&rbm);

	gsl_matrix_free(reconstruction);
	gsl_matrix_free(activations);
	gsl_matrix_free(test_X_p);

	return retval;
}

