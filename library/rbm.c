/*---------------------------------------------------------------------------*/
/* RESTRICTED BOLTZMANN MACHINES in C for R                                  */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// Inspired by the implementations from:
// * David Buchaca   : https://github.com/davidbp/connectionist
// * Andrew Landgraf : https://www.r-bloggers.com/restricted-boltzmann-machines-in-r/
// * Graham Taylor   : http://www.uoguelph.ca/~gwtaylor/
// * Yusuke Sugomori : https://github.com/yusugomori/DeepLearning/

// Compile using "R CMD SHLIB rbm.c"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>

#include "matrix_ops.h"

typedef struct {
	int N;
	int n_visible;
	int n_hidden;
	double** W;
	double* hbias;
	double* vbias;
	double** vel_W;
	double* vel_v;
	double* vel_h;
	int batch_size;
} RBM;

/*---------------------------------------------------------------------------*/
/* AUXILIAR FUNCTIONS                                                        */
/*---------------------------------------------------------------------------*/

// Function to produce the Sigmoid
double sigmoid (double x)
{
	return 1.0 / (1.0 + exp(-x));
}

// This function implements the operation: t(A1) %*% A2 - t(B1) %*% B2
double** delta_function_1(double** A1, double** A2, double** B1, double** B2, int acol, int arow, int bcol)
{
	double temp[acol][bcol];
	memset(temp, 0, sizeof(double) * acol * bcol);
	for(int i = 0; i < arow; i++)
		for(int j = 0; j < acol; j++)
			for(int k = 0; k < bcol; k++)
				temp[j][k] += A1[i][j] * A2[i][k] - B1[i][j] * B2[i][k];

	int size_of_bcol = sizeof(double) * bcol;
	double** delta = (double**) malloc(sizeof(double*) * acol);
	for(int j = 0; j < acol; j++)
	{
		delta[j] = (double*) malloc(size_of_bcol);
		memcpy(delta[j], temp[j], size_of_bcol);
	}
	return delta;
}

// This function implements the operation: colmeans(A - B)
double* delta_function_2(double** A, double** B, int nrow, int ncol)
{
	double temp[ncol];
	memset(temp, 0, sizeof(double) * ncol);
	for(int i = 0; i < nrow; i++)
		for(int j = 0; j < ncol; j++)
			temp[j] += A[i][j] - B[i][j];

	double* delta = (double*) malloc(sizeof(double) * ncol);
	memcpy(delta, temp, sizeof(double) * ncol);

	return delta;
}

/*---------------------------------------------------------------------------*/
/* RBM FUNCTIONS                                                             */
/*---------------------------------------------------------------------------*/

// Restricted Boltzmann Machine (RBM). Constructor
void create_RBM (RBM* rbm, int N, int n_visible, int n_hidden, int batch_size,
		double** W, double* hbias, double* vbias)
{
	// Initialize Parameters
	rbm->N = N;
	rbm->n_visible = n_visible;
	rbm->n_hidden = n_hidden;
	rbm->batch_size = batch_size;

	// Initialize Matrices and Vectors
	if (W == NULL) rbm->W = matrix_normal(n_visible, n_hidden, 0, 1, 0.01);
	else rbm->W = W;

	if (hbias == NULL) rbm->hbias = vector_zeros(n_hidden);
	else rbm->hbias = hbias;

	if (vbias == NULL) rbm->vbias = vector_zeros(n_visible);
	else rbm->vbias = vbias;

	// Initialize Velocity for Momentum
	rbm->vel_W = matrix_zeros(n_visible, n_hidden);
	rbm->vel_h = vector_zeros(n_hidden);
	rbm->vel_v = vector_zeros(n_visible);
}

// Destructor of RBMs
void free_RBM (RBM* rbm)
{
	matrix_free(rbm->W, rbm->n_visible);
	matrix_free(rbm->vel_W, rbm->n_visible);

	free(rbm->hbias);
	free(rbm->vbias);
	free(rbm->vel_h);
	free(rbm->vel_v);
}

// This function infers state of hidden units given visible units
void visible_state_to_hidden_probabilities (RBM* rbm, double** v_sample, double*** h_mean, double*** h_sample)
{
	double size_of_hidden = sizeof(double) * rbm->n_hidden;
	double temp_mean[rbm->n_hidden];
	double temp_sample[rbm->n_hidden];
	for(int i = 0; i < rbm->batch_size; i++)
	{
		memset(temp_mean, 0, size_of_hidden);
		memset(temp_sample, 0, size_of_hidden);

		for(int j = 0; j < rbm->n_visible; j++)
			for(int k = 0; k < rbm->n_hidden; k++)
				temp_mean[k] += v_sample[i][j] * rbm->W[j][k];
		for(int j = 0; j < rbm->n_hidden; j++)
		{
			temp_mean[j] = sigmoid(temp_mean[j] + rbm->hbias[j]);
			if (temp_mean[j] >= 0 && temp_mean[j] <= 1 && (rand()/(RAND_MAX + 1.0)) <= temp_mean[j]) temp_sample[j] = 1;
		}

		memcpy((*h_mean)[i], temp_mean, size_of_hidden);
		memcpy((*h_sample)[i], temp_sample, size_of_hidden);
	}

}

// This function infers state of visible units given hidden units
void hidden_state_to_visible_probabilities (RBM* rbm, double** h_sample, double*** v_mean, double*** v_sample)
{
	double size_of_visible = sizeof(double) * rbm->n_visible;
	double temp_mean[rbm->n_visible];
	double temp_sample[rbm->n_visible];
	for(int i = 0; i < rbm->batch_size; i++)
	{
		memset(temp_mean, 0, size_of_visible);
		memset(temp_sample, 0, size_of_visible);

		for(int j = 0; j < rbm->n_visible; j++)
		{
			for(int k = 0; k < rbm->n_hidden; k++)
				temp_mean[j] += h_sample[i][k] * rbm->W[j][k];
			temp_mean[j] = sigmoid(temp_mean[j] + rbm->vbias[j]);
			if (temp_mean[j] >= 0 && temp_mean[j] <= 1 && (rand()/(RAND_MAX + 1.0)) <= temp_mean[j]) temp_sample[j] = 1;
		}

		memcpy((*v_mean)[i], temp_mean, size_of_visible);
		memcpy((*v_sample)[i], temp_sample, size_of_visible);
	}
}

// This functions implements one step of CD-k
//   param input    : matrix input from batch data (n_vis x n_seq)
//   param lr       : learning rate used to train the RBM
//   param momentum : value for momentum coefficient on learning
//   param k        : number of Gibbs steps to do in CD-k
double cdk_RBM (RBM* rbm, double** input, double lr, double momentum, int k)
{
	// compute positive phase (awake)
	double** ph_means = matrix_zeros(rbm->batch_size, rbm->n_hidden);
	double** ph_sample = matrix_zeros(rbm->batch_size, rbm->n_hidden);
	visible_state_to_hidden_probabilities (rbm, input, &ph_means, &ph_sample);

	// perform negative phase (asleep)
	double** nv_means = matrix_zeros(rbm->batch_size, rbm->n_visible);
	double** nv_sample = matrix_zeros(rbm->batch_size, rbm->n_visible);
	double** nh_means = matrix_zeros(rbm->batch_size, rbm->n_hidden);
	double** nh_sample = matrix_copy(ph_sample, rbm->batch_size, rbm->n_hidden);
	for (int step = 0; step < k; step++)
	{
		hidden_state_to_visible_probabilities (rbm, nh_sample, &nv_means, &nv_sample);
		visible_state_to_hidden_probabilities (rbm, nv_sample, &nh_means, &nh_sample);
	}

	// applies gradients on RBM: Delta_W, Delta_h, Delta_v
	double** delta_W = delta_function_1(input, ph_means, nv_sample, nh_means, rbm->n_visible, rbm->batch_size, rbm->n_hidden);
	double* delta_h = delta_function_2(ph_means, nh_means, rbm->batch_size, rbm->n_visible);
	double* delta_v = delta_function_2(input, nv_sample, rbm->batch_size, rbm->n_visible);

	double ratio = lr / rbm->batch_size;

	for(int i = 0; i < rbm->n_visible; i++)
	{
		for(int j = 0; j < rbm->n_hidden; j++)
		{
			rbm->vel_W[i][j] = rbm->vel_W[i][j] * momentum + delta_W[i][j] * ratio;
			rbm->W[i][j] += rbm->vel_W[i][j];
		}
		rbm->vel_v[i] = rbm->vel_v[i] * momentum + delta_v[i] * ratio;
		rbm->vbias[i] += rbm->vel_v[i];
	}

	for(int i = 0; i < rbm->n_hidden; i++)
	{
		rbm->vel_h[i] = rbm->vel_h[i] * momentum + delta_h[i] * ratio;
		rbm->hbias[i] += rbm->vel_h[i];
	}

	// approximation to the reconstruction error: sum over dimensions, mean over cases
	double recon = 0;
	for(int i = 0; i < rbm->batch_size; i++)
		for(int j = 0; j < rbm->n_visible; j++)
			recon += pow(input[i][j] - nv_means[i][j],2);
	recon /= rbm->batch_size;

	matrix_free(ph_means, rbm->batch_size);
	matrix_free(ph_sample, rbm->batch_size);
	matrix_free(nv_means, rbm->batch_size);
	matrix_free(nv_sample, rbm->batch_size);
	matrix_free(nh_means, rbm->batch_size);
	matrix_free(nh_sample, rbm->batch_size);

	matrix_free(delta_W, rbm->n_visible);
	free(delta_v);
	free(delta_h);

	return recon;
}

/*---------------------------------------------------------------------------*/
/* HOW TO TRAIN YOUR RBM                                                     */
/*---------------------------------------------------------------------------*/

// Function to train the RBM
//   param batchdata       : loaded dataset (rows = examples, cols = features)
//   param batch_size      : size of a batch used to train the RBM
//   param n_hidden        : number of hidden units in the RBM
//   param training_epochs : number of epochs used for training the RBM
//   param learning_rate   : learning rate used for training the RBM
//   param momentum        : momentum weight used for training the RBM
void train_rbm (RBM* rbm, double** batchdata, int nrow, int ncol, int batch_size,
                int n_hidden, int training_epochs, double learning_rate,
		double momentum, int rand_seed)
{
	srand(rand_seed);

	int n_train_batches = nrow / batch_size;
	int n_visible = ncol;

	// Shuffle Indices
	int* permindex = shuffle(nrow);

	// construct RBM
	create_RBM (rbm, nrow, n_visible, n_hidden, batch_size, NULL, NULL, NULL);

	// go through the training epochs and training set
	double mean_cost;
	for(int epoch = 0; epoch < training_epochs; epoch++)
	{
		mean_cost = 0;
		for(int batch_index = 0; batch_index < n_train_batches; batch_index++)
		{
			int idx_aux_ini = batch_index * batch_size;
			int idx_aux_fin = idx_aux_ini + batch_size;

			if (idx_aux_fin >= nrow) break;

			double** input = (double**) malloc(sizeof(double*) * batch_size);
			for (int i = 0; i < batch_size; i++)
			{
				int index = permindex[i + idx_aux_ini];

				input[i] = (double*) malloc(sizeof(double) * ncol);
				memcpy(input[i], batchdata[index], sizeof(double) * ncol);
			}

			// get the cost and the gradient corresponding to one step of CD-k
			mean_cost += cdk_RBM (rbm, input, learning_rate, momentum, 1);

			matrix_free(input, batch_size);
		}
		mean_cost /= batch_size;
		if (epoch % 100 == 0) printf("Training epoch %d, cost is %f\n", epoch, mean_cost);
	}
	free(permindex);

	printf("Training epoch %d, cost is %f\n", training_epochs, mean_cost);
	return;
}

/*---------------------------------------------------------------------------*/
/* PREDICT AND RECONSTRUCT USING THE RBM                                     */
/*---------------------------------------------------------------------------*/

// This function computes the activation of Vector V
//   param v       : vector to predict
double* activation_vector_RBM (RBM* rbm, double* v)
{
	double* activation = (double*) malloc(sizeof(double) * rbm->n_hidden);
	for (int i = 0; i < rbm->n_hidden; i++)
	{
		double pre_sigmoid_activation = rbm->hbias[i];
		for(int j = 0; j < rbm->n_visible; j++) pre_sigmoid_activation += rbm->W[j][i] * v[j];
		activation[i] = sigmoid(pre_sigmoid_activation);
	}
	return activation;
}

// This function computes the activation of Matrix V
//   param v : matrix to predict
double** activation_RBM (RBM* rbm, double** v, int nrow)
{
	double** activation = (double**) malloc(sizeof(double) * nrow);
	for (int i = 0; i < nrow; i++)
		activation[i] = activation_vector_RBM(rbm, v[i]);
	return activation;
}

// This function makes a reconstruction of Vector V
//   param v : vector to reconstruct
double* reconstruct_vector_RBM (RBM* rbm, double* h)
{
	double* reconstructed = (double*) malloc(sizeof(double) * rbm->n_visible);
	for (int i = 0; i < rbm->n_visible; i++)
	{
		double pre_sigmoid_activation = rbm->vbias[i];
		for(int j = 0; j < rbm->n_hidden; j++) pre_sigmoid_activation += rbm->W[i][j] * h[j];
		reconstructed[i] = sigmoid(pre_sigmoid_activation);
	}
	return reconstructed;
}

// This function makes a reconstruction of Matrix V
//   param v : matrix to reconstruct
double** reconstruct_RBM (RBM* rbm, double** v, int nrow)
{
	double** reconstruct = (double**) malloc(sizeof(double) * nrow);
	for (int i = 0; i < nrow; i++)
	{
		double* h = activation_vector_RBM (rbm, v[i]);
		reconstruct[i] = reconstruct_vector_RBM(rbm, h);
		free(h);
	}
	return reconstruct;
}

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
	double** train_X_p = malloc(sizeof(double*) * nrow);
	for (int i = 0; i < nrow; i++)
	{
		train_X_p[i] = malloc(sizeof(double) * ncol);
		for (int j = 0; j < ncol; j++) train_X_p[i][j] = RMATRIX(dataset,i,j);
	}

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
			REAL(VECTOR_ELT(retval, 3))[i * nhid + j] = rbm.W[i][j];
			REAL(VECTOR_ELT(retval, 6))[i * nhid + j] = rbm.vel_W[i][j];
		}
		REAL(VECTOR_ELT(retval, 5))[i] = rbm.vbias[i];
		REAL(VECTOR_ELT(retval, 8))[i] = rbm.vel_v[i];
	}

	SET_VECTOR_ELT(retval, 4, allocVector(REALSXP, nhid));
	SET_VECTOR_ELT(retval, 7, allocVector(REALSXP, nhid));
	for (int i = 0; i < nhid; i++)
	{
		REAL(VECTOR_ELT(retval, 4))[i] = rbm.hbias[i];
		REAL(VECTOR_ELT(retval, 7))[i] = rbm.vel_h[i];
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
	matrix_free(train_X_p, nrow);
	free_RBM(&rbm);

	return retval;
}

// Interface for Predicting and Reconstructing using an RBM
SEXP _C_RBM_predict (SEXP newdata, SEXP n_visible, SEXP n_hidden, SEXP W_input, SEXP hbias_input, SEXP vbias_input)
{
 	int nrow = INTEGER(GET_DIM(newdata))[0];
 	int ncol = INTEGER(GET_DIM(newdata))[1];

	int nhid = INTEGER_VALUE(n_hidden);

 	int wrow = INTEGER(GET_DIM(W_input))[0];
 	int wcol = INTEGER(GET_DIM(W_input))[1];

	// Re-assemble the RBM
	double** W = malloc(sizeof(double*) * wrow);
	for (int i = 0; i < wrow; i++)
	{
		W[i] = malloc(sizeof(double) * wcol);
		for (int j = 0; j < wcol; j++) W[i][j] = RMATRIX(W_input,i,j);
	}

	double* hbias = malloc(sizeof(double) * nhid);
	for (int i = 0; i < nhid; i++) hbias[i] = RVECTOR(hbias_input,i);

	double* vbias = malloc(sizeof(double) * ncol);
	for (int i = 0; i < ncol; i++) vbias[i] = RVECTOR(vbias_input,i);

	RBM rbm;
	create_RBM (&rbm, 0, ncol, nhid, 1, W, hbias, vbias);

	// Prepare Test Dataset
	double** test_X_p = malloc(sizeof(double*) * nrow);
	for (int i = 0; i < nrow; i++)
	{
		test_X_p[i] = malloc(sizeof(double) * ncol);
		for (int j = 0; j < ncol; j++) test_X_p[i][j] = RMATRIX(newdata,i,j);
	}



	// Pass through RBM
	double** reconstruct_p = reconstruct_RBM(&rbm, test_X_p, nrow);
	double** activation_p = activation_RBM(&rbm, test_X_p, nrow);

	// Prepare Results
	SEXP retval = PROTECT(allocVector(VECSXP, 2));

	SET_VECTOR_ELT(retval, 0, allocMatrix(REALSXP, nrow, ncol));
	SET_VECTOR_ELT(retval, 1, allocMatrix(REALSXP, nrow, nhid));
	for (int i = 0; i < nrow; i++)
	{
		for (int j = 0; j < ncol; j++)
			REAL(VECTOR_ELT(retval, 0))[i * ncol + j] = reconstruct_p[i][j];
		for (int j = 0; j < nhid; j++)
			REAL(VECTOR_ELT(retval, 1))[i * nhid + j] = activation_p[i][j];
		free(reconstruct_p[i]);
		free(activation_p[i]);
	}
	free(reconstruct_p);
	free(activation_p);

	SEXP nms = PROTECT(allocVector(STRSXP, 2));
	SET_STRING_ELT(nms, 0, mkChar("reconstruction"));
	SET_STRING_ELT(nms, 1, mkChar("activation"));

	setAttrib(retval, R_NamesSymbol, nms);
	UNPROTECT(2);

	// Free the structures and the RBM
	matrix_free(test_X_p, nrow);
	free_RBM(&rbm);

	return retval;
}

/*---------------------------------------------------------------------------*/
/* Main Function - Program Entry Point                                       */
/*---------------------------------------------------------------------------*/

int main (void)
{
	printf("START\n");

	// training data
	double train_X[6][6] = {
		{1, 1, 1, 0, 0, 0},
		{1, 0, 1, 0, 0, 0},
		{1, 1, 1, 0, 0, 0},
		{0, 0, 1, 1, 1, 0},
		{0, 0, 1, 0, 1, 0},
		{0, 0, 1, 1, 1, 0}
	};
	double** train_X_p = malloc(sizeof(double*) * 6);
	for (int i = 0; i < 6; i++) train_X_p[i] = train_X[i];

	int train_N = 6;

	// train the RBM
	int n_visible = 6;
	int n_hidden = 3;
	double learning_rate = 0.1;
	double momentum = 0.8;
	int training_epochs = 1000;
	int batch_size = 1;

	RBM rbm;
	train_rbm (&rbm, train_X_p, train_N, n_visible, batch_size,
                   n_hidden, training_epochs, learning_rate, momentum, 1234);

	free(train_X_p);

	// test data
	double test_X[2][6] = {
		{1, 1, 0, 0, 0, 0},
		{0, 0, 0, 1, 1, 0}
	};
	double** test_X_p = malloc(sizeof(double*) * 2);
	for (int i = 0; i < 2; i++) test_X_p[i] = test_X[i];

	int test_N = 2;

	// pass the test on the RBM
	double** reconstruct_p = reconstruct_RBM(&rbm, test_X_p, test_N);

	double reconstruct[2][6];
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 6; j++) reconstruct[i][j] = reconstruct_p[i][j];
		free(reconstruct_p[i]);
	}
	free(reconstruct_p);
	free(test_X_p);

	// print results
	for(int i = 0; i < test_N; i++)
	{
		for(int j = 0; j < n_visible; j++)
			printf("%.5f ", reconstruct[i][j]);
		printf("\n");
	}

	// free the RBM
	free_RBM(&rbm);

	printf("FIN\n");
	return 0;
}

