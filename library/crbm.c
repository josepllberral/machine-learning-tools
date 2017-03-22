/*---------------------------------------------------------------------------*/
/* CONDITIONAL RESTRICTED BOLTZMANN MACHINES in C for R                      */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// Inspired by the implementations from:
// * David Buchaca   : https://github.com/davidbp/connectionist
// * Andrew Landgraf : https://www.r-bloggers.com/restricted-boltzmann-machines-in-r/
// * Graham Taylor   : http://www.uoguelph.ca/~gwtaylor/
// * Yusuke Sugomori : https://github.com/yusugomori/DeepLearning/

// References:
// * Approach based on Graham Taylor's CRBM:
//   http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/gwtaylor_nips.pdf

// Mocap data:
// * R converted version: (you should find it in in the same GIT)
// * Original file: http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/motion.mat
// * Data originally from Eugene Hsu, MIT. http://people.csail.mit.edu/ehsu/work/sig05stf/

// Compile using "R CMD SHLIB crbm.c"

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
	double** A;
	double** B;
	double** W;
	double* hbias;
	double* vbias;
	double** vel_A;
	double** vel_B;
	double** vel_W;
	double* vel_v;
	double* vel_h;
	int delay;
	int batch_size;
} CRBM;

/*---------------------------------------------------------------------------*/
/* AUXILIAR FUNCTIONS                                                        */
/*---------------------------------------------------------------------------*/

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
/* CRBM FUNCTIONS                                                            */
/*---------------------------------------------------------------------------*/

// Conditional Restricted Boltzmann Machine (CRBM). Constructor
void create_CRBM (CRBM* crbm, int N, int n_visible, int n_hidden, int delay, int batch_size,
		double** A, double** B, double** W, double* hbias, double* vbias)
{
	// Initialize Parameters
	crbm->N = N;
	crbm->n_visible = n_visible;
	crbm->n_hidden = n_hidden;
	crbm->delay = delay;
	crbm->batch_size = batch_size;

	// Initialize Matrices and Vectors
	if (W == NULL) crbm->W = matrix_normal(n_visible, n_hidden, 0, 1, 0.01);
	else crbm->W = W;

	if (B == NULL) crbm->B = matrix_normal(n_visible * delay, n_hidden, 0, 1, 0.01);
	else crbm->B = B;

	if (A == NULL) crbm->A = matrix_normal(n_visible * delay, n_visible, 0, 1, 0.01);
	else crbm->A = A;

	if (hbias == NULL) crbm->hbias = vector_zeros(n_hidden);
	else crbm->hbias = hbias;

	if (vbias == NULL) crbm->vbias = vector_zeros(n_visible);
	else crbm->vbias = vbias;

	// Initialize Velocity for Momentum
	crbm->vel_W = matrix_zeros(n_visible, n_hidden);
	crbm->vel_B = matrix_zeros(n_visible * delay, n_hidden);
	crbm->vel_A = matrix_zeros(n_visible * delay, n_visible);
	crbm->vel_h = vector_zeros(n_hidden);
	crbm->vel_v = vector_zeros(n_visible);
}

// Destructor of RBMs
void free_CRBM (CRBM* crbm)
{
	matrix_free(crbm->W, crbm->n_visible);
	matrix_free(crbm->B, crbm->n_visible * crbm->delay);
	matrix_free(crbm->A, crbm->n_visible * crbm->delay);
	matrix_free(crbm->vel_W, crbm->n_visible);
	matrix_free(crbm->vel_B, crbm->n_visible * crbm->delay);
	matrix_free(crbm->vel_A, crbm->n_visible * crbm->delay);

	free(crbm->hbias);
	free(crbm->vbias);
	free(crbm->vel_h);
	free(crbm->vel_v);
}

// This function infers state of hidden units given visible units
//   returns h_mean   : batch_size x hidden
//   returns h_sample : batch_size x hidden
// It performs the Prop-Up: h_means = sigmoid(v_sample * W + v_history * B + hbias)
//                          h_sample = bernoulli(h_means)
//   param v_sample : batch_size x visible
//   param W        : visible x hidden
//   param v_hist   : batch_size x (visible * delay)
//   param B        : (visible * delay) x hidden
//   returns retval : batch_size x hidden
void visible_state_to_hidden_probabilities (CRBM* crbm, double** v_sample, double** v_history, double*** h_mean, double*** h_sample)
{
	double size_of_hidden = sizeof(double) * crbm->n_hidden;
	double temp_mean[crbm->n_hidden];
	double temp_sample[crbm->n_hidden];
	for(int i = 0; i < crbm->batch_size; i++)
	{
		memset(temp_mean, 0, size_of_hidden);
		memset(temp_sample, 0, size_of_hidden);

		for(int j = 0; j < crbm->n_visible; j++)
			for(int k = 0; k < crbm->n_hidden; k++)
				temp_mean[k] += v_sample[i][j] * crbm->W[j][k];
		for(int j = 0; j < crbm->n_visible * crbm->delay; j++)
			for(int k = 0; k < crbm->n_hidden; k++)
				temp_mean[k] += v_history[i][j] * crbm->B[j][k];
		for(int j = 0; j < crbm->n_hidden; j++)
		{
			temp_mean[j] = sigmoid(temp_mean[j] + crbm->hbias[j]);
			if (temp_mean[j] >= 0 && temp_mean[j] <= 1 && (rand()/(RAND_MAX + 1.0)) <= temp_mean[j]) temp_sample[j] = 1;
		}

		memcpy((*h_mean)[i], temp_mean, size_of_hidden);
		memcpy((*h_sample)[i], temp_sample, size_of_hidden);
	}
}

// This function infers state of visible units given hidden units
//   returns v_mean   : batch_size x visible
//   returns v_sample : batch_size x visible
// It performs the Prop-Down: v_mean = v_sample = (h_sample * t(W) + v_history * A + vbias)
//   param h_sample : batch_size x hidden
//   param W        : visible x hidden
//   param v_hist   : batch_size x (visible * delay)
//   param A        : (visible * delay) x visible
//   returns retval : batch_size x visible
void hidden_state_to_visible_probabilities (CRBM* crbm, double** h_sample, double** v_history, double*** v_mean, double*** v_sample)
{
	double size_of_visible = sizeof(double) * crbm->n_visible;
	double temp_mean[crbm->n_visible];
	for(int i = 0; i < crbm->batch_size; i++)
	{
		memset(temp_mean, 0, size_of_visible);

		for(int j = 0; j < crbm->n_visible * crbm->delay; j++)
			for(int k = 0; k < crbm->n_visible; k++)
				temp_mean[k] += v_history[i][j] * crbm->A[j][k];
		for(int j = 0; j < crbm->n_visible; j++)
		{
			for(int k = 0; k < crbm->n_hidden; k++)
				temp_mean[j] += h_sample[i][k] * crbm->W[j][k];
			temp_mean[j] += crbm->vbias[j];
		}

		memcpy((*v_mean)[i], temp_mean, size_of_visible);
		memcpy((*v_sample)[i], temp_mean, size_of_visible);
	}
}

// This function implements one step of CD-k
//   param input      : matrix input from batch data (batch_size x visible)
//   param input_hist : matrix input_history from batch data (batch_size x (visible * delay))
//   param lr         : learning rate used to train the CRBM
//   param momentum   : value for momentum coefficient on learning
//   param k          : number of Gibbs steps to do in CD-k
double cdk_CRBM (CRBM* crbm, double** input, double** input_history, double lr, double momentum, int k)
{
	// compute positive phase (awake)
	double** ph_means = matrix_zeros(crbm->batch_size, crbm->n_hidden);
	double** ph_sample = matrix_zeros(crbm->batch_size, crbm->n_hidden);
	visible_state_to_hidden_probabilities (crbm, input, input_history, &ph_means, &ph_sample);

	// perform negative phase (asleep)
	double** nv_means = matrix_zeros(crbm->batch_size, crbm->n_visible);
	double** nv_sample = matrix_zeros(crbm->batch_size, crbm->n_visible);
	double** nh_means = matrix_zeros(crbm->batch_size, crbm->n_hidden);
	double** nh_sample = matrix_copy(ph_sample, crbm->batch_size, crbm->n_hidden);
	for (int step = 0; step < k; step++)
	{
		hidden_state_to_visible_probabilities (crbm, nh_sample, input_history, &nv_means, &nv_sample);
		visible_state_to_hidden_probabilities (crbm, nv_sample, input_history, &nh_means, &nh_sample);
	}

	// applies gradients on CRBM: Delta_W, Delta_A, Delta_B, Delta_h, Delta_v
	double** delta_W = delta_function_1(input, ph_means, nv_sample, nh_means, crbm->n_visible, crbm->batch_size, crbm->n_hidden);
	double** delta_B = delta_function_1(input_history, ph_means, input_history, nh_means, crbm->n_visible * crbm->delay, crbm->batch_size, crbm->n_hidden);
	double** delta_A = delta_function_1(input_history, input, input_history, nv_sample, crbm->n_visible * crbm->delay, crbm->batch_size, crbm->n_visible);
	double* delta_h = delta_function_2(ph_means, nh_means, crbm->batch_size, crbm->n_visible);
	double* delta_v = delta_function_2(input, nv_sample, crbm->batch_size, crbm->n_visible);

	double ratio = lr / crbm->batch_size;

	for(int i = 0; i < crbm->n_visible; i++)
	{
		for(int j = 0; j < crbm->n_hidden; j++)
		{
			crbm->vel_W[i][j] = crbm->vel_W[i][j] * momentum + delta_W[i][j] * ratio;
			crbm->W[i][j] += crbm->vel_W[i][j];
		}
		crbm->vel_v[i] = crbm->vel_v[i] * momentum + delta_v[i] * ratio;
		crbm->vbias[i] += crbm->vel_v[i];
	}

	for(int i = 0; i < crbm->n_visible * crbm->delay; i++)
	{
		for(int j = 0; j < crbm->n_hidden; j++)
		{
			crbm->vel_B[i][j] = crbm->vel_B[i][j] * momentum + delta_B[i][j] * ratio;
			crbm->B[i][j] += crbm->vel_B[i][j];
		}
		for(int j = 0; j < crbm->n_visible; j++)
		{
			crbm->vel_A[i][j] = crbm->vel_A[i][j] * momentum + delta_A[i][j] * ratio;
			crbm->A[i][j] += crbm->vel_A[i][j];
		}
	}

	for(int i = 0; i < crbm->n_hidden; i++)
	{
		crbm->vel_h[i] = crbm->vel_h[i] * momentum + delta_h[i] * ratio;
		crbm->hbias[i] += crbm->vel_h[i];
	}

	// approximation to the reconstruction error: sum over dimensions, mean over cases
	double recon = 0;
	for(int i = 0; i < crbm->batch_size; i++)
		for(int j = 0; j < crbm->n_visible; j++)
			recon += pow(input[i][j] - nv_means[i][j],2);
	recon /= crbm->batch_size;

	// free the used space
	matrix_free(ph_means, crbm->batch_size);
	matrix_free(ph_sample, crbm->batch_size);
	matrix_free(nv_means, crbm->batch_size);
	matrix_free(nv_sample, crbm->batch_size);
	matrix_free(nh_means, crbm->batch_size);
	matrix_free(nh_sample, crbm->batch_size);

	matrix_free(delta_W, crbm->n_visible);
	matrix_free(delta_B, crbm->n_visible * crbm->delay);
	matrix_free(delta_A, crbm->n_visible * crbm->delay);
	free(delta_v);
	free(delta_h);

	return recon;
}

/*---------------------------------------------------------------------------*/
/* HOW TO TRAIN YOUR CRBM                                                    */
/*---------------------------------------------------------------------------*/

// Function to train the CRBM
//   param batchdata       : loaded dataset (rows = examples, cols = features)
//   param batch_size      : size of a batch used to train the CRBM
//   param n_hidden        : number of hidden units in the CRBM
//   param training_epochs : number of epochs used for training the CRBM
//   param learning_rate   : learning rate used for training the CRBM
//   param momentum        : momentum weight used for training the CRBM
//   param delay           : number of observations in history window
void train_crbm (CRBM* crbm, double** batchdata, int* seqlen, int nseq,
		int nrow, int ncol, int batch_size, int n_hidden,
		int training_epochs, double learning_rate, double momentum,
		int delay, int rand_seed)
{
	srand(rand_seed);

	// Shuffle indices having sequences into account
	int bdi_len = 0;
	for(int i = 0; i < nseq; i++) bdi_len += seqlen[i] - delay;
	int* batchdataindex = (int*) malloc(sizeof(int) * bdi_len);

	int last = 0;
	int bdi_count = 0;
	for (int s = 0; s < nseq; s++)
	{
		int slen = seqlen[s];
		int* bdi = sequence(last + delay, last + slen);
		for (int i = 0; i < slen - delay; i++) batchdataindex[bdi_count + i] = bdi[i];
		last += slen;
		bdi_count += slen - delay;
	}

	int* permute = shuffle(bdi_len);
	int* permindex = (int*) malloc(sizeof(int) * bdi_len);
	for (int i = 0; i < bdi_len; i++) permindex[i] = batchdataindex[permute[i]];

	free(batchdataindex);
	free(permute);

	int n_train_batches = bdi_len / batch_size;
	int n_visible = ncol;

	// construct RBM
	create_CRBM (crbm, nrow, n_visible, n_hidden, delay, batch_size, NULL, NULL, NULL, NULL, NULL);

	// go through the training epochs and training set
	double mean_cost;
	for(int epoch = 0; epoch < training_epochs; epoch++)
	{
		mean_cost = 0;
		for(int batch_index = 0; batch_index < n_train_batches; batch_index++)
		{
			int idx_aux_ini = batch_index * batch_size;
			int idx_aux_fin = idx_aux_ini + batch_size;

			if (idx_aux_fin >= bdi_len) break;

			double** input = (double**) malloc(sizeof(double*) * batch_size);
			double** input_hist = (double**) malloc(sizeof(double) * batch_size);
			for (int i = 0; i < batch_size; i++)
			{
				int index = permindex[i + idx_aux_ini];

				input[i] = (double*) malloc(sizeof(double) * ncol);
				memcpy(input[i], batchdata[index], sizeof(double) * ncol);

				input_hist[i] = (double*) malloc(sizeof(double) * ncol * delay);
				for (int d = 0; d < delay; d++)
				{
					int d_pos = d * ncol;
					int i_pos = index - 1 - d;
					for (int j = 0; j < ncol; j++) input_hist[i][d_pos + j] = batchdata[i_pos][j];
				}
			}

			// get the cost and the gradient corresponding to one step of CD-k
			mean_cost += cdk_CRBM (crbm, input, input_hist, learning_rate, momentum, 1);

			matrix_free(input, batch_size);
			matrix_free(input_hist, batch_size);
		}
		mean_cost /= n_train_batches;
		if (epoch % 50 == 0) printf("Training epoch %d, cost is %f\n", epoch, mean_cost);
	}
	free(permindex);

	printf("Training epoch %d, cost is %f\n", training_epochs, mean_cost);
	return;
}

/*---------------------------------------------------------------------------*/
/* PREDICT AND RECONSTRUCT USING THE CRBM                                    */
/*---------------------------------------------------------------------------*/

// This function computes the activation of Vector V
//   param v       : vector to predict
//   param v_hist  : history matrix for conditioning
double* activation_vector_CRBM (CRBM* crbm, double* v, double* v_hist)
{
	double* activation = (double*) malloc(sizeof(double) * crbm->n_hidden);
	for (int i = 0; i < crbm->n_hidden; i++)
	{
		double pre_sigmoid_activation = crbm->hbias[i];
		for(int j = 0; j < crbm->n_visible; j++) pre_sigmoid_activation += crbm->W[j][i] * v[j];
		for(int j = 0; j < crbm->n_visible * crbm->delay; j++) pre_sigmoid_activation += crbm->B[j][i] * v_hist[j];
		activation[i] = sigmoid(pre_sigmoid_activation);
	}
	return activation;
}

// This function computes the activation of Matrix V
//   param v       : matrix to predict
//   Now we suppose that the input matrix is a time serie
double** activation_CRBM (CRBM* crbm, double** v, int nrow)
{
	double** activation = (double**) malloc(sizeof(double*) * nrow);
	double* v_hist = (double*) malloc(sizeof(double) * crbm->n_visible * crbm->delay);
	for (int i = 0; i < crbm->delay; i++) activation[i] = vector_zeros (crbm->delay);
	for (int i = crbm->delay; i < nrow; i++)
	{
		for (int j = 0; j < crbm->delay; j++)
			for (int k = 0; k < crbm->n_visible; k++)
				v_hist[j * crbm->n_visible + k] = v[i - j - 1][k];
		activation[i] = activation_vector_CRBM(crbm, v[i], v_hist);
	}
	free(v_hist);
	return activation;
}

// This function makes a reconstruction of Vector V
//   param v : vector to reconstruct
//   param v_hist  : history matrix for conditioning
double* reconstruct_vector_CRBM (CRBM* crbm, double* h, double* v_hist)
{
	double* reconstructed = (double*) malloc(sizeof(double) * crbm->n_visible);
	for (int i = 0; i < crbm->n_visible; i++)
	{
		double pre_sigmoid_activation = crbm->vbias[i];
		for(int j = 0; j < crbm->n_hidden; j++) pre_sigmoid_activation += crbm->W[i][j] * h[j];
		for(int j = 0; j < crbm->n_visible * crbm->delay; j++) pre_sigmoid_activation += crbm->A[j][i] * v_hist[j];
		reconstructed[i] = sigmoid(pre_sigmoid_activation);
	}
	return reconstructed;
}

// This function makes a reconstruction of Matrix V
//   param v : matrix to reconstruct
//   param v_hist  : history matrix for conditioning
double** reconstruct_CRBM (CRBM* crbm, double** v, int nrow)
{
	double** reconstruct = (double**) malloc(sizeof(double*) * nrow);
	double* v_hist = (double*) malloc(sizeof(double) * crbm->n_visible * crbm->delay);
	for (int i = 0; i < crbm->delay; i++) reconstruct[i] = vector_zeros (crbm->n_visible);
	for (int i = crbm->delay; i < nrow; i++)
	{
		for (int j = 0; j < crbm->delay; j++)
			for (int k = 0; k < crbm->n_visible; k++)
				v_hist[j * crbm->n_visible + k] = v[i - j - 1][k];
		double* h = activation_vector_CRBM (crbm, v[i], v_hist);
		reconstruct[i] = reconstruct_vector_CRBM(crbm, h, v_hist);
		free(h);
	}
	free(v_hist);
	return reconstruct;
}

/*---------------------------------------------------------------------------*/
/* FORECAST AND SIMULATION USING THE CRBM                                    */
/*---------------------------------------------------------------------------*/

// Construct the function that implements our persistent chain
double* sample_fn(CRBM* crbm, int n_gibbs, double** vis_sample, double*** v_history)
{
	int size_of_visible = sizeof(double) * crbm->n_visible;

	double** nv_means = matrix_zeros(1, crbm->n_visible);
	double** nv_sample = matrix_zeros(1, crbm->n_visible);
	double** nh_means = matrix_zeros(1, crbm->n_hidden);
	double** nh_sample = matrix_zeros(1, crbm->n_hidden);

	memcpy(nv_sample[0], *vis_sample, size_of_visible);

	for(int k = 0; k < n_gibbs; k++)
	{
		visible_state_to_hidden_probabilities (crbm, nv_sample, (*v_history), &nh_means, &nh_sample);
		hidden_state_to_visible_probabilities (crbm, nh_sample, (*v_history), &nv_means, &nv_sample);
	}

	// Add to updates the shared variable that takes care of our persistent chain
	memcpy(*vis_sample, nv_sample[0], crbm->n_visible);	

	for (int j = crbm->delay - 1; j > 0; j--)
		memcpy((*v_history)[j - 1], (*v_history)[j], size_of_visible);
	memcpy((*v_history)[0], *vis_sample, size_of_visible);

	// Prepare results
	double* retval = (double*) malloc(size_of_visible);
	memcpy(retval, nv_means[0], size_of_visible);

	matrix_free(nh_means, 1);
	matrix_free(nh_sample, 1);
	matrix_free(nv_means, 1);
	matrix_free(nv_sample, 1);

	return retval;
}

// Function to reproduce N samples from a time serie, given an input data and its history
//   param n_samples : number of samples to be simulated
//   param n_gibbs   : number of gibbs iterations
double** generate_samples(CRBM* crbm, double** sequence, int n_seq, int n_samples, int n_gibbs)
{
	int size_of_visible = sizeof(double) * crbm->n_visible;

	double* p_vis_chain = (double*) malloc(size_of_visible);
	memcpy(p_vis_chain, sequence[0], size_of_visible);

	double** p_history = (double**) malloc(sizeof(double*) * crbm->delay);
	for (int j = 0; j < crbm->delay; j++)
	{
		p_history[j] = (double*) malloc(size_of_visible);
		memcpy(p_history[j], sequence[j+1], size_of_visible);
	}

	double** generated_series = (double**) malloc(sizeof(double*) * n_samples);
	for (int t = 0; t < n_samples; t++)
		generated_series[t] = sample_fn(crbm, n_gibbs, &p_vis_chain, &p_history); 

	free(p_vis_chain);
	matrix_free(p_history, crbm->delay);

	return generated_series;
}

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
	double** train_X_p = malloc(sizeof(double*) * nrow);
	for (int i = 0; i < nrow; i++)
	{
		train_X_p[i] = malloc(sizeof(double) * ncol);
		for (int j = 0; j < ncol; j++) train_X_p[i][j] = RMATRIX(dataset,i,j);
	}

	int* seq_len_p = malloc(sizeof(int) * nseq);
	for (int i = 0; i < nseq; i++) seq_len_p[i] = RVECTORI(seqlen,i);

	// Perform Training
	CRBM crbm;
	train_crbm (&crbm, train_X_p, seq_len_p, nseq, nrow, ncol, basi, nhid, trep, lera, mome, dely, rase);

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
			REAL(VECTOR_ELT(retval, 3))[i * nhid + j] = crbm.W[i][j];
			REAL(VECTOR_ELT(retval, 8))[i * nhid + j] = crbm.vel_W[i][j];
		}
		REAL(VECTOR_ELT(retval, 7))[i] = crbm.vbias[i];
		REAL(VECTOR_ELT(retval, 12))[i] = crbm.vel_v[i];
	}

	SET_VECTOR_ELT(retval, 4, allocMatrix(REALSXP, ncol * dely, nhid));
	SET_VECTOR_ELT(retval, 5, allocMatrix(REALSXP, ncol * dely, ncol));
	SET_VECTOR_ELT(retval, 9, allocMatrix(REALSXP, ncol * dely, nhid));
	SET_VECTOR_ELT(retval, 10, allocMatrix(REALSXP, ncol * dely, ncol));
	for (int i = 0; i < ncol * dely; i++)
	{
		for (int j = 0; j < nhid; j++)
		{
			REAL(VECTOR_ELT(retval, 4))[i * nhid + j] = crbm.B[i][j];
			REAL(VECTOR_ELT(retval, 9))[i * nhid + j] = crbm.vel_B[i][j];
		}
		for (int j = 0; j < ncol; j++)
		{
			REAL(VECTOR_ELT(retval, 5))[i * ncol + j] = crbm.A[i][j];
			REAL(VECTOR_ELT(retval, 10))[i * ncol + j] = crbm.vel_A[i][j];
		}
	}

	SET_VECTOR_ELT(retval, 6, allocVector(REALSXP, nhid));
	SET_VECTOR_ELT(retval, 11, allocVector(REALSXP, nhid));
	for (int i = 0; i < nhid; i++)
	{
		REAL(VECTOR_ELT(retval, 6))[i] = crbm.hbias[i];
		REAL(VECTOR_ELT(retval, 11))[i] = crbm.vel_h[i];
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
	matrix_free(train_X_p, nrow);
	free(seq_len_p);
	free_CRBM(&crbm);

	return retval;
}

// Function to Re-assemble the CRBM
void reassemble_CRBM (CRBM* crbm, SEXP W_input, SEXP B_input, SEXP A_input,
	SEXP hbias_input, SEXP vbias_input, SEXP delay, int wrow, int wcol,
	int brow, int bcol, int arow, int acol, int nhid, int ncol, int nvis,
	int dely)
{
	double** W = malloc(sizeof(double*) * wrow);
	for (int i = 0; i < wrow; i++)
	{
		W[i] = malloc(sizeof(double) * wcol);
		for (int j = 0; j < wcol; j++) W[i][j] = RMATRIX(W_input,i,j);
	}

	double** B = malloc(sizeof(double*) * brow);
	for (int i = 0; i < brow; i++)
	{
		B[i] = malloc(sizeof(double) * bcol);
		for (int j = 0; j < bcol; j++) B[i][j] = RMATRIX(B_input,i,j);
	}

	double** A = malloc(sizeof(double*) * arow);
	for (int i = 0; i < arow; i++)
	{
		A[i] = malloc(sizeof(double) * acol);
		for (int j = 0; j < acol; j++) A[i][j] = RMATRIX(A_input,i,j);
	}

	double* hbias = malloc(sizeof(double) * nhid);
	for (int i = 0; i < nhid; i++) hbias[i] = RVECTOR(hbias_input,i);

	double* vbias = malloc(sizeof(double) * ncol);
	for (int i = 0; i < nvis; i++) vbias[i] = RVECTOR(vbias_input,i);

	create_CRBM (crbm, 0, nvis, nhid, dely, 1, A, B, W, hbias, vbias);
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

 	int wrow = INTEGER(GET_DIM(W_input))[0];
 	int wcol = INTEGER(GET_DIM(W_input))[1];

 	int brow = INTEGER(GET_DIM(B_input))[0];
 	int bcol = INTEGER(GET_DIM(B_input))[1];

 	int arow = INTEGER(GET_DIM(A_input))[0];
 	int acol = INTEGER(GET_DIM(A_input))[1];

	// Re-assemble the CRBM
	CRBM crbm;
	reassemble_CRBM (&crbm, W_input, B_input, A_input, hbias_input, vbias_input, delay,
		wrow, wcol, brow, bcol, arow, acol, nhid, ncol, nvis, dely);

	// Prepare Test Dataset
	double** test_X_p = malloc(sizeof(double*) * nrow);
	for (int i = 0; i < nrow; i++)
	{
		test_X_p[i] = malloc(sizeof(double) * ncol);
		for (int j = 0; j < ncol; j++) test_X_p[i][j] = RMATRIX(newdata,i,j);
	}

	// Pass through CRBM
	double** reconstruct_p = reconstruct_CRBM(&crbm, test_X_p, nrow);
	double** activation_p = activation_CRBM(&crbm, test_X_p, nrow);

	// Prepare Results
	SEXP retval = PROTECT(allocVector(VECSXP, 2));

	SET_VECTOR_ELT(retval, 0, allocMatrix(REALSXP, nrow, ncol));
	for (int i = 0; i < nrow; i++)
	{
		for (int j = 0; j < ncol; j++)
			REAL(VECTOR_ELT(retval, 0))[i * ncol + j] = reconstruct_p[i][j];
		free(reconstruct_p[i]);
	}
	free(reconstruct_p);

	SET_VECTOR_ELT(retval, 1, allocMatrix(REALSXP, nrow, nhid));
	for (int i = 0; i < nrow; i++)
	{
		for (int j = 0; j < nhid; j++)
			REAL(VECTOR_ELT(retval, 1))[i * nhid + j] = activation_p[i][j];
		free(activation_p[i]);
	}
	free(activation_p);

	SEXP nms = PROTECT(allocVector(STRSXP, 2));
	SET_STRING_ELT(nms, 0, mkChar("reconstruction"));
	SET_STRING_ELT(nms, 1, mkChar("activation"));

	setAttrib(retval, R_NamesSymbol, nms);
	UNPROTECT(2);

	// Free the structures and the CRBM
	matrix_free(test_X_p, nrow);
	free_CRBM(&crbm);

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

 	int wrow = INTEGER(GET_DIM(W_input))[0];
 	int wcol = INTEGER(GET_DIM(W_input))[1];

 	int brow = INTEGER(GET_DIM(B_input))[0];
 	int bcol = INTEGER(GET_DIM(B_input))[1];

 	int arow = INTEGER(GET_DIM(A_input))[0];
 	int acol = INTEGER(GET_DIM(A_input))[1];

	int nsamp = INTEGER_VALUE(n_samples);
	int ngibb = INTEGER_VALUE(n_gibbs);

	// Re-assemble the CRBM
	CRBM crbm;
	reassemble_CRBM (&crbm, W_input, B_input, A_input, hbias_input, vbias_input, delay,
		wrow, wcol, brow, bcol, arow, acol, nhid, ncol, nvis, dely);

	// Prepare Test Dataset
	double** data_p = malloc(sizeof(double*) * nrow);
	for (int i = 0; i < nrow; i++)
	{
		data_p[i] = malloc(sizeof(double) * ncol);
		for (int j = 0; j < ncol; j++) data_p[i][j] = RMATRIX(newdata,i,j);
	}

	// Pass through CRBM
	double** results = generate_samples(&crbm, data_p, nrow, nsamp, ngibb);

	// Prepare Results
	SEXP retval = PROTECT(allocVector(VECSXP, 1));

	SET_VECTOR_ELT(retval, 0, allocMatrix(REALSXP, nsamp, ncol));
	for (int i = 0; i < nsamp; i++)
	{
		for (int j = 0; j < ncol; j++)
			REAL(VECTOR_ELT(retval, 0))[i * ncol + j] = results[i][j];
		free(results[i]);
	}
	free(results);

	SEXP nms = PROTECT(allocVector(STRSXP, 1));
	SET_STRING_ELT(nms, 0, mkChar("generated"));

	setAttrib(retval, R_NamesSymbol, nms);
	UNPROTECT(2);

	// Free the structures and the CRBM
	matrix_free(data_p, nrow);
	free_CRBM(&crbm);

	return retval;
}

/*---------------------------------------------------------------------------*/
/* Main Function - Program Entry Point                                       */
/*---------------------------------------------------------------------------*/

// TODO - Find more adequate Main Example
int main (void)
{
	printf("START\n");

	// training data
	double train_X[8][6] = {
		{1, 1, 1, 0, 0, 0},
		{1, 0, 1, 0, 0, 0},
		{1, 1, 1, 0, 0, 0},
		{0, 0, 1, 1, 1, 0},
		{0, 0, 1, 0, 1, 0},
		{0, 0, 1, 1, 1, 0},
		{0, 1, 0, 0, 1, 0},
		{0, 0, 1, 1, 1, 1}
	};
	double** train_X_p = malloc(sizeof(double*) * 8);
	for (int i = 0; i < 8; i++) train_X_p[i] = train_X[i];

	int train_N = 8;

	// train the RBM
	int n_visible = 6;
	int n_hidden = 3;
	double learning_rate = 0.1;
	double momentum = 0.8;
	int training_epochs = 1000;
	int batch_size = 100;
	int delay = 2;

	int* seqlen = (int*) malloc(sizeof(int));
	seqlen[0] = 1;
	int nseq = 1;


	CRBM crbm;
	train_crbm (&crbm, train_X_p, seqlen, nseq, train_N, n_visible, batch_size,
			n_hidden, training_epochs, learning_rate,
			momentum, delay, 1234);

	free(train_X_p);

	// test data
	double test_X[6][6] = {
		{1, 1, 0, 0, 0, 0},
		{0, 0, 1, 1, 1, 0},
		{0, 0, 1, 0, 1, 0},
		{0, 0, 1, 1, 1, 0},
		{0, 1, 0, 0, 1, 0},
		{0, 0, 0, 1, 1, 0}
	};
	double** test_X_p = malloc(sizeof(double*) * 6);
	for (int i = 0; i < 6; i++) test_X_p[i] = test_X[i];

	int test_N = 6;

	// pass the test on the RBM
	double** reconstruct_p = reconstruct_CRBM(&crbm, test_X_p, test_N);

	double reconstruct[6][6];
	for (int i = 0; i < 6; i++)
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
	free_CRBM(&crbm);

	printf("FIN\n");
	return 0;
}

