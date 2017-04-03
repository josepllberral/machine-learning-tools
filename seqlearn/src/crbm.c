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

// Compile using "R CMD SHLIB crbm.c -lgsl -lgslcblas"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>

typedef struct {
	int N;
	int n_visible;
	int n_hidden;
	gsl_matrix* A;
	gsl_matrix* B;
	gsl_matrix* W;
	gsl_vector* hbias;
	gsl_vector* vbias;
	gsl_matrix* vel_A;
	gsl_matrix* vel_B;
	gsl_matrix* vel_W;
	gsl_vector* vel_v;
	gsl_vector* vel_h;
	int delay;
	int batch_size;
} CRBM;

/*---------------------------------------------------------------------------*/
/* AUXILIAR FUNCTIONS                                                        */
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

/*---------------------------------------------------------------------------*/
/* CRBM FUNCTIONS                                                            */
/*---------------------------------------------------------------------------*/

// Conditional Restricted Boltzmann Machine (CRBM). Constructor
void create_CRBM (CRBM* crbm, int N, int n_visible, int n_hidden, int delay, int batch_size,
		gsl_matrix* A, gsl_matrix* B, gsl_matrix* W, gsl_vector* hbias, gsl_vector* vbias)
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

	if (hbias == NULL) crbm->hbias = gsl_vector_calloc(n_hidden);
	else crbm->hbias = hbias;

	if (vbias == NULL) crbm->vbias = gsl_vector_calloc(n_visible);
	else crbm->vbias = vbias;

	// Initialize Velocity for Momentum
	crbm->vel_W = gsl_matrix_calloc(n_visible, n_hidden);
	crbm->vel_B = gsl_matrix_calloc(n_visible * delay, n_hidden);
	crbm->vel_A = gsl_matrix_calloc(n_visible * delay, n_visible);
	crbm->vel_h = gsl_vector_calloc(n_hidden);
	crbm->vel_v = gsl_vector_calloc(n_visible);
}

// Destructor of RBMs
void free_CRBM (CRBM* crbm)
{
	gsl_matrix_free(crbm->W);
	gsl_matrix_free(crbm->B);
	gsl_matrix_free(crbm->A);
	gsl_matrix_free(crbm->vel_W);
	gsl_matrix_free(crbm->vel_B);
	gsl_matrix_free(crbm->vel_A);

	gsl_vector_free(crbm->hbias);
	gsl_vector_free(crbm->vbias);
	gsl_vector_free(crbm->vel_h);
	gsl_vector_free(crbm->vel_v);
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
void visible_state_to_hidden_probabilities_crbm (CRBM* crbm, gsl_matrix* v_sample, gsl_matrix* v_history, gsl_matrix** h_mean, gsl_matrix** h_sample)
{
	int nrow = v_sample->size1;

	gsl_matrix* pre_sigmoid = gsl_matrix_calloc(nrow, crbm->n_hidden);
	int res1 = gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, v_sample, crbm->W, 1.0, pre_sigmoid);
	int res2 = gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, v_history, crbm->B, 1.0, pre_sigmoid);

	gsl_matrix * M_bias = gsl_matrix_calloc(nrow, crbm->n_hidden);
	for(int i = 0; i < nrow; i++) gsl_matrix_set_row(M_bias, i, crbm->hbias);
	int res3 = gsl_matrix_add(pre_sigmoid, M_bias);

	*h_mean = matrix_sigmoid(pre_sigmoid);
	*h_sample = matrix_bernoulli(*h_mean);

	gsl_matrix_free(M_bias);
	gsl_matrix_free(pre_sigmoid);
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
void hidden_state_to_visible_probabilities_crbm (CRBM* crbm, gsl_matrix* h_sample, gsl_matrix* v_history, gsl_matrix** v_mean, gsl_matrix** v_sample)
{
	int nrow = h_sample->size1;

	gsl_matrix* pre_sigmoid = gsl_matrix_calloc(nrow, crbm->n_visible);
	int res1 = gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, h_sample, crbm->W, 1.0, pre_sigmoid);
	int res2 = gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, v_history, crbm->A, 1.0, pre_sigmoid);

	gsl_matrix * M_bias = gsl_matrix_calloc(nrow, crbm->n_visible);
	for(int i = 0; i < nrow; i++) gsl_matrix_set_row(M_bias, i, crbm->vbias);
	int res3 = gsl_matrix_add(pre_sigmoid, M_bias);

	int res4 =  gsl_matrix_memcpy(*v_mean, pre_sigmoid);
	int res5 =  gsl_matrix_memcpy(*v_sample, pre_sigmoid);

	gsl_matrix_free(M_bias);
	gsl_matrix_free(pre_sigmoid);
}

// This function implements one step of CD-k
//   param input      : matrix input from batch data (batch_size x visible)
//   param input_hist : matrix input_history from batch data (batch_size x (visible * delay))
//   param lr         : learning rate used to train the CRBM
//   param momentum   : value for momentum coefficient on learning
//   param k          : number of Gibbs steps to do in CD-k
double cdk_CRBM (CRBM* crbm, gsl_matrix* input, gsl_matrix* input_history, double lr, double momentum, int k)
{
	// compute positive phase (awake)
	gsl_matrix* ph_means = gsl_matrix_calloc(crbm->batch_size, crbm->n_hidden);
	gsl_matrix* ph_sample = gsl_matrix_calloc(crbm->batch_size, crbm->n_hidden);
	visible_state_to_hidden_probabilities_crbm (crbm, input, input_history, &ph_means, &ph_sample);

	// perform negative phase (asleep)
	gsl_matrix* nv_means = gsl_matrix_calloc(crbm->batch_size, crbm->n_visible);
	gsl_matrix* nv_sample = gsl_matrix_calloc(crbm->batch_size, crbm->n_visible);
	gsl_matrix* nh_means = gsl_matrix_calloc(crbm->batch_size, crbm->n_hidden);
	gsl_matrix* nh_sample = gsl_matrix_calloc(crbm->batch_size, crbm->n_hidden);

	int res1P =  gsl_matrix_memcpy(nh_sample, ph_sample);
	for (int step = 0; step < k; step++)
	{
		hidden_state_to_visible_probabilities_crbm (crbm, nh_sample, input_history, &nv_means, &nv_sample);
		visible_state_to_hidden_probabilities_crbm (crbm, nv_sample, input_history, &nh_means, &nh_sample);
	}

	// applies gradients on CRBM: Delta_W, Delta_A, Delta_B, Delta_h, Delta_v

	double ratio = lr / crbm->batch_size;

	gsl_matrix* identity_h = gsl_matrix_alloc(crbm->n_hidden, crbm->n_hidden);
	gsl_matrix_set_all(identity_h, 1.0);
	gsl_matrix_set_identity(identity_h);

	gsl_matrix* identity_v = gsl_matrix_alloc(crbm->n_visible, crbm->n_visible);
	gsl_matrix_set_all(identity_v, 1.0);
	gsl_matrix_set_identity(identity_v);

	gsl_vector* ones = gsl_vector_alloc(crbm->batch_size);
	gsl_vector_set_all(ones, 1.0);

	// Compute and apply Delta_W
	gsl_matrix* delta_W = gsl_matrix_calloc(crbm->n_visible, crbm->n_hidden);
	int res1W = gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, input, ph_means, 1.0, delta_W);
	int res2W = gsl_blas_dgemm(CblasTrans, CblasNoTrans, -1.0, nv_sample, nh_means, 1.0, delta_W);
	int res3W = gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, ratio, delta_W, identity_h, momentum, crbm->vel_W);
	int res4W = gsl_matrix_add(crbm->W, crbm->vel_W);

	// Compute and apply Delta_B
	gsl_matrix* delta_B = gsl_matrix_calloc(crbm->n_visible * crbm->delay, crbm->n_hidden);
	int res1B = gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, input_history, ph_means, 1.0, delta_B);
	int res2B = gsl_blas_dgemm(CblasTrans, CblasNoTrans, -1.0, input_history, nh_means, 1.0, delta_B);
	int res3B = gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, ratio, delta_B, identity_h, momentum, crbm->vel_B);
	int res4B = gsl_matrix_add(crbm->B, crbm->vel_B);

	// Compute and apply Delta_A
	gsl_matrix* delta_A = gsl_matrix_calloc(crbm->n_visible * crbm->delay, crbm->n_visible);
	int res1A = gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, input_history, input, 1.0, delta_A);
	int res2A = gsl_blas_dgemm(CblasTrans, CblasNoTrans, -1.0, input_history, nv_sample, 1.0, delta_A);
	int res3A = gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, ratio, delta_A, identity_v, momentum, crbm->vel_A);
	int res4A = gsl_matrix_add(crbm->A, crbm->vel_A);

	gsl_matrix_free(delta_W);
	gsl_matrix_free(delta_B);
	gsl_matrix_free(delta_A);

	// Compute and apply Delta_v
	int res1V = gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, input, identity_v, -1.0, nv_sample); // we don't need nv_sample anymore
	int res2V = gsl_blas_dgemv(CblasTrans, ratio, nv_sample, ones, momentum, crbm->vel_v);
	int res3V = gsl_vector_add(crbm->vbias, crbm->vel_v);

	// Compute and apply Delta_h
	int res1H = gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, ph_means, identity_h, -1.0, nh_means); // we don't need nh_mean anymore
	int res2H = gsl_blas_dgemv(CblasTrans, ratio, nh_means, ones, momentum, crbm->vel_h);
	int res3H = gsl_vector_add(crbm->hbias, crbm->vel_h);


	// approximation to the reconstruction error: sum over dimensions, mean over cases
	int res1R = gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, input, identity_v, -1.0, nv_means); // we don't need nv_mean anymore
	gsl_matrix* pow_res = gsl_matrix_alloc(crbm->batch_size, crbm->n_visible);
	gsl_matrix_memcpy(pow_res, nv_means);
	int res2R = gsl_matrix_mul_elements(pow_res, nv_means);
	gsl_vector* pow_sum = gsl_vector_calloc(crbm->n_visible);
	int res3R = gsl_blas_dgemv(CblasTrans, 1.0, pow_res, ones, 1.0, pow_sum);

	double recon = 0;
	for(int j = 0; j < crbm->n_visible; j++)
		recon += gsl_vector_get(pow_sum, j);
	recon /= crbm->batch_size;

	gsl_matrix_free(pow_res);
	gsl_vector_free(pow_sum);

	// free the used space
	gsl_matrix_free(ph_means);
	gsl_matrix_free(ph_sample);
	gsl_matrix_free(nv_means);
	gsl_matrix_free(nv_sample);
	gsl_matrix_free(nh_means);
	gsl_matrix_free(nh_sample);

	gsl_matrix_free(identity_h);
	gsl_matrix_free(identity_v);
	gsl_vector_free(ones);

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
void train_crbm (CRBM* crbm, gsl_matrix* batchdata, int* seqlen, int nseq,
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

			gsl_matrix* input = gsl_matrix_alloc(crbm->batch_size, crbm->n_visible);
			gsl_matrix* input_hist = gsl_matrix_alloc(crbm->batch_size, crbm->n_visible * delay);

			for (int i = 0; i < batch_size; i++)
			{
				int index = permindex[i + idx_aux_ini];

				for (int j = 0; j < crbm->n_visible; j++)
					gsl_matrix_set(input, i, j, gsl_matrix_get(batchdata, index, j));

				for (int d = 0; d < delay; d++)
				{
					int i_pos = index - 1 - d;
					int d_pos = d * ncol;
					for (int j = 0; j < crbm->n_visible; j++)
						gsl_matrix_set(input_hist, i, d_pos + j, gsl_matrix_get(batchdata, i_pos, j));
				}
			}

			// get the cost and the gradient corresponding to one step of CD-k
			mean_cost += cdk_CRBM (crbm, input, input_hist, learning_rate, momentum, 1);

			gsl_matrix_free(input);
			gsl_matrix_free(input_hist);
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

// This function makes a reconstruction of Matrix V
//   param v_sample    : matrix to reconstruct from
//   param activations : matrix to store activations
//   param reconstruct : matrix to store reconstruction
void reconstruct_CRBM (CRBM* crbm, gsl_matrix* v_sample, gsl_matrix** activations, gsl_matrix** reconstruct)
{
	int nrow = v_sample->size1;

	// Recreate history
	gsl_matrix* pv_sample = gsl_matrix_calloc(nrow - crbm->delay, crbm->n_visible);
	gsl_matrix* v_history = gsl_matrix_calloc(nrow - crbm->delay, crbm->n_visible * crbm->delay);
	for (int i = crbm->delay; i < nrow; i++)
	{
		for (int j = 0; j < crbm->delay; j++)
			for (int k = 0; k < crbm->n_visible; k++)
				gsl_matrix_set(v_history, i - crbm->delay, j * crbm->n_visible + k, gsl_matrix_get(v_sample, i - j - 1, k));

		gsl_vector* aux = gsl_vector_alloc(crbm->n_visible);
		gsl_matrix_get_row(aux, v_sample, i);
		gsl_matrix_set_row(pv_sample, i - crbm->delay, aux);
		gsl_vector_free(aux);
	}

	// Activation and Reconstruction process
	gsl_matrix* nh_means = gsl_matrix_calloc(nrow - crbm->delay, crbm->n_hidden);
	gsl_matrix* nh_sample = gsl_matrix_calloc(nrow - crbm->delay, crbm->n_hidden);
	gsl_matrix* nv_means = gsl_matrix_calloc(nrow - crbm->delay, crbm->n_visible);
	gsl_matrix* nv_sample = gsl_matrix_calloc(nrow - crbm->delay, crbm->n_visible);

	visible_state_to_hidden_probabilities_crbm (crbm, pv_sample, v_history, &nh_means, &nh_sample);
	hidden_state_to_visible_probabilities_crbm (crbm, nh_sample, v_history, &nv_means, &nv_sample);

	// Copy results to activation and reconstruction matrix (and padding for delay)
	gsl_vector* vec_zeros_h = gsl_vector_calloc(crbm->n_hidden);
	gsl_vector* vec_zeros_v = gsl_vector_calloc(crbm->n_visible);
	for (int i = 0; i < crbm->delay; i++)
	{
		gsl_matrix_set_row(*reconstruct, i, vec_zeros_v);
		gsl_matrix_set_row(*activations, i, vec_zeros_h);
	}
	for (int i = crbm->delay; i < nrow; i++)
	{
		gsl_vector* aux_v = gsl_vector_alloc(crbm->n_visible);
		gsl_vector* aux_h = gsl_vector_alloc(crbm->n_hidden);
		gsl_matrix_get_row(aux_v, nv_sample, i - crbm->delay);
		gsl_matrix_get_row(aux_h, nh_means, i - crbm->delay);
		gsl_matrix_set_row(*reconstruct, i, aux_v);
		gsl_matrix_set_row(*activations, i, aux_h);
		gsl_vector_free(aux_v);
		gsl_vector_free(aux_h);
	}

	// Free auxiliar structures
	gsl_matrix_free(nh_means);
	gsl_matrix_free(nh_sample);
	gsl_matrix_free(nv_means);
	gsl_matrix_free(nv_sample);

	gsl_matrix_free(pv_sample);
	gsl_matrix_free(v_history);
	gsl_vector_free(vec_zeros_h);
	gsl_vector_free(vec_zeros_v);
}

/*---------------------------------------------------------------------------*/
/* FORECAST AND SIMULATION USING THE CRBM                                    */
/*---------------------------------------------------------------------------*/

// Construct the function that implements our persistent chain
//   param n_gibbs    : number of gibbs iterations
//   param vis_sample : visible sample vector
//   param v_history  : matrix with previous samples
gsl_vector* sample_fn(CRBM* crbm, int n_gibbs, gsl_vector** vis_sample, gsl_matrix** v_history)
{
	// Activation and Reconstruction process
	gsl_matrix* nh_means = gsl_matrix_calloc(1, crbm->n_hidden);
	gsl_matrix* nh_sample = gsl_matrix_calloc(1, crbm->n_hidden);
	gsl_matrix* nv_means = gsl_matrix_calloc(1, crbm->n_visible);
	gsl_matrix* nv_sample = gsl_matrix_calloc(1, crbm->n_visible);

	gsl_matrix_set_row(nv_sample, 0, *vis_sample);

	gsl_matrix* v_hist = gsl_matrix_alloc(1, crbm->n_visible * crbm->delay);
	for (int i = 0; i < crbm->delay; i++)
		for (int j = 0; j < crbm->n_visible; j++)
			gsl_matrix_set(v_hist, 0, i * crbm->n_visible + j, gsl_matrix_get(*v_history, i, j));

	for(int k = 0; k < n_gibbs; k++)
	{
		visible_state_to_hidden_probabilities_crbm (crbm, nv_sample, v_hist, &nh_means, &nh_sample);
		hidden_state_to_visible_probabilities_crbm (crbm, nh_sample, v_hist, &nv_means, &nv_sample);
	}

	// Add to updates the shared variable that takes care of our persistent chain
	gsl_vector* aux = gsl_vector_alloc(crbm->n_visible);
	gsl_matrix_get_row(aux, nv_sample, 0);
	gsl_vector_memcpy(*vis_sample, aux);
	gsl_vector_free(aux);	

	for (int j = crbm->delay - 2; j >= 0; j--)
	{
		gsl_vector* aux = gsl_vector_alloc(crbm->n_visible);
		gsl_matrix_get_row(aux, *v_history, j);
		gsl_matrix_set_row(*v_history, j + 1, aux);
		gsl_vector_free(aux);
	}
	gsl_matrix_set_row(*v_history, 0, *vis_sample);

	// Prepare results
	gsl_vector* retval = gsl_vector_alloc(crbm->n_visible);
	gsl_matrix_get_row(retval, nv_means, 0);

	// Free auxiliar structures
	gsl_matrix_free(nh_means);
	gsl_matrix_free(nh_sample);
	gsl_matrix_free(nv_means);
	gsl_matrix_free(nv_sample);

	return retval;
}

// Function to reproduce N samples from a time serie, given an input data and its history
//   param sequence  : matrix with ordered visible samples
//   param n_samples : number of samples to be simulated
//   param n_gibbs   : number of gibbs iterations
gsl_matrix* generate_samples(CRBM* crbm, gsl_matrix* sequence, int n_samples, int n_gibbs)
{
	int n_seq = sequence->size1;

	gsl_vector* p_vis_chain = gsl_vector_alloc(crbm->n_visible);
	gsl_matrix_get_row(p_vis_chain, sequence, crbm->delay);

	gsl_matrix* p_history = gsl_matrix_calloc(crbm->delay, crbm->n_visible);
	for (int d = 0; d < crbm->delay; d++)
	{
		gsl_vector* aux = gsl_vector_alloc(crbm->n_visible);
		gsl_matrix_get_row(aux, sequence, crbm->delay - d - 1);
		gsl_matrix_set_row(p_history, d, aux);
		gsl_vector_free(aux);
	}

	gsl_matrix* generated_series = gsl_matrix_calloc(n_samples, crbm->n_visible);
	for (int t = 0; t < n_samples; t++)
		gsl_matrix_set_row(generated_series, t, sample_fn(crbm, n_gibbs, &p_vis_chain, &p_history));

	gsl_vector_free(p_vis_chain);
	gsl_matrix_free(p_history);

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
	gsl_matrix* train_X_p = gsl_matrix_alloc(nrow, ncol);
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
			gsl_matrix_set(train_X_p, i, j, RMATRIX(dataset, i, j));


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
			REAL(VECTOR_ELT(retval, 3))[i * nhid + j] = gsl_matrix_get(crbm.W, i, j);
			REAL(VECTOR_ELT(retval, 8))[i * nhid + j] = gsl_matrix_get(crbm.vel_W, i, j);
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
			REAL(VECTOR_ELT(retval, 4))[i * nhid + j] = gsl_matrix_get(crbm.B, i, j);
			REAL(VECTOR_ELT(retval, 9))[i * nhid + j] = gsl_matrix_get(crbm.vel_B, i, j);
		}
		for (int j = 0; j < ncol; j++)
		{
			REAL(VECTOR_ELT(retval, 5))[i * ncol + j] = gsl_matrix_get(crbm.A, i, j);
			REAL(VECTOR_ELT(retval, 10))[i * ncol + j] = gsl_matrix_get(crbm.vel_A, i, j);
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

	// Free the structures and the CRBM
	free_CRBM(&crbm);

	gsl_matrix_free(reconstruction);
	gsl_matrix_free(activations);
	gsl_matrix_free(test_X_p);

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
			REAL(VECTOR_ELT(retval, 0))[i * ncol + j] = gsl_matrix_get(results, i, j);

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

