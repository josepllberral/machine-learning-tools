/*---------------------------------------------------------------------------*/
/* CONDITIONAL RESTRICTED BOLTZMANN MACHINES in C for R                      */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Function Implementations
// Compile using "gcc -c crbm.c matrix_ops.c -lgsl -lgslcblas -lm -o crbm"

#include "crbm.h"

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

	matrix_sigmoid(pre_sigmoid, *h_mean);
	matrix_bernoulli(*h_mean, *h_sample);

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
	recon /= crbm->n_visible;

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

		free(bdi);
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
/* MAIN FUNCTION - TEST                                                      */
/*---------------------------------------------------------------------------*/

/*
int main (void)
{
	printf("Start\n");

	int basi = 100;
	int nhid = 100;
	int dely = 6;
	int trep = 300;
	double lera = 1e-4;
	double mome = 0.8;
	int rase = 1234;

	int nrow = 3826;
	int ncol = 49;
	int nseq = 3;

	FILE * fp;
	char * line = NULL;
	size_t len = 0;

	// Read Batchdata
	fp = fopen("../datasets/motion_bd.data", "r");
	if (fp == NULL) exit(EXIT_FAILURE);

	gsl_matrix* train_X_p = gsl_matrix_alloc(nrow, ncol);
	for (int i = 0; i < nrow; i++)
	{
		ssize_t read = getline(&line, &len, fp);
		char *ch = strtok(line, " ");
		for (int j = 0; j < ncol; j++)
		{
			gsl_matrix_set(train_X_p, i, j, atof(ch));
			ch = strtok(NULL, " ");
		}
		free(ch);
	}
	fclose(fp);

	printf("Read Batch Data\n");

	// Read SeqLength
	fp = fopen("../datasets/motion_sl.data", "r");
	if (fp == NULL) exit(EXIT_FAILURE);

	int* seq_len_p = malloc(sizeof(int) * nseq);
	for (int i = 0; i < nseq; i++)
	{
		ssize_t read = getline(&line, &len, fp);
		seq_len_p[i] = atof(line);
	}
	fclose(fp);

	printf("Read Sequence Length\n");

	if (line) free(line);

	// Perform Training
	CRBM crbm;
	train_crbm (&crbm, train_X_p, seq_len_p, nseq, nrow, ncol, basi, nhid, trep, lera, mome, dely, rase);

	printf("Training Finished\n");

	// Perform Testing
	gsl_matrix* reconstruction = gsl_matrix_calloc(nrow, ncol);
	gsl_matrix* activations = gsl_matrix_calloc(nrow, nhid);
	reconstruct_CRBM(&crbm, train_X_p, &activations, &reconstruction);

	printf("Auto-validation Finished\n");

	// Check Error again
	double error_recons = 0;
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
		{
			double diff = gsl_matrix_get(train_X_p, i, j) - gsl_matrix_get(reconstruction, i, j);
			error_recons += pow(diff, 2);
		}
	printf("Error: %f\n", error_recons / (nrow * ncol));

	// Free Dataset Structure
	free(seq_len_p);
	free_CRBM(&crbm);

	gsl_matrix_free(train_X_p);
	gsl_matrix_free(reconstruction);
	gsl_matrix_free(activations);

	return 0;
}
*/
