/*---------------------------------------------------------------------------*/
/* RESTRICTED BOLTZMANN MACHINES in C                                        */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Function Implementations
// Compile using "gcc -c rbm.c matrix_ops.c -lgsl -lgslcblas -lm -o rbm"

#include "rbm.h"

/*---------------------------------------------------------------------------*/
/* RBM FUNCTIONS                                                             */
/*---------------------------------------------------------------------------*/

// Restricted Boltzmann Machine (RBM). Constructor
void create_RBM (RBM* rbm, int N, int n_visible, int n_hidden, int batch_size,
		gsl_matrix* W, gsl_vector* hbias, gsl_vector* vbias)
{
	// Initialize Parameters
	rbm->N = N;
	rbm->n_visible = n_visible;
	rbm->n_hidden = n_hidden;
	rbm->batch_size = batch_size;

	// Initialize Matrices and Vectors
	if (W == NULL) rbm->W = matrix_normal(n_visible, n_hidden, 0, 1, 0.01);
	else { rbm->W = gsl_matrix_calloc(n_visible, n_hidden); gsl_matrix_memcpy(rbm->W, W); }

	rbm->hbias = gsl_vector_calloc(n_hidden);
	if (hbias != NULL) gsl_vector_memcpy(rbm->hbias, hbias);

	rbm->vbias = gsl_vector_calloc(n_visible);
	if (vbias != NULL) gsl_vector_memcpy(rbm->vbias, vbias);

	// Initialize Velocity for Momentum
	rbm->vel_W = gsl_matrix_calloc(n_visible, n_hidden);
	rbm->vel_h = gsl_vector_calloc(n_hidden);
	rbm->vel_v = gsl_vector_calloc(n_visible);
}

// Destructor of RBMs
void free_RBM (RBM* rbm)
{
	gsl_matrix_free(rbm->W);
	gsl_matrix_free(rbm->vel_W);

	gsl_vector_free(rbm->hbias);
	gsl_vector_free(rbm->vbias);
	gsl_vector_free(rbm->vel_h);
	gsl_vector_free(rbm->vel_v);
}


// This function infers state of hidden units given visible units
//   returns h_mean   : batch_size x hidden
//   returns h_sample : batch_size x hidden
// It performs the Prop-Up: h_means = sigmoid(v_sample * W + hbias)
//                          h_sample = bernoulli(h_means)
//   param v_sample : batch_size x visible
//   param W        : visible x hidden
//   returns retval : batch_size x hidden
void visible_state_to_hidden_probabilities_rbm (RBM* rbm, gsl_matrix* v_sample, gsl_matrix** h_mean, gsl_matrix** h_sample)
{
	int nrow = v_sample->size1;

	gsl_matrix* pre_sigmoid = gsl_matrix_calloc(nrow, rbm->n_hidden);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, v_sample, rbm->W, 1.0, pre_sigmoid);

	gsl_matrix * M_bias = gsl_matrix_calloc(nrow, rbm->n_hidden);
	for(int i = 0; i < nrow; i++) gsl_matrix_set_row(M_bias, i, rbm->hbias);
	gsl_matrix_add(pre_sigmoid, M_bias);

	matrix_sigmoid(pre_sigmoid, *h_mean);
	matrix_bernoulli(*h_mean, *h_sample);

	gsl_matrix_free(M_bias);
	gsl_matrix_free(pre_sigmoid);
}

// This function infers state of visible units given hidden units
//   returns v_mean   : batch_size x visible
//   returns v_sample : batch_size x visible
// It performs the Prop-Down: v_mean = v_sample = (h_sample * t(W) + vbias)
//   param h_sample : batch_size x hidden
//   param W        : visible x hidden
//   returns retval : batch_size x visible
void hidden_state_to_visible_probabilities_rbm (RBM* rbm, gsl_matrix* h_sample, gsl_matrix** v_mean, gsl_matrix** v_sample)
{
	int nrow = h_sample->size1;

	gsl_matrix* pre_sigmoid = gsl_matrix_calloc(nrow, rbm->n_visible);
	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, h_sample, rbm->W, 1.0, pre_sigmoid);

	gsl_matrix * M_bias = gsl_matrix_calloc(nrow, rbm->n_visible);
	for(int i = 0; i < nrow; i++) gsl_matrix_set_row(M_bias, i, rbm->vbias);
	gsl_matrix_add(pre_sigmoid, M_bias);

	gsl_matrix_memcpy(*v_mean, pre_sigmoid);
	gsl_matrix_memcpy(*v_sample, pre_sigmoid);

	gsl_matrix_free(M_bias);
	gsl_matrix_free(pre_sigmoid);
}

// This function implements one step of CD-k
//   param input      : matrix input from batch data (batch_size x visible)
//   param lr         : learning rate used to train the RBM
//   param momentum   : value for momentum coefficient on learning
//   param k          : number of Gibbs steps to do in CD-k
double cdk_RBM (RBM* rbm, gsl_matrix* input, double lr, double momentum, int k)
{
	// compute positive phase (awake)
	gsl_matrix* ph_means = gsl_matrix_calloc(rbm->batch_size, rbm->n_hidden);
	gsl_matrix* ph_sample = gsl_matrix_calloc(rbm->batch_size, rbm->n_hidden);
	visible_state_to_hidden_probabilities_rbm (rbm, input, &ph_means, &ph_sample);

	// perform negative phase (asleep)
	gsl_matrix* nv_means = gsl_matrix_calloc(rbm->batch_size, rbm->n_visible);
	gsl_matrix* nv_sample = gsl_matrix_calloc(rbm->batch_size, rbm->n_visible);
	gsl_matrix* nh_means = gsl_matrix_calloc(rbm->batch_size, rbm->n_hidden);
	gsl_matrix* nh_sample = gsl_matrix_calloc(rbm->batch_size, rbm->n_hidden);

	gsl_matrix_memcpy(nh_sample, ph_sample);
	for (int step = 0; step < k; step++)
	{
		hidden_state_to_visible_probabilities_rbm (rbm, nh_sample, &nv_means, &nv_sample);
		visible_state_to_hidden_probabilities_rbm (rbm, nv_sample, &nh_means, &nh_sample);
	}

	// applies gradients on RBM: Delta_W, Delta_h, Delta_v

	double ratio = lr / rbm->batch_size;

	gsl_matrix* identity_h = gsl_matrix_alloc(rbm->n_hidden, rbm->n_hidden);
	gsl_matrix_set_all(identity_h, 1.0);
	gsl_matrix_set_identity(identity_h);

	gsl_matrix* identity_v = gsl_matrix_alloc(rbm->n_visible, rbm->n_visible);
	gsl_matrix_set_all(identity_v, 1.0);
	gsl_matrix_set_identity(identity_v);

	gsl_vector* ones = gsl_vector_alloc(rbm->batch_size);
	gsl_vector_set_all(ones, 1.0);

	// Compute and apply Delta_W
	gsl_matrix* delta_W = gsl_matrix_calloc(rbm->n_visible, rbm->n_hidden);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, input, ph_means, 1.0, delta_W);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, -1.0, nv_sample, nh_means, 1.0, delta_W);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, ratio, delta_W, identity_h, momentum, rbm->vel_W);
	gsl_matrix_add(rbm->W, rbm->vel_W);
	gsl_matrix_free(delta_W);

	// Compute and apply Delta_v
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, input, identity_v, -1.0, nv_sample); // we don't need nv_sample anymore
	gsl_blas_dgemv(CblasTrans, ratio, nv_sample, ones, momentum, rbm->vel_v);
	gsl_vector_add(rbm->vbias, rbm->vel_v);

	// Compute and apply Delta_h
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, ph_means, identity_h, -1.0, nh_means); // we don't need nh_mean anymore
	gsl_blas_dgemv(CblasTrans, ratio, nh_means, ones, momentum, rbm->vel_h);
	gsl_vector_add(rbm->hbias, rbm->vel_h);

	// approximation to the reconstruction error: sum over dimensions, mean over cases
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, input, identity_v, -1.0, nv_means); // we don't need nv_mean anymore
	gsl_matrix* pow_res = gsl_matrix_alloc(rbm->batch_size, rbm->n_visible);
	gsl_matrix_memcpy(pow_res, nv_means);
	gsl_matrix_mul_elements(pow_res, nv_means);
	gsl_vector* pow_sum = gsl_vector_calloc(rbm->n_visible);
	gsl_blas_dgemv(CblasTrans, 1.0, pow_res, ones, 1.0, pow_sum);

	double recon = 0;
	for(int j = 0; j < rbm->n_visible; j++) recon += gsl_vector_get(pow_sum, j);
	recon /= rbm->batch_size;

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
/* HOW TO TRAIN YOUR RBM                                                     */
/*---------------------------------------------------------------------------*/

// Function to train the RBM
//   param batchdata       : loaded dataset (rows = examples, cols = features)
//   param batch_size      : size of a batch used to train the RBM
//   param n_hidden        : number of hidden units in the RBM
//   param training_epochs : number of epochs used for training the RBM
//   param learning_rate   : learning rate used for training the RBM
//   param momentum        : momentum weight used for training the RBM
//   param init_W          : initial W weights for the RBM (can be NULL)
//   param init_hbias      : initial hbias weights for the RBM (can be NULL)
//   param init_vbias      : initial vbias weights for the RBM (can be NULL)
void train_rbm (RBM* rbm, gsl_matrix* batchdata, int nrow, int ncol, int batch_size,
                int n_hidden, int training_epochs, double learning_rate,
		double momentum, int rand_seed, gsl_matrix* init_W,
		gsl_vector* init_hbias, gsl_vector* init_vbias)
{
	srand(rand_seed);

	int n_train_batches = nrow / batch_size;
	int n_visible = ncol;

	// Shuffle Indices
	int* permindex = shuffle(nrow);

	// construct RBM
	create_RBM (rbm, nrow, n_visible, n_hidden, batch_size, init_W, init_hbias, init_vbias);

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

			gsl_matrix* input = gsl_matrix_alloc(rbm->batch_size, rbm->n_visible);
			for (int i = 0; i < batch_size; i++)
			{
				int index = permindex[i + idx_aux_ini];

				for (int j = 0; j < rbm->n_visible; j++)
					gsl_matrix_set(input, i, j, gsl_matrix_get(batchdata, index, j));
			}

			// get the cost and the gradient corresponding to one step of CD-k
			mean_cost += cdk_RBM (rbm, input, learning_rate, momentum, 1);

			gsl_matrix_free(input);
		}
		mean_cost /= n_train_batches;
//		if (epoch % 100 == 0)
			printf("Training epoch %d, cost is %f\n", epoch, mean_cost);
	}
	free(permindex);

	printf("Training epoch %d, cost is %f\n", training_epochs, mean_cost);
	return;
}

/*---------------------------------------------------------------------------*/
/* PREDICT AND RECONSTRUCT USING THE RBM                                     */
/*---------------------------------------------------------------------------*/

// This function makes a forward-backward pass of Matrix v_sample
//   param pv_sample   : matrix to activate and reconstruct from
//   param activations : matrix to store activations
//   param reconstruct : matrix to store reconstruction
void reconstruct_RBM (RBM* rbm, gsl_matrix* pv_sample, gsl_matrix** activations, gsl_matrix** reconstruct)
{
	int nrow = pv_sample->size1;

	// Activation and Reconstruction process
	gsl_matrix* nh_means = gsl_matrix_calloc(nrow, rbm->n_hidden);
	gsl_matrix* nh_sample = gsl_matrix_calloc(nrow, rbm->n_hidden);
	gsl_matrix* nv_means = gsl_matrix_calloc(nrow, rbm->n_visible);
	gsl_matrix* nv_sample = gsl_matrix_calloc(nrow, rbm->n_visible);

	visible_state_to_hidden_probabilities_rbm (rbm, pv_sample, &nh_means, &nh_sample);
	hidden_state_to_visible_probabilities_rbm (rbm, nh_sample, &nv_means, &nv_sample);

	// Copy results to activation and reconstruction matrix (and padding for delay)
	gsl_matrix_memcpy(*activations, nh_means);
	gsl_matrix_memcpy(*reconstruct, nv_sample);

	// Free auxiliar structures
	gsl_matrix_free(nh_means);
	gsl_matrix_free(nh_sample);
	gsl_matrix_free(nv_means);
	gsl_matrix_free(nv_sample);
}

// This function makes a forward pass of Matrix v_sample
//   param pv_sample   : matrix to activate from
//   param activations : matrix to store activations
void forward_RBM (RBM* rbm, gsl_matrix* pv_sample, gsl_matrix** activations)
{
	int nrow = pv_sample->size1;

	// Activation and Reconstruction process
	gsl_matrix* nh_means = gsl_matrix_calloc(nrow, rbm->n_hidden);
	gsl_matrix* nh_sample = gsl_matrix_calloc(nrow, rbm->n_hidden);

	visible_state_to_hidden_probabilities_rbm (rbm, pv_sample, &nh_means, &nh_sample);

	// Copy results to activation matrix (and padding for delay)
	gsl_matrix_memcpy(*activations, nh_means);

	// Free auxiliar structures
	gsl_matrix_free(nh_means);
	gsl_matrix_free(nh_sample);
}

// This function makes a backward pass of Matrix h_sample
//   param nh_sample   : matrix to reconstruct from
//   param reconstruct : matrix to store reconstruction
void backward_RBM (RBM* rbm, gsl_matrix* nh_sample, gsl_matrix** reconstruct)
{
	int nrow = nh_sample->size1;

	// Activation and Reconstruction process
	gsl_matrix* nv_means = gsl_matrix_calloc(nrow, rbm->n_visible);
	gsl_matrix* nv_sample = gsl_matrix_calloc(nrow, rbm->n_visible);

	hidden_state_to_visible_probabilities_rbm (rbm, nh_sample, &nv_means, &nv_sample);

	// Copy results to reconstruction matrix (and padding for delay)
	gsl_matrix_memcpy(*reconstruct, nv_sample);

	// Free auxiliar structures
	gsl_matrix_free(nv_means);
	gsl_matrix_free(nv_sample);
}

/*---------------------------------------------------------------------------*/
/* AUXILIAR FUNCTIONS                                                        */
/*---------------------------------------------------------------------------*/

// This function reads a dataset into a matrix. Consider a matrix of Images x Features (flattened images)
//   param dataset  : empty matrix to fill with data
//   param filename : filename for the dataset to read
//   param nrow     : number of images in the dataset
//   param ncol     : number of features for each image
void read_dataset (gsl_matrix* dataset, char* filename, int nrow, int ncol)
{
	FILE * fp;
	char * line = NULL;
	size_t len = 0;

	fp = fopen(filename, "r");
	if (fp == NULL) exit(EXIT_FAILURE);

	for (int i = 0; i < nrow; i++)
	{
		ssize_t read = getline(&line, &len, fp);
		char *ch = strtok(line, " ");
		for (int j = 0; j < ncol; j++)
		{
			gsl_matrix_set(dataset, i, j, atof(ch));
			ch = strtok(NULL, " ");
		}
		free(ch);
	}
	fclose(fp);

	if (line) free(line);
}

// This function compares a dataset (e.g. the test set) against its reconstruction,
// returning the squared average.
//   param dataset        : the ground truth dataset
//   param reconstruction : the reconstructed dataset
//   param nrow           : number of images in the dataset
//   param ncol           : number of features for each image
//   returns mse          : mean squared error between dataset and reconstruction
double check_error (gsl_matrix* dataset, gsl_matrix* reconstruction, int nrow, int ncol)
{
	double error_recons = 0;
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
		{
			double diff = gsl_matrix_get(dataset, i, j) - gsl_matrix_get(reconstruction, i, j);
			error_recons += pow(diff, 2);
		}
	return error_recons / (nrow * ncol);
}

/*---------------------------------------------------------------------------*/
/* MAIN FUNCTION                                                             */
/*---------------------------------------------------------------------------*/

int main (int argc, char** argv)
{
	if (argc != 12)
	{
		printf("Usage: rbm <batch size> <hidden units> <epochs> <learning rate> <momentum> <random seed> <training file> <num images> <num features> <test file> <num images test>\n");
		printf("Example: rbm 10 30 10 1e-3 0.5 1234 mnist_trainx.data 60000 784 mnist_test.data 10000\n");
		exit(0);
	}

	int basi = atoi(argv[1]);		// Batch Size, e.g. 10
	int nhid = atoi(argv[2]);		// Num. Hidden Units, e.g. 30
	int trep = atoi(argv[3]);		// Training Epochs, e.g. 10
	double lera = atof(argv[4]);	// Learning Rate, e.g. 1e-3
	double mome = atof(argv[5]);	// Momentum, e.g. 0.5
	int rase = atoi(argv[6]);		// Random Seed, e.g. 1234

	char* filename_tr = argv[7];	// Training Filename
	int nrow = atoi(argv[8]);		// Number Images in Training Dataset, e.g. 60000
	int ncol = atoi(argv[9]);		// Number Features per Image, e.g. 784

	char* filename_ts = argv[10];	// Testing Filename
	int nrow_ts = atoi(argv[11]);	// Number Images in Testing Dataset, e.g. 10000

	// Read Train Data
	gsl_matrix* train_X_p = gsl_matrix_alloc(nrow, ncol);
	read_dataset(train_X_p, filename_tr, nrow, ncol);

	printf("Read Train Data\n");

	// Perform Training
	RBM rbm;
	train_rbm (&rbm, train_X_p, nrow, ncol, basi, nhid, trep, lera, mome, rase, NULL, NULL, NULL);

	printf("Training Finished\n");

	// Read Test Data
	gsl_matrix* test_X_p = gsl_matrix_alloc(nrow_ts, ncol);
	read_dataset(test_X_p, filename_ts, nrow_ts, ncol);

	printf("Read Test Data\n");

	// Perform Testing
	gsl_matrix* reconstruction = gsl_matrix_calloc(nrow_ts, ncol);
	gsl_matrix* activations = gsl_matrix_calloc(nrow_ts, nhid);
	reconstruct_RBM(&rbm, test_X_p, &activations, &reconstruction);

	printf("Testing Finished\n");

	// Check Error again
	double error_recons = check_error(test_X_p, reconstruction, nrow_ts, ncol);

	printf("Reconstruction Error: %f\n", error_recons);

	// Free Dataset Structure
	free_RBM(&rbm);

	gsl_matrix_free(train_X_p);
	gsl_matrix_free(test_X_p);
	gsl_matrix_free(reconstruction);
	gsl_matrix_free(activations);

	return 0;
}

