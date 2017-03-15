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
} CRBM;

/*---------------------------------------------------------------------------*/
/* AUXILIAR FUNCTIONS                                                        */
/*---------------------------------------------------------------------------*/

// Function to produce Uniform Samples
double uniform (double min, double max)
{
	return rand() / (RAND_MAX + 1.0) * (max - min) + min;  
}

// Function to produce Normal Samples
double normal(double mean, double stdev) 
{
	double rnd1 = (rand() + 1.0)/(RAND_MAX + 1.0);
	double rnd2 = (rand() + 1.0)/(RAND_MAX + 1.0);
	return mean + sqrt(-2 * log(rnd1)) * cos( 2 * M_PI * rnd2) / stdev;
}

// Function to produce Bernoulli Samples
int binomial (int n, double p)
{
	if (p < 0 || p > 1) return 0;

	int c = 0;
	double r;

	for(int i = 0; i < n; i++)
	{
		r = rand() / (RAND_MAX + 1.0);
		if (r < p) c++;
	}

	return c;
}

// Function to produce the Sigmoid
double sigmoid (double x)
{
	return 1.0 / (1.0 + exp(-x));
}

// Function to transpose a matrix
double** transpose (double** mat, int nrow, int ncol)
{
	double** retval = (double**) malloc(sizeof(double) * nrow * ncol);
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
			retval[i * ncol + j] = mat[j * nrow + i];
	return retval;
}

// Function to produce a random shuffle from 1 to limit
int* shuffle (int limit)
{
	int* vec = (int*) malloc(sizeof(int) * limit);
	for (int i = 0; i < limit; i++) vec[i] = i;

	if (limit > 1)
		for (int i = limit - 1; i > 0; i--)
		{
			int j = (int) uniform(0, i); //(unsigned int) (drand48()*(i + 1));
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
void create_CRBM (CRBM* crbm, int N, int n_visible, int n_hidden, int delay,
		double** A, double** B, double** W, double* hbias, double* vbias)
{
	double a = 1.0 / n_visible;

	// Initialize Parameters
	crbm->N = N;
	crbm->n_visible = n_visible;
	crbm->n_hidden = n_hidden;
	crbm->delay = delay;

	// Initialize Matrices and Vectors
	if (W == NULL)
	{
		crbm->W = (double**) malloc(sizeof(double*) * n_hidden);
		for(int i = 0; i < n_hidden; i++)
		{
			crbm->W[i] = (double*) malloc(sizeof(double) * n_visible);
			for(int j = 0; j < n_visible; j++) crbm->W[i][j] = 0.01 * normal(0,1);
		}
	} else crbm->W = W;

	if (A == NULL)
	{
		crbm->A = (double**) malloc(sizeof(double*) * n_visible);
		for(int i = 0; i < n_visible; i++)
		{
			crbm->A[i] = (double*) malloc(sizeof(double) * n_visible * delay);
			for(int j = 0; j < n_visible * delay; j++) crbm->A[i][j] = 0.01 * normal(0,1);
		}
	} else crbm->A = A;

	if (B == NULL)
	{
		crbm->B = (double**) malloc(sizeof(double*) * n_hidden);
		for(int i = 0; i < n_hidden; i++)
		{
			crbm->B[i] = (double*) malloc(sizeof(double) * n_visible * delay);
			for(int j = 0; j < n_visible * delay; j++) crbm->B[i][j] = 0.01 * normal(0,1);
		}
	} else crbm->B = B;

	if (hbias == NULL)
	{
		crbm->hbias = (double*) malloc(sizeof(double) * n_hidden);
		for(int i = 0; i < n_hidden; i++) crbm->hbias[i] = 0;
	} else crbm->hbias = hbias;

	if (vbias == NULL)
	{
		crbm->vbias = (double*) malloc(sizeof(double) * n_visible);
		for(int i = 0; i < n_visible; i++) crbm->vbias[i] = 0;
	} else crbm->vbias = vbias;

	// Initialize Velocity for Momentum
	crbm->vel_W = (double**) malloc(sizeof(double*) * n_hidden);
	crbm->vel_B = (double**) malloc(sizeof(double*) * n_hidden);
	crbm->vel_h = (double*) malloc(sizeof(double) * n_hidden);
	for(int i = 0; i < n_hidden; i++)
	{
		crbm->vel_W[i] = (double*) malloc(sizeof(double) * n_visible);
		for(int j = 0; j < n_visible; j++) crbm->vel_W[i][j] = 0;

		crbm->vel_B[i] = (double*) malloc(sizeof(double) * n_visible * delay);
		for(int j = 0; j < n_visible * delay; j++) crbm->vel_B[i][j] = 0;
		crbm->vel_h[i] = 0;
	}

	crbm->vel_A = (double**) malloc(sizeof(double*) * n_visible);
	crbm->vel_v = (double*) malloc(sizeof(double) * n_visible);
	for(int i = 0; i < n_visible; i++)
	{
		crbm->vel_A[i] = (double*) malloc(sizeof(double) * n_visible * delay);
		for(int j = 0; j < n_visible * delay; j++) crbm->vel_A[i][j] = 0;
		crbm->vel_v[i] = 0;
	}
}

// Destructor of RBMs
void free_CRBM (CRBM* crbm)
{
	for(int i = 0; i < crbm->n_hidden; i++)
	{
		free(crbm->W[i]);
		free(crbm->B[i]);
		free(crbm->vel_W[i]);
		free(crbm->vel_B[i]);
	}
	free(crbm->W);
	free(crbm->B);
	free(crbm->hbias);
	free(crbm->vbias);

	for(int i = 0; i < crbm->n_visible; i++)
	{
		free(crbm->A[i]);
		free(crbm->vel_A[i]);
	}
	free(crbm->A);

	free(crbm->vel_W);
	free(crbm->vel_B);
	free(crbm->vel_A);
	free(crbm->vel_h);
	free(crbm->vel_v);
}

// Prop-Up Function
double CRBM_propup (CRBM* crbm, double* vis, double* v_hist, int i)
{
	double pre_sigmoid_activation = 0.0;
	for(int j = 0; j < crbm->n_visible; j++) pre_sigmoid_activation += crbm->W[i][j] * vis[j];
	for(int j = 0; j < crbm->n_visible * crbm->delay; j++) pre_sigmoid_activation += crbm->B[i][j] * v_hist[j];
	pre_sigmoid_activation += crbm->hbias[i];
	return sigmoid(pre_sigmoid_activation);
}

// Prop-Down Function
double CRBM_propdown (CRBM* crbm, double* hid, double* v_hist, int i)
{
	double pre_sigmoid_activation = 0.0;
	for(int j = 0; j < crbm->n_hidden; j++) pre_sigmoid_activation += crbm->W[j][i] * hid[j];
	for(int j = 0; j < crbm->n_visible * crbm->delay; j++) pre_sigmoid_activation += crbm->A[i][j] * v_hist[j];
	pre_sigmoid_activation += crbm->vbias[i];

	return sigmoid(pre_sigmoid_activation);
}

// This function infers state of hidden units given visible units
void visible_state_to_hidden_probabilities (CRBM* crbm, double* v0_sample, double* v_history, double* mean, double* sample)
{
	for(int i = 0; i < crbm->n_hidden; i++)
	{
		mean[i] = CRBM_propup(crbm, v0_sample, v_history, i);
		sample[i] = binomial(1, mean[i]);
	}
}

// This function infers state of visible units given hidden units
void hidden_state_to_visible_probabilities (CRBM* crbm, double* h0_sample, double* v_history, double* mean, double* sample)
{
	for (int i = 0; i < crbm->n_visible; i++)
	{
		mean[i] = CRBM_propdown(crbm, h0_sample, v_history, i);
		sample[i] = binomial(1, mean[i]);
	}
}

// This functions implements one step of CD-k
//   param input      : matrix input from batch data (n_vis x n_seq)
//   param input_hist : matrix input_history from batch data (n_seq x (n_vis * delay))
//   param lr         : learning rate used to train the RBM
//   param momentum   : value for momentum coefficient on learning
//   param k          : number of Gibbs steps to do in CD-k
double cdk_CRBM (CRBM* crbm, double* input, double* input_history, double lr, double momentum, int k)
{
	double* ph_mean = (double*) malloc(sizeof(double) * crbm->n_hidden);
	double* ph_sample = (double*) malloc(sizeof(double) * crbm->n_hidden);
	double* nv_means = (double*) malloc(sizeof(double) * crbm->n_visible);
	double* nv_sample = (double*) malloc(sizeof(double) * crbm->n_visible);
	double* nh_means = (double*) malloc(sizeof(double) * crbm->n_hidden);
	double* nh_sample = (double*) malloc(sizeof(double) * crbm->n_hidden);

	// compute positive phase (awake)
	visible_state_to_hidden_probabilities (crbm, input, input_history, ph_mean, ph_sample);

	// perform negative phase (asleep)
	for (int i = 0; i < crbm->n_hidden; i++) nh_sample[i] = ph_sample[i];
	for (int step = 0; step < k; step++)
	{
		hidden_state_to_visible_probabilities (crbm, nh_sample, input_history, nv_means, nv_sample);
		visible_state_to_hidden_probabilities (crbm, nv_sample, input_history, nh_means, nh_sample);
	}

	// determine gradients on CRMB: Delta_W, Delta_B, Delta_h
	for (int i = 0; i < crbm->n_hidden; i++)
	{
		for(int j = 0; j < crbm->n_visible; j++)
		{
			double delta = (ph_mean[i] * input[j] - nh_means[i] * nv_sample[j]);// / crbm->N;
			crbm->vel_W[i][j] = crbm->vel_W[i][j] * momentum + lr * delta;
			crbm->W[i][j] += crbm->vel_W[i][j];
		}
		for(int j = 0; j < crbm->n_visible * crbm->delay; j++)
		{
			double delta = (ph_mean[i] * input_history[j] - nh_means[i] * input_history[j]);// / crbm->N;
			crbm->vel_B[i][j] = crbm->vel_B[i][j] * momentum + lr * delta;
			crbm->B[i][j] += crbm->vel_B[i][j];
		}
		double delta = (ph_sample[i] - nh_means[i]);// / crbm->N;
		crbm->vel_h[i] = crbm->vel_h[i] * momentum + lr * delta;
		crbm->hbias[i] += crbm->vel_h[i];
	}

	// determine gradients on CRMB: Delta_A, Delta_v
	for (int i = 0; i < crbm->n_visible; i++)
	{
		for(int j = 0; j < crbm->n_visible * crbm->delay; j++)
		{
			double delta = (input[i] * input_history[j] - nv_sample[i] * input_history[j]);// / crbm->N;
			crbm->vel_A[i][j] = crbm->vel_A[i][j] * momentum + lr * delta;
			crbm->A[i][j] += crbm->vel_A[i][j];
		}
		double delta = (input[i] - nv_sample[i]);// / crbm->N;
		crbm->vel_v[i] = crbm->vel_v[i] * momentum + lr * delta;
		crbm->vbias[i] += crbm->vel_v[i];
	}

	// approximation to the reconstruction error: sum over dimensions, mean over cases
	double recon = 0;
	for (int i = 0; i < crbm->n_visible; i++)
		recon += pow(input[i] - nv_means[i], 2);
	recon /= crbm->n_visible;

	free(ph_mean);
	free(ph_sample);
	free(nv_means);
	free(nv_sample);
	free(nh_means);
	free(nh_sample);

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
void train_crbm (CRBM* crbm, double** batchdata, int nrow, int ncol,
		int batch_size, int n_hidden, int training_epochs,
		double learning_rate, double momentum, int delay,
		int rand_seed)
{
	srand(rand_seed);

	batch_size = 1; //FIXME - At this time, one input at a time

	int n_train_batches = nrow / batch_size;
	int n_visible = ncol;

	// Shuffle Indices
	int* permindex = shuffle(nrow - delay);
	for (int i = 0; i < nrow - delay; i++) permindex[i] = permindex[i] + delay;

	// construct RBM
	create_CRBM (crbm, nrow, n_visible, n_hidden, delay, NULL, NULL, NULL, NULL, NULL);

	// go through the training epochs and training set
	double mean_cost;
	for(int epoch = 0; epoch < training_epochs; epoch++)
	{
		mean_cost = 0;
		for(int batch_index = 0; batch_index < n_train_batches; batch_index++)
		{
			int idx_aux_ini = batch_index * batch_size;
			int idx_aux_fin = idx_aux_ini + batch_size;

			if (idx_aux_fin >= nrow - delay) break;

			int index = permindex[batch_index];
			double* input = (double*) malloc(sizeof(double) * ncol);
			for (int j = 0; j < ncol; j++) input[j] = batchdata[index][j];

			double* input_hist = (double*) malloc(sizeof(double) * ncol * delay);
			for (int i = 0; i < delay; i++)
				for (int j = 0; j < ncol; j++)
					input_hist[i * ncol + j] = batchdata[index - 1 - i][j];

			// get the cost and the gradient corresponding to one step of CD-k
			mean_cost += cdk_CRBM (crbm, input, input_hist, learning_rate, momentum, 1);

			free(input);
			free(input_hist);
		}
		mean_cost /= batch_size;
		if (epoch % 100 == 0) printf("Training epoch %d, cost is %f\n", epoch, mean_cost);
	}
	free(permindex);

	printf("Training epoch %d, cost is %f\n", training_epochs, mean_cost);
	return;
}

/*---------------------------------------------------------------------------*/
/* PREDICT AND RECONSTRUCT USING THE CRBM                                    */
/*---------------------------------------------------------------------------*/

// This function computes the activation of Vector V
//   param v : vector to predict
//   param v_hist  : history matrix for conditioning
double* activation_vector_CRBM (CRBM* crbm, double* v, double* v_hist)
{
	double* activation = (double*) malloc(sizeof(double) * crbm->n_hidden);
	for (int i = 0; i < crbm->n_hidden; i++)
		activation[i] = CRBM_propup(crbm, v, v_hist, i);
	return activation;
}

// This function computes the activation of Matrix V with history V_hist
//   param v       : matrix to predict
//   param v_hist  : history matrix for conditioning
double** activation_CRBM (CRBM* crbm, double** v, int nrow)
{
	double** activation = (double**) malloc(sizeof(double) * nrow);
	double* v_hist = (double*) malloc(sizeof(double) * crbm->n_visible * crbm->delay);
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
double* reconstruct_vector_CRBM (CRBM* crbm, double* v, double* v_hist)
{
	double* reconstructed = (double*) malloc(sizeof(double) * crbm->n_visible);
	double* h = activation_vector_CRBM (crbm, v, v_hist);
	for (int i = 0; i < crbm->n_visible; i++)
		reconstructed[i] = CRBM_propdown(crbm, h, v_hist, i);
	free(h);
	return reconstructed;
}

// This function makes a reconstruction of Matrix V
//   param v : matrix to reconstruct
//   param v_hist  : history matrix for conditioning
double** reconstruct_CRBM (CRBM* crbm, double** v, int nrow)
{
	double** reconstruct = (double**) malloc(sizeof(double) * nrow);
	double* v_hist = (double*) malloc(sizeof(double) * crbm->n_visible * crbm->delay);
	for (int i = crbm->delay; i < nrow; i++)
	{
		for (int j = 0; j < crbm->delay; j++)
			for (int k = 0; k < crbm->n_visible; k++)
				v_hist[j * crbm->n_visible + k] = v[i - j - 1][k];
		reconstruct[i] = reconstruct_vector_CRBM(crbm, v[i], v_hist);
	}
	free(v_hist);
	return reconstruct;
}

/*---------------------------------------------------------------------------*/
/* FORECAST AND SIMULATION USING THE CRBM                                    */
/*---------------------------------------------------------------------------*/

//   param n_gibbs : number of gibbs iterations
double** generate_samples(CRBM* crbm, double** orig_data, double** orig_hist,
	int n_samples, int n_gibbs)
{
	// TODO - ...
}

/*---------------------------------------------------------------------------*/
/* INTERFACE TO R                                                            */
/*---------------------------------------------------------------------------*/

#define RMATRIX(m,i,j) (REAL(m)[ INTEGER(GET_DIM(m))[0]*(j)+(i) ])
#define RVECTOR(v,i) (REAL(v)[(i)])

// Interface for Training a CRBM
SEXP _C_CRBM_train(SEXP dataset, SEXP batch_size, SEXP n_hidden, SEXP training_epochs,
           SEXP learning_rate, SEXP momentum, SEXP delay, SEXP rand_seed)
{
 	int nrow = INTEGER(GET_DIM(dataset))[0];
 	int ncol = INTEGER(GET_DIM(dataset))[1];

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

	// Perform Training
	CRBM crbm;
	train_crbm (&crbm, train_X_p, nrow, ncol, basi, nhid, trep, lera, mome, dely, rase);

	// Return Structure
	SEXP retval = PROTECT(allocVector(VECSXP, 14));

	SET_VECTOR_ELT(retval, 0, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 0))[0] = crbm.N;

	SET_VECTOR_ELT(retval, 1, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 1))[0] = crbm.n_visible;

	SET_VECTOR_ELT(retval, 2, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 2))[0] = crbm.n_hidden;

	SET_VECTOR_ELT(retval, 3, allocMatrix(REALSXP, nhid, ncol));
	SET_VECTOR_ELT(retval, 4, allocMatrix(REALSXP, nhid, ncol * dely));
	SET_VECTOR_ELT(retval, 6, allocVector(REALSXP, nhid));
	SET_VECTOR_ELT(retval, 8, allocMatrix(REALSXP, nhid, ncol));
	SET_VECTOR_ELT(retval, 9, allocMatrix(REALSXP, nhid, ncol * dely));
	SET_VECTOR_ELT(retval, 11, allocVector(REALSXP, nhid));
	for (int i = 0; i < nhid; i++)
	{
		for (int j = 0; j < ncol; j++)
		{
			REAL(VECTOR_ELT(retval, 3))[i * ncol + j] = crbm.W[i][j];
			REAL(VECTOR_ELT(retval, 8))[i * ncol + j] = crbm.vel_W[i][j];
		}
		for (int j = 0; j < ncol * dely; j++)
		{
			REAL(VECTOR_ELT(retval, 4))[i * ncol + j] = crbm.B[i][j];
			REAL(VECTOR_ELT(retval, 9))[i * ncol + j] = crbm.vel_B[i][j];
		}
		REAL(VECTOR_ELT(retval, 6))[i] = crbm.hbias[i];
		REAL(VECTOR_ELT(retval, 11))[i] = crbm.vel_h[i];
	}

	SET_VECTOR_ELT(retval, 5, allocMatrix(REALSXP, ncol, ncol * dely));
	SET_VECTOR_ELT(retval, 7, allocVector(REALSXP, ncol));
	SET_VECTOR_ELT(retval, 10, allocMatrix(REALSXP, ncol, ncol * dely));
	SET_VECTOR_ELT(retval, 12, allocVector(REALSXP, ncol));
	for (int i = 0; i < ncol; i++)
	{
		for (int j = 0; j < ncol * dely; j++)
		{
			REAL(VECTOR_ELT(retval, 5))[i * ncol + j] = crbm.A[i][j];
			REAL(VECTOR_ELT(retval, 10))[i * ncol + j] = crbm.vel_A[i][j];
		}
		REAL(VECTOR_ELT(retval, 7))[i] = crbm.vbias[i];
		REAL(VECTOR_ELT(retval, 12))[i] = crbm.vel_v[i];
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
	for (int i = 0; i < nrow; i++) free(train_X_p[i]);
	free(train_X_p);

	free_CRBM(&crbm);

	return retval;
}

// Interface for Predicting and Reconstructing using an RBM
SEXP _C_CRBM_predict (SEXP newdata, SEXP n_visible, SEXP n_hidden, SEXP W_input,
	SEXP B_input, SEXP A_input, SEXP hbias_input, SEXP vbias_input, SEXP delay)
{
 	int nrow = INTEGER(GET_DIM(newdata))[0];
 	int ncol = INTEGER(GET_DIM(newdata))[1];

	int nhid = INTEGER_VALUE(n_hidden);
	int dely = INTEGER_VALUE(delay);

 	int wrow = INTEGER(GET_DIM(W_input))[0];
 	int wcol = INTEGER(GET_DIM(W_input))[1];

 	int brow = INTEGER(GET_DIM(B_input))[0];
 	int bcol = INTEGER(GET_DIM(B_input))[1];

 	int arow = INTEGER(GET_DIM(A_input))[0];
 	int acol = INTEGER(GET_DIM(A_input))[1];

	// Re-assemble the RBM
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
	for (int i = 0; i < ncol; i++) vbias[i] = RVECTOR(vbias_input,i);

	CRBM crbm;
	create_CRBM (&crbm, 0, ncol, nhid, dely, A, B, W, hbias, vbias);

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

	SET_VECTOR_ELT(retval, 1, allocMatrix(REALSXP, nhid, ncol));
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

	// Free the structures and the RBM
	for (int i = 0; i < nrow; i++) free(test_X_p[i]);
	free(test_X_p);

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
	int batch_size = 1;
	int delay = 2;

	CRBM crbm;
	train_crbm (&crbm, train_X_p, train_N, n_visible, batch_size,
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

