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
} RBM;

/*---------------------------------------------------------------------------*/
/* AUXILIAR FUNCTIONS                                                        */
/*---------------------------------------------------------------------------*/

// Function to produce Uniform Samples
double uniform (double min, double max)
{
	return rand() / (RAND_MAX + 1.0) * (max - min) + min;  
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
/* RBM FUNCTIONS                                                             */
/*---------------------------------------------------------------------------*/

// Restricted Boltzmann Machine (RBM). Constructor
void create_RBM (RBM* rbm, int N, int n_visible, int n_hidden, double** W, double* hbias, double* vbias)
{
	double a = 1.0 / n_visible;

	// Initialize Parameters
	rbm->N = N;
	rbm->n_visible = n_visible;
	rbm->n_hidden = n_hidden;

	// Initialize Matrices and Vectors
	if (W == NULL)
	{
		rbm->W = (double**) malloc(sizeof(double*) * n_hidden);
		for(int i = 0; i < n_hidden; i++)
		{
			rbm->W[i] = (double*) malloc(sizeof(double) * n_visible);
			for(int j = 0; j < n_visible; j++) rbm->W[i][j] = uniform(-a, a);
		}
	} else rbm->W = W;

	if (hbias == NULL)
	{
		rbm->hbias = (double*) malloc(sizeof(double) * n_hidden);
		for(int i = 0; i < n_hidden; i++) rbm->hbias[i] = 0;
	} else rbm->hbias = hbias;

	if (vbias == NULL)
	{
		rbm->vbias = (double*) malloc(sizeof(double) * n_visible);
		for(int i = 0; i < n_visible; i++) rbm->vbias[i] = 0;
	} else rbm->vbias = vbias;

	// Initialize Velocity for Momentum
	rbm->vel_W = (double**) malloc(sizeof(double*) * n_hidden);
	for(int i = 0; i < n_hidden; i++)
	{
		rbm->vel_W[i] = (double*) malloc(sizeof(double) * n_visible);
		for(int j = 0; j < n_visible; j++) rbm->vel_W[i][j] = 0;
	}

	rbm->vel_h = (double*) malloc(sizeof(double) * n_hidden);
	for(int i = 0; i < n_hidden; i++) rbm->vel_h[i] = 0;

	rbm->vel_v = (double*) malloc(sizeof(double) * n_visible);
	for(int i = 0; i < n_visible; i++) rbm->vel_v[i] = 0;
}

// Destructor of RBMs
void free_RBM (RBM* rbm)
{
	for(int i = 0; i < rbm->n_hidden; i++)
	{
		free(rbm->W[i]);
		free(rbm->vel_W[i]);
	}
	free(rbm->W);
	free(rbm->hbias);
	free(rbm->vbias);

	free(rbm->vel_W);
	free(rbm->vel_h);
	free(rbm->vel_v);
}

// Prop-Up Function
double RBM_propup (RBM* rbm, double* v, double* w, double b)
{
	double pre_sigmoid_activation = 0.0;

	for(int j = 0; j < rbm->n_visible; j++) pre_sigmoid_activation += w[j] * v[j];
	pre_sigmoid_activation += b;

	return sigmoid(pre_sigmoid_activation);
}

// Prop-Down Function
double RBM_propdown (RBM* rbm, double* h, int i, double b)
{
	double pre_sigmoid_activation = 0.0;

	for(int j = 0; j < rbm->n_hidden; j++) pre_sigmoid_activation += rbm->W[j][i] * h[j];
	pre_sigmoid_activation += b;

	return sigmoid(pre_sigmoid_activation);
}

// This function infers state of hidden units given visible units
void visible_state_to_hidden_probabilities (RBM* rbm, double* v0_sample, double* mean, double* sample)
{
	for(int i = 0; i < rbm->n_hidden; i++)
	{
		mean[i] = RBM_propup(rbm, v0_sample, rbm->W[i], rbm->hbias[i]);
		sample[i] = binomial(1, mean[i]);
	}
}

// This function infers state of visible units given hidden units
void hidden_state_to_visible_probabilities (RBM* rbm, double* h0_sample, double* mean, double* sample)
{
	for (int i = 0; i < rbm->n_visible; i++)
	{
		mean[i] = RBM_propdown(rbm, h0_sample, i, rbm->vbias[i]);
		sample[i] = binomial(1, mean[i]);
	}
}

// This functions implements one step of CD-k
//   param input    : matrix input from batch data (n_vis x n_seq)
//   param lr       : learning rate used to train the RBM
//   param momentum : value for momentum coefficient on learning
//   param k        : number of Gibbs steps to do in CD-k
double cdk_RBM (RBM* rbm, double* input, double lr, double momentum, int k)
{
	double* ph_mean = (double*) malloc(sizeof(double) * rbm->n_hidden);
	double* ph_sample = (double*) malloc(sizeof(double) * rbm->n_hidden);
	double* nv_means = (double*) malloc(sizeof(double) * rbm->n_visible);
	double* nv_sample = (double*) malloc(sizeof(double) * rbm->n_visible);
	double* nh_means = (double*) malloc(sizeof(double) * rbm->n_hidden);
	double* nh_sample = (double*) malloc(sizeof(double) * rbm->n_hidden);

	// compute positive phase (awake)
	visible_state_to_hidden_probabilities (rbm, input, ph_mean, ph_sample);

	// perform negative phase (asleep)
	for (int i = 0; i < rbm->n_hidden; i++) nh_sample[i] = ph_sample[i];
	for (int step = 0; step < k; step++)
	{
		hidden_state_to_visible_probabilities (rbm, nh_sample, nv_means, nv_sample);
		visible_state_to_hidden_probabilities (rbm, nv_sample, nh_means, nh_sample);
	}

	// determine gradients on RMB: Delta_W
	for (int i = 0; i < rbm->n_hidden; i++)
		for(int j = 0; j < rbm->n_visible; j++)
		{
			double delta = (ph_mean[i] * input[j] - nh_means[i] * nv_sample[j]) / rbm->N;
			rbm->vel_W[i][j] = rbm->vel_W[i][j] * momentum + lr * delta;
			rbm->W[i][j] += rbm->vel_W[i][j];
		}


	// determine gradients on RMB: Delta_h
	for (int i = 0; i < rbm->n_hidden; i++)
	{
		double delta = (ph_sample[i] - nh_means[i]) / rbm->N;
		rbm->vel_h[i] = rbm->vel_h[i] * momentum + lr * delta;
		rbm->hbias[i] += rbm->vel_h[i];
	}

	// determine gradients on RMB: Delta_v
	for(int i = 0; i < rbm->n_visible; i++)
	{
		double delta = (input[i] - nv_sample[i]) / rbm->N;
		rbm->vel_v[i] = rbm->vel_v[i] * momentum + lr * delta;
		rbm->vbias[i] += rbm->vel_v[i];
	}

	// approximation to the reconstruction error: sum over dimensions, mean over cases
	double recon = 0;
	for (int i = 0; i < rbm->n_visible; i++)
		recon += pow(input[i] - nv_means[i], 2);
	recon /= rbm->n_visible;

	free(ph_mean);
	free(ph_sample);
	free(nv_means);
	free(nv_sample);
	free(nh_means);
	free(nh_sample);

	return recon;
}

/*---------------------------------------------------------------------------*/
/* HOW TO TRAIN YOUR RBM                                                     */
/*---------------------------------------------------------------------------*/

// Function to train the RBM
//   param dataset         : loaded dataset (rows = examples, cols = features)
//   param learning_rate   : learning rate used for training the RBM
//   param training_epochs : number of epochs used for training
//   param batch_size      : size of a batch used to train the RBM
void train_rbm (RBM* rbm, double** batchdata, int nrow, int ncol, int batch_size,
                int n_hidden, int training_epochs, double learning_rate,
		double momentum, int rand_seed)
{
	srand(rand_seed);

	batch_size = 1; //FIXME - At this time, one input at a time

	int n_train_batches = nrow / batch_size;
	int n_visible = ncol;

	// Shuffle Indices
	int* permindex = shuffle(nrow);

	// construct RBM
	create_RBM (rbm, nrow, n_visible, n_hidden, NULL, NULL, NULL);

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

		/*	double input[batch_size][ncol];
			int p = 0;
			for (int i = idx_aux_ini; i < idx_aux_fin; i++)
			{
				int index = permindex[i];
				for (int j = 0; j < ncol; j++)
					input[p][j] = batchdata[index][j];
				p++;
			} */

			double* input = (double*) malloc(sizeof(double) * ncol);
			int index = permindex[batch_index];

			for (int j = 0; j < ncol; j++)
				input[j] = batchdata[index][j];

			// get the cost and the gradient corresponding to one step of CD-k
			mean_cost += cdk_RBM (rbm, input, learning_rate, momentum, 1);

			free(input);
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
//   param v : vector to predict
double* activation_vector_RBM (RBM* rbm, double* v)
{
	double* activation = (double*) malloc(sizeof(double) * rbm->n_hidden);
	for (int i = 0; i < rbm->n_hidden; i++)
		activation[i] = RBM_propup(rbm, v, rbm->W[i], rbm->hbias[i]);
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
double* reconstruct_vector_RBM (RBM* rbm, double* v)
{
	double* reconstructed = (double*) malloc(sizeof(double) * rbm->n_visible);
	double* h = activation_vector_RBM (rbm, v);
	double pre_sigmoid_activation;
	for (int i = 0; i < rbm->n_visible; i++)
	{
		pre_sigmoid_activation = 0.0;
		for (int j = 0; j < rbm->n_hidden; j++)
			pre_sigmoid_activation += rbm->W[j][i] * h[j];
		pre_sigmoid_activation += rbm->vbias[i];
		reconstructed[i] = sigmoid(pre_sigmoid_activation);
	}
	free(h);
	return reconstructed;
}

// This function makes a reconstruction of Matrix V
//   param v : matrix to reconstruct
double** reconstruct_RBM (RBM* rbm, double** v, int nrow)
{
	double** reconstruct = (double**) malloc(sizeof(double) * nrow);
	for (int i = 0; i < nrow; i++)
		reconstruct[i] = reconstruct_vector_RBM(rbm, v[i]);
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

	SET_VECTOR_ELT(retval, 3, allocMatrix(REALSXP, nhid, ncol));
	for (int i = 0; i < nhid; i++)
		for (int j = 0; j < ncol; j++)
			REAL(VECTOR_ELT(retval, 3))[i * ncol + j] = rbm.W[i][j];

	SET_VECTOR_ELT(retval, 4, allocVector(REALSXP, nhid));
	for (int i = 0; i < nrow; i++)
		REAL(VECTOR_ELT(retval, 4))[i] = rbm.hbias[i];

	SET_VECTOR_ELT(retval, 5, allocVector(REALSXP, ncol));
	for (int i = 0; i < ncol; i++)
		REAL(VECTOR_ELT(retval, 5))[i] = rbm.vbias[i];

	SET_VECTOR_ELT(retval, 6, allocMatrix(REALSXP, nhid, ncol));
	for (int i = 0; i < nhid; i++)
		for (int j = 0; j < ncol; j++)
			REAL(VECTOR_ELT(retval, 6))[i * ncol + j] = rbm.vel_W[i][j];

	SET_VECTOR_ELT(retval, 7, allocVector(REALSXP, nhid));
	for (int i = 0; i < nhid; i++)
		REAL(VECTOR_ELT(retval, 7))[i] = rbm.vel_h[i];

	SET_VECTOR_ELT(retval, 8, allocVector(REALSXP, ncol));
	for (int i = 0; i < ncol; i++)
		REAL(VECTOR_ELT(retval, 8))[i] = rbm.vel_v[i];

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
	for (int i = 0; i < nrow; i++) free(train_X_p[i]);
	free(train_X_p);

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
	create_RBM (&rbm, 0, ncol, nhid, W, hbias, vbias);

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
	double** test_X_p = malloc(sizeof(double*) * 6);
	for (int i = 0; i < 6; i++) test_X_p[i] = test_X[i];

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

