/*----------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                          */
/*----------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Function Implementations

#include "cnn.h"

/*----------------------------------------------------------------------------*/
/* GAUSSIAN BERNOULLY RESTRICTED BOLTZMANN MACHINES LAYER                     */
/*----------------------------------------------------------------------------*/

// This function passes from Visible State to Hidden Probabilities
void vs2hp_rbml(RBML* rbml, gsl_matrix* v_sample, gsl_matrix** h_mean, gsl_matrix** h_sample)
{
	int nrow = v_sample->size1;

	gsl_matrix* pre_sigmoid = gsl_matrix_calloc(nrow, rbml->n_hidden);
	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, v_sample, rbml->W, 1.0, pre_sigmoid);

	gsl_matrix * M_bias = gsl_matrix_calloc(nrow, rbml->n_hidden);
	for(int i = 0; i < nrow; i++) gsl_matrix_set_row(M_bias, i, rbml->hbias);
	gsl_matrix_add(pre_sigmoid, M_bias);

	matrix_sigmoid(pre_sigmoid, *h_mean);
	matrix_bernoulli(*h_mean, *h_sample);

	gsl_matrix_free(M_bias);
	gsl_matrix_free(pre_sigmoid);
}

// This function passes from Hidden State to Visible Probabilities
void hs2vp_rbml(RBML* rbml, gsl_matrix* h_sample, gsl_matrix** v_mean, gsl_matrix** v_sample)
{
	int nrow = h_sample->size1;

	gsl_matrix* pre_sigmoid = gsl_matrix_calloc(nrow, rbml->n_visible);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, h_sample, rbml->W, 1.0, pre_sigmoid);

	gsl_matrix * M_bias = gsl_matrix_calloc(nrow, rbml->n_visible);
	for(int i = 0; i < nrow; i++) gsl_matrix_set_row(M_bias, i, rbml->vbias);
	gsl_matrix_add(pre_sigmoid, M_bias);

	gsl_matrix_memcpy(*v_mean, pre_sigmoid);
	gsl_matrix_memcpy(*v_sample, pre_sigmoid);

	gsl_matrix_free(M_bias);
	gsl_matrix_free(pre_sigmoid);
}

// Forward for a GB-RBM layer
//	param x :	Numeric matrix (batch_size, n_visible)
//	returns :	Numeric matrix (batch_size, n_hidden)
//	updates :	gb-rbm_layer
gsl_matrix* forward_rbml(RBML* rbml, gsl_matrix* x)
{
	// Positive Phase
	gsl_matrix* ph_means = gsl_matrix_calloc(rbml->batch_size, rbml->n_hidden);
	gsl_matrix* ph_sample = gsl_matrix_calloc(rbml->batch_size, rbml->n_hidden);
	vs2hp_rbml(rbml, x, &ph_means, &ph_sample);

	// Save "x" and "ph_means" for back-propagation
	gsl_matrix_free(rbml->x);
	rbml->x = gsl_matrix_calloc(x->size1, x->size2);
	gsl_matrix_memcpy(rbml->x, x);

	gsl_matrix_free(rbml->ph_means);
	rbml->ph_means = gsl_matrix_calloc(ph_means->size1, ph_means->size2);
	gsl_matrix_memcpy(rbml->ph_means, ph_means);

	gsl_matrix_free(ph_means);

	return ph_sample;
}

// Backpropagation for a GB-RBM layer
//	param dy :	Numeric vector <n_hidden>
//	returns  :	Numeric vector <n_visible>
//	updates  :	gb-rbm_layer
gsl_matrix* backward_rbml(RBML* rbml, gsl_matrix* dy)
{
	// Negative Phase
	gsl_matrix* nv_means = gsl_matrix_calloc(rbml->batch_size, rbml->n_visible);
	gsl_matrix* nv_sample = gsl_matrix_calloc(rbml->batch_size, rbml->n_visible);
	gsl_matrix* nh_means = gsl_matrix_calloc(rbml->batch_size, rbml->n_hidden);
	gsl_matrix* nh_sample = gsl_matrix_calloc(rbml->batch_size, rbml->n_hidden);

	gsl_matrix_memcpy(nh_sample, dy);
	for (int step = 0; step < rbml->n_gibbs; step++)
	{
		hs2vp_rbml(rbml, nh_sample, &nv_means, &nv_sample);
		vs2hp_rbml(rbml, nv_sample, &nh_means, &nh_sample);
	}

	// Compute gradients: Delta_W, Delta_h, Delta_v
	gsl_matrix* identity_h = gsl_matrix_alloc(rbml->n_hidden, rbml->n_hidden);
	gsl_matrix_set_all(identity_h, 1.0);
	gsl_matrix_set_identity(identity_h);

	gsl_matrix* identity_v = gsl_matrix_alloc(rbml->n_visible, rbml->n_visible);
	gsl_matrix_set_all(identity_v, 1.0);
	gsl_matrix_set_identity(identity_v);
	
	gsl_vector* ones = gsl_vector_alloc(rbml->batch_size);
	gsl_vector_set_all(ones, 1.0);

	gsl_matrix* delta_W = gsl_matrix_calloc(rbml->n_hidden, rbml->n_visible);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, rbml->ph_means, rbml->x, 1.0, delta_W);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, nh_means, nv_sample, -1.0, delta_W);
	gsl_matrix_memcpy(rbml->grad_W, delta_W);

	gsl_matrix* pre_delta_v = gsl_matrix_calloc(rbml->batch_size, rbml->n_visible);
	gsl_matrix* pre_delta_h = gsl_matrix_calloc(rbml->batch_size, rbml->n_hidden);

	gsl_vector* delta_v = gsl_vector_calloc(rbml->n_visible);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, rbml->x, identity_v, -1.0, pre_delta_v);
	gsl_blas_dgemv(CblasTrans, 1.0, pre_delta_v, ones, 1.0, delta_v);
	gsl_vector_memcpy(rbml->grad_vbias, delta_v);

	gsl_vector* delta_h = gsl_vector_calloc(rbml->n_hidden);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, rbml->ph_means, identity_h, -1.0, pre_delta_h);
	gsl_blas_dgemv(CblasTrans, 1.0, pre_delta_h, ones, 1.0, delta_h);
	gsl_vector_memcpy(rbml->grad_hbias, delta_h);

	// approximation to the reconstruction error: sum over dimensions, mean over cases
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, rbml->x, identity_v, -1.0, nv_means); // we don't need nv_mean anymore
	gsl_matrix* pow_res = gsl_matrix_alloc(rbml->batch_size, rbml->n_visible);
	gsl_matrix_memcpy(pow_res, nv_means);
	gsl_matrix_mul_elements(pow_res, nv_means);
	gsl_vector* pow_sum = gsl_vector_calloc(rbml->n_visible);
	gsl_blas_dgemv(CblasTrans, 1.0, pow_res, ones, 1.0, pow_sum);

	for(int j = 0; j < rbml->n_visible; j++) rbml->loss += gsl_vector_get(pow_sum, j);
	rbml->loss /= rbml->batch_size;
	
	// Free the used space
	gsl_matrix_free(nh_means);
	gsl_matrix_free(nh_sample);
	gsl_matrix_free(nv_means);
	
	gsl_matrix_free(identity_h);
	gsl_matrix_free(identity_v);
	gsl_vector_free(ones);

	gsl_matrix_free(pre_delta_v);
	gsl_matrix_free(pre_delta_h);
	
	gsl_matrix_free(delta_W);
	gsl_vector_free(delta_v);
	gsl_vector_free(delta_h);

	gsl_matrix_free(pow_res);
	gsl_vector_free(pow_sum);

	return nv_sample;
}

// Updates the GB-RBM Layer
void get_updates_rbml (RBML* rbml, double lr)
{
	double learn_factor = -1.0 * lr / rbml->batch_size;
	gsl_matrix_scale(rbml->grad_W, learn_factor);
	gsl_matrix_add(rbml->W, rbml->grad_W);
	gsl_blas_daxpy(learn_factor, rbml->grad_vbias, rbml->vbias);
	gsl_blas_daxpy(learn_factor, rbml->grad_hbias, rbml->hbias);
}

// Evaluates a DBN using a RBM for input and labels (forward and backward)
//	param x       : Numeric matrix
//	param targets : Numeric matrix (standard interface, not used here)
//	param lr      : Number 
//	param loss    : Number (pointer) for Loss
//	param accl    : Number (standard interface, not used here)
//	returns       : Numeric matrix
//	updates       : xent_layer
gsl_matrix* evaluate_rbml (RBML* rbml, gsl_matrix* output, gsl_matrix* targets, double lr, double* loss, double* accl)
{
	gsl_matrix* pred_y = forward_rbml(rbml, output);
	gsl_matrix* results = backward_rbml(rbml, pred_y);

	get_updates_rbml(rbml, lr);

	*(loss) = rbml->loss;

	gsl_matrix_free(pred_y);
	gsl_matrix_free(targets);
	
	return results;
}

// Initializes a GB-RBM layer
void create_RBML (RBML* rbml, int n_visible, int n_hidden, double scale, int n_gibbs, int batch_size)
{
	rbml->batch_size = batch_size;
	rbml->n_hidden = n_hidden;
	rbml->n_visible = n_visible;
	
	rbml->n_gibbs = n_gibbs;

	rbml->W = matrix_normal(n_hidden, n_visible, 0, 1, scale);
	rbml->grad_W = gsl_matrix_calloc(n_hidden, n_visible);

	rbml->vbias = gsl_vector_calloc(n_visible);
	rbml->grad_vbias = gsl_vector_calloc(n_visible);
	
	rbml->hbias = gsl_vector_calloc(n_hidden);
	rbml->grad_hbias = gsl_vector_calloc(n_hidden);

	rbml->x = gsl_matrix_calloc(batch_size, n_visible);
	rbml->ph_means = gsl_matrix_calloc(batch_size, n_hidden);
	
	rbml->loss = -1;
}

// Destructor of GB-RBM Layer
void free_RBML (RBML* rbml)
{
	// Free weights matrix and bias
	gsl_matrix_free(rbml->W);
	gsl_matrix_free(rbml->grad_W);
	gsl_vector_free(rbml->vbias);
	gsl_vector_free(rbml->grad_vbias);
	gsl_vector_free(rbml->hbias);
	gsl_vector_free(rbml->grad_hbias);
	gsl_matrix_free(rbml->x);
	gsl_matrix_free(rbml->ph_means);
}

// Function to copy a GB-RBM layer
// Important: destination must NOT be initialized
void copy_RBML (RBML* destination, RBML* origin)
{
	destination->batch_size = origin->batch_size;
	destination->n_hidden = origin->n_hidden;
	destination->n_visible = origin->n_visible;
	
	destination->n_gibbs = origin->n_gibbs;
	
	destination->loss = origin->loss;

	destination->W = gsl_matrix_alloc(origin->n_hidden, origin->n_visible);
	destination->grad_W = gsl_matrix_alloc(origin->n_hidden, origin->n_visible);
	destination->x = gsl_matrix_alloc(origin->batch_size, origin->n_visible);
	destination->ph_means = gsl_matrix_alloc(origin->batch_size, origin->n_hidden);

	gsl_matrix_memcpy(destination->W, origin->W);
	gsl_matrix_memcpy(destination->grad_W, origin->grad_W);
	gsl_matrix_memcpy(destination->x, origin->x);
	gsl_matrix_memcpy(destination->ph_means, origin->ph_means);

	destination->hbias = gsl_vector_alloc(origin->n_hidden);
	destination->grad_hbias = gsl_vector_alloc(origin->n_hidden);
	gsl_vector_memcpy(destination->hbias, origin->hbias);
	gsl_vector_memcpy(destination->grad_hbias, origin->grad_hbias);
	
	destination->vbias = gsl_vector_alloc(origin->n_visible);
	destination->grad_vbias = gsl_vector_alloc(origin->n_visible);
	gsl_vector_memcpy(destination->vbias, origin->vbias);
	gsl_vector_memcpy(destination->grad_vbias, origin->grad_vbias);
}

// Function to compare a GB-RBM layer
int compare_RBML (RBML* C1, RBML* C2)
{
	int equal = 1;

	if (
		C1->batch_size != C2->batch_size ||
		C1->n_hidden != C2->n_hidden ||
		C1->n_visible != C2->n_visible ||
		C1->n_gibbs != C2->n_gibbs ||
		C1->loss != C2->loss
	) equal = 0;

	equal += 1 - gsl_matrix_equal(C1->W, C2->W);
	equal += 1 - gsl_matrix_equal(C1->grad_W, C2->grad_W);
	equal += 1 - gsl_vector_equal(C1->vbias, C2->vbias);
	equal += 1 - gsl_vector_equal(C1->grad_vbias, C2->grad_vbias);
	equal += 1 - gsl_vector_equal(C1->hbias, C2->hbias);
	equal += 1 - gsl_vector_equal(C1->grad_hbias, C2->grad_hbias);
	equal += 1 - gsl_matrix_equal(C1->x, C2->x);
	equal += 1 - gsl_matrix_equal(C1->ph_means, C2->ph_means);

	return equal;
}

