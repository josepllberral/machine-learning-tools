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
void vs2hp_gbrl(GBRL* gbrl, gsl_matrix* v_sample, gsl_matrix** h_mean, gsl_matrix** h_sample)
{
	int nrow = v_sample->size1;

	gsl_matrix* pre_sigmoid = gsl_matrix_calloc(nrow, gbrl->n_hidden);
	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, v_sample, gbrl->W, 1.0, pre_sigmoid);

	gsl_matrix * M_bias = gsl_matrix_calloc(nrow, gbrl->n_hidden);
	for(int i = 0; i < nrow; i++) gsl_matrix_set_row(M_bias, i, gbrl->hbias);
	gsl_matrix_add(pre_sigmoid, M_bias);

	matrix_sigmoid(pre_sigmoid, *h_mean);
	matrix_bernoulli(*h_mean, *h_sample);

	gsl_matrix_free(M_bias);
	gsl_matrix_free(pre_sigmoid);
}

// This function passes from Hidden State to Visible Probabilities
void hs2vp_gbrl(GBRL* gbrl, gsl_matrix* h_sample, gsl_matrix** v_mean, gsl_matrix** v_sample)
{
	int nrow = h_sample->size1;

	gsl_matrix* pre_sigmoid = gsl_matrix_calloc(nrow, gbrl->n_visible);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, h_sample, gbrl->W, 1.0, pre_sigmoid);

	gsl_matrix * M_bias = gsl_matrix_calloc(nrow, gbrl->n_visible);
	for(int i = 0; i < nrow; i++) gsl_matrix_set_row(M_bias, i, gbrl->vbias);
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
gsl_matrix* forward_gbrl(GBRL* gbrl, gsl_matrix* x)
{
	// Positive Phase
	gsl_matrix* ph_means = gsl_matrix_calloc(gbrl->batch_size, gbrl->n_hidden);
	gsl_matrix* ph_sample = gsl_matrix_calloc(gbrl->batch_size, gbrl->n_hidden);
	vs2hp_gbrl(gbrl, x, &ph_means, &ph_sample);

	// Save "x" and "ph_means" for back-propagation
	gsl_matrix_free(gbrl->x);
	gbrl->x = gsl_matrix_calloc(x->size1, x->size2);
	gsl_matrix_memcpy(gbrl->x, x);

	gsl_matrix_free(gbrl->ph_means);
	gbrl->ph_means = gsl_matrix_calloc(ph_means->size1, ph_means->size2);
	gsl_matrix_memcpy(gbrl->ph_means, ph_means);

	gsl_matrix_free(ph_means);

	return ph_sample;
}

// Backpropagation for a GB-RBM layer
//	param dy :	Numeric vector <n_hidden>
//	returns  :	Numeric vector <n_visible>
//	updates  :	gb-rbm_layer
gsl_matrix* backward_gbrl(GBRL* gbrl, gsl_matrix* dy)
{
	// Negative Phase
	gsl_matrix* nv_means = gsl_matrix_calloc(gbrl->batch_size, gbrl->n_visible);
	gsl_matrix* nv_sample = gsl_matrix_calloc(gbrl->batch_size, gbrl->n_visible);
	gsl_matrix* nh_means = gsl_matrix_calloc(gbrl->batch_size, gbrl->n_hidden);
	gsl_matrix* nh_sample = gsl_matrix_calloc(gbrl->batch_size, gbrl->n_hidden);

	gsl_matrix_memcpy(nh_sample, dy);
	for (int step = 0; step < gbrl->n_gibbs; step++)
	{
		hs2vp_gbrl(gbrl, nh_sample, &nv_means, &nv_sample);
		vs2hp_gbrl(gbrl, nv_sample, &nh_means, &nh_sample);
	}

	// Compute gradients: Delta_W, Delta_h, Delta_v
	gsl_matrix* identity_h = gsl_matrix_alloc(gbrl->n_hidden, gbrl->n_hidden);
	gsl_matrix_set_all(identity_h, 1.0);
	gsl_matrix_set_identity(identity_h);

	gsl_matrix* identity_v = gsl_matrix_alloc(gbrl->n_visible, gbrl->n_visible);
	gsl_matrix_set_all(identity_v, 1.0);
	gsl_matrix_set_identity(identity_v);
	
	gsl_vector* ones = gsl_vector_alloc(gbrl->batch_size);
	gsl_vector_set_all(ones, 1.0);

	gsl_matrix* delta_W = gsl_matrix_calloc(gbrl->n_hidden, gbrl->n_visible);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, gbrl->ph_means, gbrl->x, 1.0, delta_W);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, -1.0, nh_means, nv_sample, 1.0, delta_W);
	gsl_matrix_memcpy(gbrl->grad_W, delta_W);

	gsl_matrix* pre_delta_v = gsl_matrix_calloc(gbrl->batch_size, gbrl->n_visible);
	gsl_matrix* pre_delta_h = gsl_matrix_calloc(gbrl->batch_size, gbrl->n_hidden);

	gsl_vector* delta_v = gsl_vector_calloc(gbrl->n_visible);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, gbrl->x, identity_v, -1.0, pre_delta_v);
	gsl_blas_dgemv(CblasTrans, 1.0, pre_delta_v, ones, 1.0, delta_v);
	gsl_vector_memcpy(gbrl->grad_vbias, delta_v);

	gsl_vector* delta_h = gsl_vector_calloc(gbrl->n_hidden);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, gbrl->ph_means, identity_h, -1.0, pre_delta_h);
	gsl_blas_dgemv(CblasTrans, 1.0, pre_delta_h, ones, 1.0, delta_h);
	gsl_vector_memcpy(gbrl->grad_hbias, delta_h);

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

	return nv_sample;
}

// Updates the GB-RBM Layer
void get_updates_gbrl (GBRL* gbrl, double lr)
{
	double learn_factor = -1.0 * lr / gbrl->batch_size;
	gsl_matrix_scale(gbrl->grad_W, learn_factor);
	gsl_matrix_add(gbrl->W, gbrl->grad_W);
	gsl_blas_daxpy(learn_factor, gbrl->grad_vbias, gbrl->vbias);
	gsl_blas_daxpy(learn_factor, gbrl->grad_hbias, gbrl->hbias);
}

// Initializes a GB-RBM layer
void create_GBRL (GBRL* gbrl, int n_visible, int n_hidden, double scale, int n_gibbs, int batch_size)
{
	gbrl->batch_size = batch_size;
	gbrl->n_hidden = n_hidden;
	gbrl->n_visible = n_visible;
	
	gbrl->n_gibbs = n_gibbs;

	gbrl->W = matrix_normal(n_hidden, n_visible, 0, 1, scale);
	gbrl->grad_W = gsl_matrix_calloc(n_hidden, n_visible);

	gbrl->vbias = gsl_vector_calloc(n_visible);
	gbrl->grad_vbias = gsl_vector_calloc(n_visible);
	
	gbrl->hbias = gsl_vector_calloc(n_hidden);
	gbrl->grad_hbias = gsl_vector_calloc(n_hidden);

	gbrl->x = gsl_matrix_calloc(batch_size, n_visible);
	gbrl->ph_means = gsl_matrix_calloc(batch_size, n_hidden);
}

// Destructor of GB-RBM Layer
void free_GBRL (GBRL* gbrl)
{
	// Free weights matrix and bias
	gsl_matrix_free(gbrl->W);
	gsl_matrix_free(gbrl->grad_W);
	gsl_vector_free(gbrl->vbias);
	gsl_vector_free(gbrl->grad_vbias);
	gsl_vector_free(gbrl->hbias);
	gsl_vector_free(gbrl->grad_hbias);
	gsl_matrix_free(gbrl->x);
	gsl_matrix_free(gbrl->ph_means);
}

// Function to copy a GB-RBM layer
// Important: destination must NOT be initialized
void copy_GBRL (GBRL* destination, GBRL* origin)
{
	destination->batch_size = origin->batch_size;
	destination->n_hidden = origin->n_hidden;
	destination->n_visible = origin->n_visible;
	
	destination->n_gibbs = origin->n_gibbs;

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
int compare_GBRL (GBRL* C1, GBRL* C2)
{
	int equal = 1;

	if (
		C1->batch_size != C2->batch_size ||
		C1->n_hidden != C2->n_hidden ||
		C1->n_visible != C2->n_visible ||
		C1->n_gibbs != C2->n_gibbs
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

