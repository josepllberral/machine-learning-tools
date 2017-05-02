/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                         */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Function Implementations

#include "cnn.h"

/*---------------------------------------------------------------------------*/
/* SIGMOID LAYERS                                                            */
/*---------------------------------------------------------------------------*/

// Forward through a sigmoid function
//	param x :	Numeric matrix (batch_size, n_units)
//	returns :	Numeric matrix (batch_size, n_units)
//	updates :	sigm_layer
gsl_matrix* forward_sigm (SIGM* sigm, gsl_matrix* x)
{
	// Perform operation: a <- sigmoid(x)
	gsl_matrix* a = gsl_matrix_alloc(x->size1, x->size2);
	matrix_sigmoid(x, a);
	

	// Save sigmoid into Layer
	gsl_matrix_memcpy(sigm->a, a);

	return a;
}

// Backward through the sigmoid layer
//	param x :	Numeric matrix (batch_size, n_units)
//	returns :	Numeric matrix (batch_data, n_units)
gsl_matrix* backward_sigm (SIGM* sigm, gsl_matrix* dy)
{
	// Perform operation: dx <- sigm$a * (1 - sigm$a) * dy
	gsl_matrix* dx = gsl_matrix_alloc(dy->size1, dy->size2);
	gsl_matrix_memcpy(dx, sigm->a);
	gsl_matrix_scale (dx, -1.0);
	gsl_matrix_add_constant(dx, 1.0);
	gsl_matrix_mul_elements(dx, sigm->a);
	gsl_matrix_mul_elements(dx, dy);
	return dx;
}

// Updates the Sigmoid Layer (Does Nothing)
void get_updates_sigm (SIGM* sigm, double lr)
{
	return;
}

// Initializes a Sigmoid layer
void create_SIGM (SIGM* sigm, int n_units, int batch_size)
{
	sigm->batch_size = batch_size;
	sigm->n_units = n_units;
	sigm->a = gsl_matrix_calloc(batch_size, n_units);
}

// Destructor of Sigmoid Layer
void free_SIGM (SIGM* sigm)
{
	gsl_matrix_free(sigm->a);
}


// Function to copy a Sigmoid layer
// Important: destination must NOT be initialized
void copy_SIGM (SIGM* destination, SIGM* origin)
{
	destination->batch_size = origin->batch_size;
	destination->n_units = origin->n_units;

	destination->a = gsl_matrix_alloc(origin->a->size1, origin->a->size2);
	gsl_matrix_memcpy(destination->a, origin->a);
}

// Function to compare a SIGM layer
int compare_SIGM (SIGM* C1, SIGM* C2)
{
	int equal = 1;

	if (
		C1->batch_size != C2->batch_size ||
		C1->n_units != C2->n_units
	) equal = 0;

	equal = equal * gsl_matrix_equal(C1->a, C2->a);

	return equal;
}

