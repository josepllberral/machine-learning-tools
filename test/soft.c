/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                         */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Function Implementations

#include "cnn.h"

/*---------------------------------------------------------------------------*/
/* SOFTMAX LAYERS                                                            */
/*---------------------------------------------------------------------------*/

// Forward through a sigmoid function
//	param x :	Numeric matrix (batch_size, n_units)
//	returns :	Numeric matrix (batch_size, n_units)
//	updates :	softmax_layer
gsl_matrix* forward_soft (SOFT* soft, gsl_matrix* x)
{
	// Perform operation: a <- sigmoid(x)
	gsl_matrix* a = gsl_matrix_alloc(x->size1, x->size2);
	matrix_sigmoid(x, a);

	// Save sigmoid into Layer
	gsl_matrix_memcpy(soft->a, a);

	return a;
}

// Backward through the softmax layer
//	param x :	Numeric matrix (batch_size, n_units)
//	returns :	Numeric matrix (batch_data, n_units)
gsl_matrix* backward_soft (SOFT* soft, gsl_matrix* dy)
{
	// Perform operation: dx <- soft$a * (1 - soft$a) * dy
	gsl_matrix* dx = gsl_matrix_alloc(dy->size1, dy->size2);
	gsl_matrix_memcpy(dx, soft->a);
	gsl_matrix_scale (dx, -1.0);
	gsl_matrix_add_constant(dx, 1.0);
	gsl_matrix_mul_elements(dx, soft->a);
	gsl_matrix_mul_elements(dx, dy);
	return dx;
}

// Updates the SoftMax Layer (Does Nothing)
void get_updates_soft (SOFT* soft, double lr)
{
	return;
}

// Initializes a SoftMax layer
void create_SOFT (SOFT* soft, int n_units, int batch_size)
{
	soft->batch_size = batch_size;
	soft->n_units = n_units;
	soft->a = gsl_matrix_calloc(batch_size, n_units);
}

// Destructor of SoftMax Layer
void free_SOFT (SOFT* soft)
{
	gsl_matrix_free(soft->a);
}


// Function to copy a SoftMax layer
// Important: destination must NOT be initialized
void copy_SOFT (SOFT* destination, SOFT* origin)
{
	destination->batch_size = origin->batch_size;
	destination->n_units = origin->n_units;

	destination->a = gsl_matrix_alloc(origin->batch_size, origin->n_units);
	gsl_matrix_memcpy(destination->a, origin->a);
}

// Function to compare a SOFT layer
int compare_SOFT (SOFT* C1, SOFT* C2)
{
	int equal = 1;

	if (
		C1->batch_size != C2->batch_size ||
		C1->n_units != C2->n_units
	) equal = 0;

	equal = equal * gsl_matrix_equal(C1->a, C2->a);

	return equal;
}

