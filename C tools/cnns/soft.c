/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C                                               */
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
gsl_matrix* forward_soft (SOFT* soft, gsl_matrix* x)
{
	// Perform operation: y <- exp(x) / rowSums(exp(x))
	gsl_matrix* x_exp = gsl_matrix_alloc(x->size1, x->size2);

	matrix_exponent(x, x_exp);
	for (int i = 0; i < x_exp->size1; i++)
	{
		gsl_vector* aux = gsl_vector_alloc(x_exp->size2);
		gsl_matrix_get_row(aux, x_exp, i);

		double acc = 0;
		for (int j = 0; j < x_exp->size2; j++) acc += gsl_vector_get(aux, j);
		gsl_vector_scale(aux, 1.0 / acc);

		gsl_matrix_set_row(x_exp, i, aux);
		gsl_vector_free(aux);
	}

	return x_exp;
}

// Backward through the softmax layer
//	param x :	Numeric matrix (batch_size, n_units)
//	returns :	Numeric matrix (batch_data, n_units)
gsl_matrix* backward_soft (SOFT* soft, gsl_matrix* dy)
{
	// Passes data back
	gsl_matrix* dx = gsl_matrix_alloc(dy->size1, dy->size2);
	gsl_matrix_memcpy(dx, dy);

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
}

// Destructor of SoftMax Layer (Does Nothing)
void free_SOFT (SOFT* soft)
{
	return;
}


// Function to copy a SoftMax layer
// Important: destination must NOT be initialized
void copy_SOFT (SOFT* destination, SOFT* origin)
{
	destination->batch_size = origin->batch_size;
	destination->n_units = origin->n_units;
}

// Function to compare a SOFT layer
int compare_SOFT (SOFT* C1, SOFT* C2)
{
	int equal = 1;

	if (
		C1->batch_size != C2->batch_size ||
		C1->n_units != C2->n_units
	) equal = 0;

	return equal;
}
