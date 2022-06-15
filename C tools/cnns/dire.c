/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C                                               */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Function Implementations

#include "cnn.h"

/*---------------------------------------------------------------------------*/
/* DIRECT LAYERS (BUFFERS)                                                   */
/*---------------------------------------------------------------------------*/

// Forward through a direct function (input -> output)
//	param x :	Numeric matrix (batch_size, n_units)
//	returns :	Numeric matrix (batch_size, n_units)
//	updates :	dire_layer
gsl_matrix* forward_dire (DIRE* dire, gsl_matrix* x)
{
	// Pushes forward the input
	gsl_matrix* y = gsl_matrix_alloc(x->size1, x->size2);
	gsl_matrix_memcpy(y, x);

	// Keep a copy of x
	gsl_matrix_free(dire->buff_x);
	dire->buff_x = gsl_matrix_alloc(x->size1, x->size2);
	gsl_matrix_memcpy(dire->buff_x, x);

	return y;
}

// Backward through a direct function (output -> input)
//	param x :	Numeric matrix (batch_size, n_units)
//	returns :	Numeric matrix (batch_data, n_units)
gsl_matrix* backward_dire (DIRE* dire, gsl_matrix* dy)
{
	// Returns back the output
	gsl_matrix* dx = gsl_matrix_alloc(dy->size1, dy->size2);
	gsl_matrix_memcpy(dx, dy);

	// Keep a copy of dy
	gsl_matrix_free(dire->buff_dy);
	dire->buff_dy = gsl_matrix_alloc(dy->size1, dy->size2);
	gsl_matrix_memcpy(dire->buff_dy, dy);

	return dx;
}

// Updates the Direct Layer (Does Nothing)
void get_updates_dire (DIRE* dire, double lr)
{
	return;
}

// Initializes a Direct layer
void create_DIRE (DIRE* dire, int n_units, int batch_size)
{
	dire->batch_size = batch_size;
	dire->n_units = n_units;
	dire->buff_x = gsl_matrix_calloc(batch_size, n_units);
	dire->buff_dy = gsl_matrix_calloc(batch_size, n_units);
}

// Destructor of Direct Layer
void free_DIRE (DIRE* dire)
{
	gsl_matrix_free(dire->buff_x);
	gsl_matrix_free(dire->buff_dy);
}


// Function to copy a Direct layer
// Important: destination must NOT be initialized
void copy_DIRE (DIRE* destination, DIRE* origin)
{
	destination->batch_size = origin->batch_size;
	destination->n_units = origin->n_units;

	destination->buff_x = gsl_matrix_alloc(origin->buff_x->size1, origin->buff_x->size2);
	gsl_matrix_memcpy(destination->buff_x, origin->buff_x);

	destination->buff_dy = gsl_matrix_alloc(origin->buff_dy->size1, origin->buff_dy->size2);
	gsl_matrix_memcpy(destination->buff_dy, origin->buff_dy);
}

// Function to compare a Direct layer
int compare_DIRE (DIRE* C1, DIRE* C2)
{
	int equal = 1;

	if (
		C1->batch_size != C2->batch_size ||
		C1->n_units != C2->n_units
	) equal = 0;

	equal = equal * gsl_matrix_equal(C1->buff_x, C2->buff_x);
	equal = equal * gsl_matrix_equal(C1->buff_dy, C2->buff_dy);

	return equal;
}
