/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                         */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Function Implementations

#include "cnn.h"

/*---------------------------------------------------------------------------*/
/* CROSS-ENTROPY LOSS LAYERS                                                 */
/*---------------------------------------------------------------------------*/

// Computes the cross-entriopy for input and labels
//	param x       :	Numeric Matrix
//	param targets : Matrix of Solutions, it could be a matrix (1,S)...
//	returns       :	Numeric Matrix
//      updates       : cell layer
gsl_matrix* forward_cell (CELL* cell, gsl_matrix* x, gsl_matrix* targets)
{
	// Performs: l <- -1.0 * targets * log(x + 1e-08)
	gsl_matrix* pre_log = gsl_matrix_calloc(x->size1, x->size2);
	gsl_matrix* l = gsl_matrix_calloc(x->size1, x->size2);
	gsl_matrix_memcpy(pre_log, x);
	gsl_matrix_add_constant(pre_log, 1e-08);
	matrix_log(pre_log, l);
	gsl_matrix_scale(l, -1.0);
	gsl_matrix_mul_elements(l, targets);

	// Performs: mean(apply(l, MARGIN = 1, sum))
	cell->loss = matrix_sum(l) / l->size1;

	gsl_matrix_free(pre_log);
	gsl_matrix_free(l);

	gsl_matrix* y = gsl_matrix_calloc(x->size1, x->size2);
	gsl_matrix_memcpy(y, x);
	return y;
}

// Backpropagation of Cross-Entropy Layer
//	param dy      :	Numeric Matrix
//	param targets : Matrix of Solutions, it could be a matrix (1,S)...
//	returns       :	Numeric Matrix
gsl_matrix* backward_cell (CELL* cell, gsl_matrix* dy, gsl_matrix* targets)
{
	// Performs: dx <- (1.0 / num_batches) * (dy - targets)
	gsl_matrix* dx = gsl_matrix_calloc(dy->size1, dy->size2);
	gsl_matrix_memcpy(dx, dy);
	gsl_matrix_scale(dx, -1.0);
	gsl_matrix_add(dx, targets);
	gsl_matrix_scale(dx, -1.0 / dy->size1);

	return dx;
}

// Updates the C-E Loss Layer (Does Nothing)
void get_updates_cell (CELL* cell, double lr)
{
	return;
}

// Initializes a C-E Loss layer
void create_CELL (CELL* cell)
{
	cell->loss = 0;
}

// Destructor of C-E Loss Layer (Does Nothing)
void free_CELL (CELL* cell)
{
	return;
}

// Function to copy a C-E Loss layer
// Important: destination must NOT be initialized
void copy_CELL (CELL* destination, CELL* origin)
{
	destination->loss = origin->loss;
}

// Function to compare a C-E Loss layer
int compare_CELL (CELL* C1, CELL* C2)
{
	int equal = 1;

	if (
		C1->loss != C2->loss
	) equal = 0;

	return equal;
}

