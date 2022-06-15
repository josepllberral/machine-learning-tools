/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                         */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Function Implementations

#include "cnn.h"

/*---------------------------------------------------------------------------*/
/* MEAN-SQUARED-ERROR LOSS LAYERS                                            */
/*---------------------------------------------------------------------------*/

// Computes the mean_squared_error loss for input and labels
//	param x       :	Numeric Matrix
//	param targets : Matrix of Solutions, it could be a matrix (1,S)...
//	returns       :	Numeric Matrix
//      updates       : msel layer
gsl_matrix* forward_msel (MSEL* msel, gsl_matrix* x, gsl_matrix* targets)
{
	// Performs: cost <- 0.5 * (x-t)**2 / num_batches
	gsl_matrix* aux = gsl_matrix_alloc(x->size1, x->size2);
	gsl_matrix* aux2 = gsl_matrix_calloc(x->size1, x->size2);
	gsl_matrix_memcpy(aux, x);
	gsl_matrix_sub(aux, targets);
	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 0.5 / x->size1, aux, aux, 1.0, aux2);
	
	// Performs: cost <- mean(colsum(cost))
	msel->loss = matrix_sum(aux2) / aux2->size2;

	gsl_matrix_free(aux);
	gsl_matrix_memcpy(aux2, x);

	return aux2;
}

// Backpropagation of mean_squared_error Layer
//	param dy      :	Numeric Matrix
//	param targets : Matrix of Solutions, it could be a matrix (1,S)...
//	returns       :	Numeric Matrix
gsl_matrix* backward_msel (MSEL* msel, gsl_matrix* dy, gsl_matrix* targets)
{
	// Performs: dy <- (1 / num_batches) * (y - t)
	gsl_matrix* dx = gsl_matrix_alloc(dy->size1, dy->size2);
	gsl_matrix_memcpy(dx, dy);
	gsl_matrix_sub(dx, targets);
	gsl_matrix_scale(dx, 1.0 / dy->size1);

	return dx;
}

// Updates the MSE Loss Layer (Does Nothing)
void get_updates_msel (MSEL* msel, double lr)
{
	return;
}

// Initializes a MSE Loss layer
void create_MSEL (MSEL* msel)
{
	msel->loss = 0;
}

// Destructor of MSE Loss Layer (Does Nothing)
void free_MSEL (MSEL* msel)
{
	return;
}

// Function to copy a MSE Loss layer
// Important: destination must NOT be initialized
void copy_MSEL (MSEL* destination, MSEL* origin)
{
	destination->loss = origin->loss;
}

// Function to compare a MSE Loss layer
int compare_MSEL (MSEL* C1, MSEL* C2)
{
	int equal = 1;

	if (
		C1->loss != C2->loss
	) equal = 0;

	return equal;
}

