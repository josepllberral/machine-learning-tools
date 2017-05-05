/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                         */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Function Implementations

#include "cnn.h"

/*---------------------------------------------------------------------------*/
/* HYPERBOLIC TANGENT LAYER                                                  */
/*---------------------------------------------------------------------------*/

// Forward through a HyperTan function
//	param x :	Numeric matrix (batch_size, n_units)
//	returns :	Numeric matrix (batch_size, n_units)
gsl_matrix* forward_tanh (TANH* tanh, gsl_matrix* x)
{
	// Perform operation: y <- (exp(x) - exp(-x)) / (exp(x) + exp(-x))
	gsl_matrix* nx = gsl_matrix_alloc(x->size1, x->size2);
	gsl_matrix_memcpy(nx, x);
	gsl_matrix_scale(nx, -1.0);

	gsl_matrix* x_nexp = gsl_matrix_alloc(x->size1, x->size2);
	gsl_matrix* x_exp_1 = gsl_matrix_alloc(x->size1, x->size2);
	gsl_matrix* x_exp_2 = gsl_matrix_alloc(x->size1, x->size2);

	matrix_exponent(nx, x_nexp);
	matrix_exponent(x, x_exp_1);
	gsl_matrix_memcpy(x_exp_2, x_exp_1);

	gsl_matrix_sub(x_exp_1, x_nexp);
	gsl_matrix_add(x_exp_2, x_nexp);
	gsl_matrix_div_elements(x_exp_1, x_exp_2);

	gsl_matrix_memcpy(tanh->a, x_exp_1);

	return x_exp_1;
}

// Backward through the HyperTan layer
//	param x :	Numeric matrix (batch_size, n_units)
//	returns :	Numeric matrix (batch_data, n_units)
gsl_matrix* backward_tanh (TANH* tanh, gsl_matrix* dy)
{
	// Passes data back: dx <- (1 - ((exp(a) - exp(-a)) / (exp(a) + exp(-a)))) * dy
	gsl_matrix* na = gsl_matrix_alloc(tanh->a->size1, tanh->a->size2);
	gsl_matrix_memcpy(na, tanh->a);
	gsl_matrix_scale(na, -1.0);

	gsl_matrix* a_nexp = gsl_matrix_alloc(na->size1, na->size2);
	gsl_matrix* a_exp_1 = gsl_matrix_alloc(na->size1, na->size2);
	gsl_matrix* a_exp_2 = gsl_matrix_alloc(na->size1, na->size2);

	matrix_exponent(na, a_nexp);
	matrix_exponent(tanh->a, a_exp_1);
	gsl_matrix_memcpy(a_exp_2, a_exp_1);

	gsl_matrix_sub(a_exp_1, a_nexp);
	gsl_matrix_add(a_exp_2, a_nexp);
	gsl_matrix_div_elements(a_exp_1, a_exp_2);

	gsl_matrix_scale(a_exp_1, -1.0);
	gsl_matrix_add_constant(a_exp_1, 1);
	gsl_matrix_mul_elements(a_exp_1, dy);

	return a_exp_1;
}

// Updates the HyperTan Layer (Does Nothing)
void get_updates_tanh (TANH* tanh, double lr)
{
	return;
}

// Initializes a HyperTan layer
void create_TANH (TANH* tanh, int n_units, int batch_size)
{
	tanh->batch_size = batch_size;
	tanh->n_units = n_units;
	tanh->a = gsl_matrix_calloc(batch_size, n_units);
}

// Destructor of HyperTan Layer (Does Nothing)
void free_TANH (TANH* tanh)
{
	gsl_matrix_free(tanh->a);
	return;
}

// Function to copy a HyperTan layer
// Important: destination must NOT be initialized
void copy_TANH (TANH* destination, TANH* origin)
{
	destination->batch_size = origin->batch_size;
	destination->n_units = origin->n_units;

	destination->a = gsl_matrix_alloc(origin->a->size1, origin->a->size2);
	gsl_matrix_memcpy(destination->a, origin->a);
}

// Function to compare a TANH layer
int compare_TANH (TANH* C1, TANH* C2)
{
	int equal = 1;

	if (
		C1->batch_size != C2->batch_size ||
		C1->n_units != C2->n_units
	) equal = 0;

	equal = equal * gsl_matrix_equal(C1->a, C2->a);

	return equal;
}

