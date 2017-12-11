/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                         */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Function Implementations

#include "cnn.h"

/*---------------------------------------------------------------------------*/
/* LINEAR LAYER                                                              */
/*---------------------------------------------------------------------------*/

// Forward for a linear layer
//	param x :	Numeric matrix (batch_size, n_visible)
//	returns :	Numeric matrix (batch_size, n_hidden)
//	updates :	linear_layer
gsl_matrix* forward_line(LINE* line, gsl_matrix* x)
{
	// Save "x" for back-propagation
	gsl_matrix_free(line->x);
	line->x = gsl_matrix_calloc(x->size1, x->size2);
	gsl_matrix_memcpy(line->x, x);

	// Perform operation: y <- (x %*% t(line$W)) + line$b;
	gsl_matrix* y = gsl_matrix_calloc(line->batch_size, line->n_hidden);
	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, x, line->W, 1.0, y);

	gsl_matrix * M_bias = gsl_matrix_calloc(line->batch_size, line->n_hidden);
	for(int i = 0; i < line->batch_size; i++) gsl_matrix_set_row(M_bias, i, line->b);
	gsl_matrix_add(y, M_bias);

	gsl_matrix_free(M_bias);

	return y;
}

// Backpropagation for a linear layer
//	param dy :	Numeric vector <n_hidden>
//	returns  :	Numeric vector <n_visible>
//	updates  :	linear_layer
gsl_matrix* backward_line(LINE* line, gsl_matrix* dy)
{
	// Perform operation: dx <- dy %*% line$W
	gsl_matrix* dx = gsl_matrix_calloc(line->batch_size, line->n_visible);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, dy, line->W, 1.0, dx);

	// Perform operation: grad_W <- t(dy) %*% line$x
	gsl_matrix* grad_W = gsl_matrix_calloc(line->n_hidden, line->n_visible);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, dy, line->x, 1.0, grad_W);

	// Perform operation: grad_b <- colSums(dy)
	gsl_vector* grad_b = gsl_vector_calloc(line->n_hidden);
	for(int i = 0; i < line->batch_size; i++)
	{
		gsl_vector* aux = gsl_vector_calloc(line->n_hidden);
		gsl_matrix_get_row(aux, dy, i);
		gsl_vector_add(grad_b, aux);
		gsl_vector_free(aux);
	}

	// Update gradient values
	gsl_matrix_memcpy(line->grad_W, grad_W);
	gsl_vector_memcpy(line->grad_b, grad_b);

	gsl_matrix_free(grad_W);
	gsl_vector_free(grad_b);

	return dx;
}

// Updates the Linear Layer
void get_updates_line (LINE* line, double lr)
{
	double learn_factor = -1.0 * lr / line->batch_size;
	gsl_matrix_scale(line->grad_W, learn_factor);
	gsl_matrix_add(line->W, line->grad_W);
	gsl_blas_daxpy(learn_factor, line->grad_b, line->b);
}

// Initializes a Linear layer
void create_LINE (LINE* line, int n_visible, int n_hidden, double scale, int batch_size)
{
	line->batch_size = batch_size;
	line->n_hidden = n_hidden;
	line->n_visible = n_visible;

	line->W = matrix_normal(n_hidden, n_visible, 0, 1, scale);
	line->grad_W = gsl_matrix_calloc(n_hidden, n_visible);

	line->b = gsl_vector_calloc(n_hidden);
	line->grad_b = gsl_vector_calloc(n_hidden);

	line->x = gsl_matrix_calloc(batch_size, n_visible);
}

// Destructor of Linear Layer
void free_LINE (LINE* line)
{
	// Free weights matrix and bias
	gsl_matrix_free(line->W);
	gsl_matrix_free(line->grad_W);
	gsl_vector_free(line->b);
	gsl_vector_free(line->grad_b);
	gsl_matrix_free(line->x);
}

// Function to copy a CONV layer
// Important: destination must NOT be initialized
void copy_LINE (LINE* destination, LINE* origin)
{
	destination->batch_size = origin->batch_size;
	destination->n_hidden = origin->n_hidden;
	destination->n_visible = origin->n_visible;

	destination->W = gsl_matrix_alloc(origin->n_hidden, origin->n_visible);
	destination->grad_W = gsl_matrix_alloc(origin->n_hidden, origin->n_visible);
	destination->x = gsl_matrix_alloc(origin->batch_size, origin->n_visible);

	gsl_matrix_memcpy(destination->W, origin->W);
	gsl_matrix_memcpy(destination->grad_W, origin->grad_W);
	gsl_matrix_memcpy(destination->x, origin->x);

	destination->b = gsl_vector_alloc(origin->n_hidden);
	destination->grad_b = gsl_vector_alloc(origin->n_hidden);
	gsl_vector_memcpy(destination->b, origin->b);
	gsl_vector_memcpy(destination->grad_b, origin->grad_b);
}

// Function to compare a CONV layer
int compare_LINE (LINE* C1, LINE* C2)
{
	int equal = 1;

	if (
		C1->batch_size != C2->batch_size ||
		C1->n_hidden != C2->n_hidden ||
		C1->n_visible != C2->n_visible
	) equal = 0;

	equal += 1 - gsl_matrix_equal(C1->W, C2->W);
	equal += 1 - gsl_matrix_equal(C1->grad_W, C2->grad_W);
	equal += 1 - gsl_vector_equal(C1->b, C2->b);
	equal += 1 - gsl_vector_equal(C1->grad_b, C2->grad_b);
	equal += 1 - gsl_matrix_equal(C1->x, C2->x);

	return equal;
}

