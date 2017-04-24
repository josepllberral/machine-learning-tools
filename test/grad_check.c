/*---------------------------------------------------------------------------*/
/* GRADIENT CHECK for CONVOLUTIONAL NEURAL NETWORKS in C for R               */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// @date 24th April 2017

// Functions to compare gradients

#include "cnn.h"

/*---------------------------------------------------------------------------*/
/* CHECK INPUT GRADIENTS AND W                                               */
/*---------------------------------------------------------------------------*/

// return: 1 = TRUE, 0 = FALSE
int gradclose (gsl_matrix*** x , gsl_matrix*** y, int size01, int size02, double rtol, double atol)
{
	if (rtol == -1) rtol = 1e-05;
	if (atol == -1) atol = 1e-08;

	int is_close = 1;
	for (int i = 0; i < size01; i++)
		for (int j = 0; j < size02; j++)
		{
			int img_h = x[i][j]->size1;
			int img_w = x[i][j]->size2;

			for (int h = 0; h < img_h; h++)
			{
				for (int w = 0; w < img_w; w++)
				{
					double xval = gsl_matrix_get(x[i][j], h, w);
					double yval = gsl_matrix_get(y[i][j], h, w);
					double diff = abs(xval - yval) - atol - rtol * (abs(xval) + abs(yval));
					if (diff >= 0) is_close = 0;
				}
			}
		}

	if (is_close == 0)
	{
		double max_re = -9e+15;
		double max_ae = -9e+15;

		for (int i = 0; i < size01; i++)
			for (int j = 0; j < size02; j++)
			{
				int img_h = x[i][j]->size1;
				for (int h = 0; h < img_h; h++)
				{
					int img_w = x[i][j]->size2;
					for (int w = 0; w < img_w; w++)
					{
						double xval = gsl_matrix_get(x[i][j], h, w);
						double yval = gsl_matrix_get(y[i][j], h, w);

						double rel_error = 0;
						double denom = abs(xval) - abs(yval);
						if (denom != 0) rel_error = abs(xval - yval) / denom;

						if (rel_error > max_re) max_re = rel_error;
						if (abs(xval - yval) > max_ae) max_ae = abs(xval - yval);
					}
				}
			}
	        printf("rel_error=%f, abs_error=%f, rtol=%f, atol=%f\n", max_re, max_ae, rtol, atol);
	}
	return is_close;
}

// return: 1 = TRUE, 0 = FALSE
int gradclose_vec (gsl_vector* x , gsl_vector* y, int size1, double rtol, double atol)
{
	if (rtol == -1) rtol = 1e-05;
	if (atol == -1) atol = 1e-08;

	int is_close = 1;
	for (int i = 0; i < size1; i++)
	{
		double xval = gsl_vector_get(x, i);
		double yval = gsl_vector_get(y, i);
		double diff = abs(xval - yval) - atol - rtol * (abs(xval) + abs(yval));
		if (diff >= 0) is_close = 0;
	}

	if (is_close == 0)
	{
		double max_re = -9e+15;
		double max_ae = -9e+15;

		for (int i = 0; i < size1; i++)
		{
			double xval = gsl_vector_get(x, i);
			double yval = gsl_vector_get(y, i);

			double rel_error = 0;
			double denom = abs(xval) - abs(yval);
			if (denom != 0) rel_error = abs(xval - yval) / denom;

			if (rel_error > max_re) max_re = rel_error;
			if (abs(xval - yval) > max_ae) max_ae = abs(xval - yval);
		}
	        printf("rel_error=%f, abs_error=%f, rtol=%f, atol=%f\n", max_re, max_ae, rtol, atol);
	}
	return is_close;
}

// Check input gradient functions
double fun (void* layer, gsl_matrix*** g, gsl_matrix*** x, int type)
{
	gsl_matrix*** y = NULL;
	int size1 = 0;
	int size2 = 0;

	if (type == 1)
	{
		CONV conv_copy;
		copy_CONV(&conv_copy, (CONV*) layer);

		// assign X to the conv_copy.X
		if (g != NULL)
			for (int f = 0; f < conv_copy.n_filters; f++)
				for (int c = 0; c < conv_copy.n_channels; c++)
					gsl_matrix_memcpy(conv_copy.W[f][c], g[f][c]);

		y = forward_conv(&conv_copy, x);

		size1 = conv_copy.batch_size;
		size2 = conv_copy.n_filters;

		free_CONV(&conv_copy);
	}
	else if (type == 2)
	{
		POOL pool_copy;
		copy_POOL(&pool_copy, (POOL*) layer);

		y = forward_pool(&pool_copy, x);

		size1 = pool_copy.batch_size;
		size2 = pool_copy.n_channels;

		free_POOL(&pool_copy);
	}

	double saux = 0;
	for (int b = 0; b < size1; b++)
	{
		for (int f = 0; f < size2; f++)
		{
			int out_h = y[b][f]->size1;
			int out_w = y[b][f]->size2;
			for (int h = 0; h < out_h; h++)
				for (int w = 0; w < out_w; w++)
					saux += gsl_matrix_get(y[b][f], h, w);
			gsl_matrix_free(y[b][f]);
		}
		free(y[b]);
	}
	free(y);

	return saux;
}

gsl_matrix*** approx_fprime (void* layer, gsl_matrix*** x, gsl_matrix*** x0, double eps, int size01, int size02, int type)
{
	if (eps == -1) eps = 1.4901161193847656e-08;

	int img_h = x[0][0]->size1;
	int img_w = x[0][0]->size2;

	// Copy X matrix and create grad
	gsl_matrix*** x1 = (gsl_matrix***) malloc(size01 * sizeof(gsl_matrix**));
	gsl_matrix*** x2 = (gsl_matrix***) malloc(size01 * sizeof(gsl_matrix**));
	gsl_matrix*** grad = (gsl_matrix***) malloc(size01 * sizeof(gsl_matrix**));
	for (int i = 0; i < size01; i++)
	{
		x1[i] = (gsl_matrix**) malloc(size02 * sizeof(gsl_matrix*));
		x2[i] = (gsl_matrix**) malloc(size02 * sizeof(gsl_matrix*));
		grad[i] = (gsl_matrix**) malloc(size02 * sizeof(gsl_matrix*));
		for (int j = 0; j < size02; j++)
		{
			x1[i][j] = gsl_matrix_alloc(img_h, img_w);
			x2[i][j] = gsl_matrix_alloc(img_h, img_w);
			gsl_matrix_memcpy(x1[i][j], x[i][j]);
			gsl_matrix_memcpy(x2[i][j], x[i][j]);
			grad[i][j] = gsl_matrix_calloc(img_h, img_w);
		}
	}

	// Compute gradient for each point in X
	for (int i = 0; i < size01; i++)
		for (int j = 0; j < size02; j++)
			for (int h = 0; h < img_h; h++)
				for (int w = 0; w < img_w; w++)
				{
					double xval = gsl_matrix_get(x[i][j], h, w);
					double step = eps * max(abs(xval), 1.0);

					gsl_matrix_set(x1[i][j], h, w, xval + step);
					gsl_matrix_set(x2[i][j], h, w, xval - step);

					double aux1 = 0;
					double aux2 = 0;
					if (x0 == NULL)
					{
						aux1 = fun(layer, NULL, x1, type);
						aux2 = fun(layer, NULL, x2, type);
					} else {
						aux1 = fun(layer, x1, x0, type);
						aux2 = fun(layer, x2, x0, type);
					}

					double g = (aux1 - aux2) / (2 * step);

					gsl_matrix_set(x1[i][j], h, w, xval);
					gsl_matrix_set(x2[i][j], h, w, xval);
					gsl_matrix_set(grad[i][j], h, w, g);
				}

	// Free structures
	for (int i = 0; i < size01; i++)
	{
		for (int j = 0; j < size02; j++)
		{
			gsl_matrix_free(x1[i][j]);
			gsl_matrix_free(x2[i][j]);
		}
		free(x1[i]);
		free(x2[i]);
	}
	free(x1);
	free(x2);

	return grad;
}

gsl_matrix*** fun_grad (void* layer, gsl_matrix*** x, int get_W, int type)
{
	gsl_matrix*** y = NULL;
	int ygrad_i = 0;
	int ygrad_j = 0;

	CONV conv_copy;
	POOL pool_copy;

	if (type == 1)
	{
		// Make a copy of layer
		copy_CONV(&conv_copy, (CONV*) layer);

		// Go forward
		y = forward_conv(&conv_copy, x);

		// Get sizes
		ygrad_i = conv_copy.batch_size;
		ygrad_j = conv_copy.n_filters;
	}
	else if (type == 2)
	{
		// Make a copy of layer
		copy_POOL(&pool_copy, (POOL*) layer);

		// Go forward
		y = forward_pool(&pool_copy, x);

		// Get sizes
		ygrad_i = pool_copy.batch_size;
		ygrad_j = pool_copy.n_channels;
	}
	int ygrad_h = y[0][0]->size1;
	int ygrad_w = y[0][0]->size2;

	// Compute gradients of param
	gsl_matrix*** ygrad = (gsl_matrix***) malloc(ygrad_i * sizeof(gsl_matrix**));
	for (int i = 0; i < ygrad_i; i++)
	{
		ygrad[i] = (gsl_matrix**) malloc(ygrad_j * sizeof(gsl_matrix*));
		for (int j = 0; j < ygrad_j; j++)
		{
			ygrad[i][j] = gsl_matrix_alloc(ygrad_h, ygrad_w);
			gsl_matrix_set_all(ygrad[i][j], 1.0);
		}
	}

	// Go backward
	gsl_matrix*** retval = NULL;
	if (type == 1)
	{
		gsl_matrix*** xgrad = backward_conv(&conv_copy, ygrad);

		if (get_W == 1)
		{
			// Rescue grad_W and return
			int win_h = conv_copy.W[0][0]->size1;
			int win_w = conv_copy.W[0][0]->size2;
			gsl_matrix*** grad_W = (gsl_matrix***) malloc(conv_copy.n_filters * sizeof(gsl_matrix**));
			for (int f = 0; f < conv_copy.n_filters; f++)
			{
				grad_W[f] = (gsl_matrix**) malloc(conv_copy.n_channels * sizeof(gsl_matrix*));
				for (int c = 0; c < conv_copy.n_channels; c++)
				{
					grad_W[f][c] = gsl_matrix_calloc(win_h, win_w);
					gsl_matrix_memcpy(grad_W[f][c], conv_copy.grad_W[f][c]);
				}
			}

			// Free xgrad
			int xgrad_i = conv_copy.batch_size;
			int xgrad_j = conv_copy.n_channels;
			for (int i = 0; i < xgrad_i; i++)
			{
				for (int j = 0; j < xgrad_j; j++)
					gsl_matrix_free(xgrad[i][j]);
				free(xgrad[i]);
			}
			free(xgrad);

			retval = grad_W;
		} else {
			retval = xgrad;
		}

		// Free conv_copy
		free_CONV(&conv_copy);
	}
	else if (type == 2)
	{
		gsl_matrix*** xgrad = backward_pool(&pool_copy, ygrad);
		retval = xgrad;

		// Free pool_copy
		free_POOL(&pool_copy);
	}

	// Free y and ygrad
	for (int i = 0; i < ygrad_i; i++)
	{
		for (int j = 0; j < ygrad_j; j++)
		{
			gsl_matrix_free(y[i][j]);
			gsl_matrix_free(ygrad[i][j]);
		}
		free(y[i]);
		free(ygrad[i]);
	}
	free(y);
	free(ygrad);

	return retval;
}

/*---------------------------------------------------------------------------*/
/* CHECK PARAMETER GRADIENT B                                                */
/*---------------------------------------------------------------------------*/

double fun_b_conv (CONV* conv, gsl_vector* g, gsl_matrix*** x0)
{
	CONV conv_copy;
	copy_CONV(&conv_copy, conv);

	gsl_vector_memcpy(conv_copy.b, g);

	gsl_matrix*** y = forward_conv(&conv_copy, x0);

	int out_h = y[0][0]->size1;
	int out_w = y[0][0]->size2;
	double saux = 0;
	for (int b = 0; b < conv_copy.batch_size; b++)
		for (int f = 0; f < conv_copy.n_filters; f++)
			for (int h = 0; h < out_h; h++)
				for (int w = 0; w < out_w; w++)
					saux += gsl_matrix_get(y[b][f], h, w);

	for (int b = 0; b < conv_copy.batch_size; b++)
	{
		for (int f = 0; f < conv_copy.n_filters; f++) gsl_matrix_free(y[b][f]);
		free(y[b]);
	}
	free(y);

	free_CONV(&conv_copy);

	return saux;
}

gsl_vector* approx_fprime_b_conv (CONV* conv, gsl_matrix*** x0, double eps)
{
	if (eps == -1) eps = 1.4901161193847656e-08;

	gsl_vector* x1 = gsl_vector_calloc(conv->n_filters);
	gsl_vector* x2 = gsl_vector_calloc(conv->n_filters);
	gsl_vector_memcpy(x1, conv->b);
	gsl_vector_memcpy(x2, conv->b);
	gsl_vector* grad = gsl_vector_calloc(conv->n_filters);

	// Compute gradient for each point in X
	for (int f = 0; f < conv->n_filters; f++)
	{
		double xval = gsl_vector_get(conv->b, f);
		double step = eps * max(abs(xval), 1.0);

		gsl_vector_set(x1, f, xval + step);
		gsl_vector_set(x2, f, xval - step);

		double aux1 = fun_b_conv (conv, x1, x0);
		double aux2 = fun_b_conv (conv, x2, x0);

		double g = (aux1 - aux2) / (2 * step);

		gsl_vector_set(x1, f, xval);
		gsl_vector_set(x2, f, xval);
		gsl_vector_set(grad, f, g);
	}

	// Free structures
	gsl_vector_free(x1);
	gsl_vector_free(x2);

	return grad;
}

gsl_vector* fun_grad_b_conv (CONV* conv, gsl_matrix*** x0)
{
	// Make a copy of layer
	CONV conv_copy;
	copy_CONV(&conv_copy, conv);

	// Go forward
	gsl_matrix*** y = forward_conv(&conv_copy, x0);

	int ygrad_i = conv->batch_size;
	int ygrad_j = conv->n_filters;
	int ygrad_h = y[0][0]->size1;
	int ygrad_w = y[0][0]->size2;

	// Compute gradients of param
	gsl_matrix*** ygrad = (gsl_matrix***) malloc(ygrad_i * sizeof(gsl_matrix**));
	for (int i = 0; i < ygrad_i; i++)
	{
		ygrad[i] = (gsl_matrix**) malloc(ygrad_j * sizeof(gsl_matrix*));
		for (int j = 0; j < ygrad_j; j++)
		{
			ygrad[i][j] = gsl_matrix_alloc(ygrad_h, ygrad_w);
			gsl_matrix_set_all(ygrad[i][j], 1.0);
		}
	}

	// Go backward
	gsl_matrix*** xgrad = backward_conv(&conv_copy, ygrad);

	// Rescue grad_b and return
	gsl_vector* grad_b = gsl_vector_calloc(conv_copy.n_filters);
	gsl_vector_memcpy(grad_b, conv_copy.grad_b);

	// Free y, ygrad and xgrad
	for (int i = 0; i < ygrad_i; i++)
	{
		for (int j = 0; j < ygrad_j; j++)
		{
			gsl_matrix_free(y[i][j]);
			gsl_matrix_free(ygrad[i][j]);
		}
		free(y[i]);
		free(ygrad[i]);
	}
	free(y);
	free(ygrad);

	int xgrad_i = conv->batch_size;
	int xgrad_j = conv->n_channels;
	for (int i = 0; i < xgrad_i; i++)
	{
		for (int j = 0; j < xgrad_j; j++)
			gsl_matrix_free(xgrad[i][j]);
		free(xgrad[i]);
	}
	free(xgrad);

	// Free conv_copy
	free_CONV(&conv_copy);

	return grad_b;
}

/*---------------------------------------------------------------------------*/
/* MAIN CHECK FUNCTION                                                       */
/*---------------------------------------------------------------------------*/

// Return: 0 means OK, 1 means KO
int check_grad_conv (CONV* conv, gsl_matrix*** x0, int rand_seed, double eps, double rtol, double atol)
{
	srand(rand_seed);

	int retval = 0;

	// Part 1: Checking Output
	gsl_matrix*** g_approx = approx_fprime((void*) conv, x0, NULL, eps, conv->batch_size, conv->n_channels, 1);
	gsl_matrix*** g_true = fun_grad((void*) conv, x0, 0, 1);
	if (gradclose(g_approx, g_true, conv->n_filters, conv->n_channels, rtol, atol) == 0)
	{
		printf("Incorrect Input Gradient:\n * Approx:\n");
		for (int f = 0; f < conv->n_filters; f++)
			for (int c = 0; c < conv->n_channels; c++)
			{
				int out_h = g_approx[f][c]->size1;
				int out_w = g_approx[f][c]->size2;

				for (int h = 0; h < out_h; h++)
				{
					for (int w = 0; w < out_w; w++)
						printf("%f ", gsl_matrix_get(g_approx[f][c], h, w));
					printf("\n");
				}
			}
		printf(" * True:\n");
		for (int f = 0; f < conv->n_filters; f++)
			for (int c = 0; c < conv->n_channels; c++)
			{
				int out_h = g_true[f][c]->size1;
				int out_w = g_true[f][c]->size2;
				for (int h = 0; h < out_h; h++)
				{
					for (int w = 0; w < out_w; w++)
						printf("%f ", gsl_matrix_get(g_true[f][c], h, w));
					printf("\n");
				}
			}
		retval = 1;
	}

	// Free auxiliar structures
	for (int b = 0; b < conv->batch_size; b++)
	{
		for (int c = 0; c < conv->n_channels; c++)
		{
			gsl_matrix_free(g_approx[b][c]);
			gsl_matrix_free(g_true[b][c]);
		}
		free(g_approx[b]);
		free(g_true[b]);
	}
	free(g_approx);
	free(g_true);

	// Part 2: Checking Gradient Params

	// Check grad_W
	gsl_matrix*** g_approx_W = approx_fprime((void*) conv, conv->W, x0, eps, conv->n_filters, conv->n_channels, 1);
	gsl_matrix*** g_true_W = fun_grad((void*) conv, x0, 1, 1);
	if (gradclose(g_approx_W, g_true_W, conv->n_filters, conv->n_channels, rtol, atol) == 0)
	{
		printf("Incorrect Parameter W Gradient:\n * Approx:\n");
		printf("------------------------------------------\n");
		for (int f = 0; f < conv->n_filters; f++)
			for (int c = 0; c < conv->n_channels; c++)
			{
				int out_h = g_approx_W[f][c]->size1;
				int out_w = g_approx_W[f][c]->size2;
				for (int h = 0; h < out_h; h++)
				{
					for (int w = 0; w < out_w; w++)
						printf("%f ", gsl_matrix_get(g_approx_W[f][c], h, w));
					printf("\n");
				}
				printf("------------------------------------------\n");
			}
		printf(" * True:\n");
		printf("------------------------------------------\n");
		for (int f = 0; f < conv->n_filters; f++)
			for (int c = 0; c < conv->n_channels; c++)
			{
				int out_h = g_true_W[f][c]->size1;
				int out_w = g_true_W[f][c]->size2;
				for (int h = 0; h < out_h; h++)
				{
					for (int w = 0; w < out_w; w++)
						printf("%f ", gsl_matrix_get(g_true_W[f][c], h, w));
					printf("\n");
				}
				printf("------------------------------------------\n");
			}
		retval = 1;
	}

	// Check grad_b
	gsl_vector* g_approx_b = approx_fprime_b_conv(conv, x0, eps);
	gsl_vector* g_true_b = fun_grad_b_conv (conv, x0);
	if (gradclose_vec(g_approx_b, g_true_b, conv->n_filters, rtol, atol) == 0)
	{
		printf("Incorrect Parameter b Gradient:\n * Approx:\n");
		printf("------------------------------------------\n");
		for (int f = 0; f < conv->n_filters; f++)
			printf("%f ", gsl_vector_get(g_approx_b, f));
		printf("\n");
		printf("------------------------------------------\n");
		printf(" * True:\n");
		printf("------------------------------------------\n");
		for (int f = 0; f < conv->n_filters; f++)
			printf("%f ", gsl_vector_get(g_true_b, f));
		printf("\n");
		printf("------------------------------------------\n");
	}

	// Free auxiliar structures
	for (int f = 0; f < conv->n_filters; f++)
	{
		for (int c = 0; c < conv->n_channels; c++)
		{
			gsl_matrix_free(g_true_W[f][c]);
			gsl_matrix_free(g_approx_W[f][c]);
		}
		free(g_true_W[f]);
		free(g_approx_W[f]);
	}
	free(g_true_W);
	free(g_approx_W);

	gsl_vector_free(g_approx_b);
	gsl_vector_free(g_true_b);

	return retval;
}

// Return: 0 means OK, 1 means KO
int check_grad_pool (POOL* pool, gsl_matrix*** x0, int rand_seed, double eps, double rtol, double atol)
{
	srand(rand_seed);

	int retval = 0;

	// Part 1: Checking Output
	gsl_matrix*** g_approx = approx_fprime((void*) pool, x0, NULL, eps, pool->batch_size, pool->n_channels, 2);
	gsl_matrix*** g_true = fun_grad((void*) pool, x0, 0, 2);
	if (gradclose(g_approx, g_true, pool->batch_size, pool->n_channels, rtol, atol) == 0)
	{
		printf("Incorrect Input Gradient:\n * Approx:\n");
		for (int f = 0; f < pool->batch_size; f++)
			for (int c = 0; c < pool->n_channels; c++)
			{
				int out_h = g_approx[f][c]->size1;
				int out_w = g_approx[f][c]->size2;

				for (int h = 0; h < out_h; h++)
				{
					for (int w = 0; w < out_w; w++)
						printf("%f ", gsl_matrix_get(g_approx[f][c], h, w));
					printf("\n");
				}
			}
		printf(" * True:\n");
		for (int f = 0; f < pool->batch_size; f++)
			for (int c = 0; c < pool->n_channels; c++)
			{
				int out_h = g_true[f][c]->size1;
				int out_w = g_true[f][c]->size2;
				for (int h = 0; h < out_h; h++)
				{
					for (int w = 0; w < out_w; w++)
						printf("%f ", gsl_matrix_get(g_true[f][c], h, w));
					printf("\n");
				}
			}
		retval = 1;
	}

	// Free auxiliar structures
	for (int b = 0; b < pool->batch_size; b++)
	{
		for (int c = 0; c < pool->n_channels; c++)
		{
			gsl_matrix_free(g_approx[b][c]);
			gsl_matrix_free(g_true[b][c]);
		}
		free(g_approx[b]);
		free(g_true[b]);
	}
	free(g_approx);
	free(g_true);

	return retval;
}

// Return: 0 means OK, 1 means KO
int check_grad_flat (FLAT* flat, gsl_matrix*** x0, int rand_seed, double eps, double rtol, double atol)
{
	srand(rand_seed);

	int retval = 0;

	// Part 1: Checking Output
	FLAT flat_copy;
	copy_FLAT(&flat_copy, flat);

	// Go forward
	gsl_matrix* y = forward_flat(&flat_copy, x0);

	// Go backward
	gsl_matrix*** dx = backward_flat(&flat_copy, y);

	// Check output reconstruction
	int is_equal = 0;
	for (int b = 0; b < flat->batch_size; b++)
		for (int c = 0; c < flat->n_channels; c++)
			is_equal += 1 - gsl_matrix_equal(x0[b][c], dx[b][c]);

	if (is_equal > 0)
	{
		printf("Incorrect Input Gradient:\n * Differences: %d\n", is_equal);
		retval = 1;
	}

	// Free Auxiliar Structures
	for (int i = 0; i < flat->batch_size; i++)
	{
		for (int j = 0; j < flat->n_channels; j++)
			gsl_matrix_free(dx[i][j]);
		free(dx[i]);
	}
	gsl_matrix_free(y);
	free(dx);

	// Free flat_copy
	free_FLAT(&flat_copy);

	return retval;
}

int check_grad_relu (RELU* relu, gsl_matrix*** x0, int rand_seed, double eps, double rtol, double atol)
{
	srand(rand_seed);

	int retval = 0;

	// Part 1: Checking Output
	RELU relu_copy;
	copy_RELU(&relu_copy, relu);

	// Go forward
	gsl_matrix*** y = forward_relu(&relu_copy, x0);

	// Go backward
	gsl_matrix*** dx = backward_relu(&relu_copy, y);

	// Check output reconstruction
	int img_h = x0[0][0]->size1;
	int img_w = x0[0][0]->size2;

	int is_equal = 0;
	for (int b = 0; b < relu->batch_size; b++)
		for (int c = 0; c < relu->n_channels; c++)
			for (int h = 0; h < img_h; h++)
				for (int w = 0; w < img_w; w++)
				{
					double value0 = gsl_matrix_get(x0[b][c], h, w);
					double value1 = gsl_matrix_get(dx[b][c], h, w);
					if ((value0 < 0 && value1 != 0) || (value0 >= 0 && value0 != value1)) is_equal += 1;
				}

	if (is_equal > 0)
	{
		printf("Incorrect Input Gradient:\n * Differences: %d\n", is_equal);
		retval = 1;
	}

	// Free auxiliar structures
	for (int b = 0; b < relu->batch_size; b++)
	{
		for (int c = 0; c < relu->n_channels; c++)
		{
			gsl_matrix_free(y[b][c]);
			gsl_matrix_free(dx[b][c]);
		}
		free(y[b]);
		free(dx[b]);
	}
	free(y);
	free(dx);

	free_RELU(&relu_copy);

	return retval;
}

